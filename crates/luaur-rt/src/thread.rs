//! The [`Thread`] handle and [`ThreadStatus`]. Mirrors `mlua::Thread` /
//! `mlua::ThreadStatus`.
//!
//! A thread is a Luau coroutine. It is created from a [`Function`] via
//! [`Lua::create_thread`] (or surfaces from `coroutine.create(...)` evaluated
//! in Lua) and driven with [`Thread::resume`].
//!
//! ## Implementation
//!
//! The thread is a first-class Lua value, so the handle holds a registry
//! reference (like every other handle) keeping the coroutine alive. We also
//! cache the raw `*mut lua_State` of the coroutine for the resume/xmove dance.
//!
//! `resume` mirrors mlua: push the args onto the *parent* state, `lua_xmove`
//! them to the coroutine, `lua_resume(co, parent, nargs)`, then `lua_xmove` the
//! results back and convert them. Status is derived from `lua_status` +
//! `lua_costatus`, matching mlua's `Resumable`/`Running`/`Normal`/`Finished`/
//! `Error` mapping.

use crate::error::{Error, Result};
use crate::function::Function;
use crate::multi::MultiValue;
use crate::state::{Lua, LuaRef};
use crate::sync::XRc;
use crate::sys::*;
use crate::traits::{FromLuaMulti, IntoLua, IntoLuaMulti};

/// Status of a Lua thread (coroutine). Mirrors `mlua::ThreadStatus`.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ThreadStatus {
    /// The thread was just created or is suspended (yielded) and can be resumed.
    Resumable,
    /// The thread is currently running.
    Running,
    /// The thread is active but not running (it resumed another thread).
    Normal,
    /// The thread has finished executing.
    Finished,
    /// The thread raised a Lua error during execution.
    Error,
}

/// The raw outcome of one async resume. See [`Thread::resume_for_async`].
#[cfg(feature = "async")]
pub(crate) enum AsyncResume {
    /// The coroutine yielded the internal "future pending" marker.
    Pending,
    /// The coroutine yielded values via `coroutine.yield` (a Stream item).
    Yielded(MultiValue),
    /// The coroutine finished, returning these values.
    Returned(MultiValue),
}

/// A handle to a Lua thread (coroutine). Mirrors `mlua::Thread`.
///
/// Under the `send` feature it is `Send + Sync` (all VM access is
/// serialized by the per-VM lock — see [`crate::state::StateRef`]; the
/// coroutine shares its parent VM's lock, since the lock is keyed by the
/// shared `global_State`).
#[derive(Clone)]
pub struct Thread {
    pub(crate) reference: XRc<LuaRef>,
    /// The raw coroutine state pointer (cached from the referenced value).
    pub(crate) thread_state: *mut lua_State,
}

// `Thread` caches a raw `*mut lua_State` (the coroutine), which is `!Send` /
// `!Sync` by default. Sound because the cached pointer is only dereferenced
// through code paths that hold the per-VM lock (the coroutine shares its
// parent's `global_State`, and the lock is keyed by that pointer); the public
// `Thread::state()` accessor merely *returns* the pointer, and using it is the
// caller's unsafe responsibility, exactly like `mlua::Thread::state`.
#[cfg(feature = "send")]
unsafe impl Send for Thread {}
#[cfg(feature = "send")]
unsafe impl Sync for Thread {}

impl Thread {
    /// Build a [`Thread`] from a registry ref to a thread value. Caches the
    /// coroutine's raw state via `lua_tothread`.
    pub(crate) fn from_ref(reference: LuaRef) -> Thread {
        let thread_state = {
            let state = reference.state();
            let state = state.get();
            unsafe {
                reference.push();
                let ts = lua_tothread(state, -1);
                lua_pop(state, 1);
                ts
            }
        };
        Thread {
            reference: XRc::new(reference),
            thread_state,
        }
    }

    pub(crate) unsafe fn push_to_stack(&self) {
        self.reference.push();
    }

    /// The owning [`Lua`].
    pub fn lua(&self) -> Lua {
        self.reference.lua()
    }

    /// The raw coroutine state pointer. Mirrors `mlua::Thread::state`.
    pub fn state(&self) -> *mut lua_State {
        self.thread_state
    }

    /// Resume the coroutine, passing `args` and converting its yielded/returned
    /// values to `R`. Mirrors `mlua::Thread::resume`.
    ///
    /// Returns [`Error::CoroutineUnresumable`] if the thread has finished,
    /// errored, or is otherwise not resumable.
    pub fn resume<R: FromLuaMulti>(&self, args: impl IntoLuaMulti) -> Result<R> {
        let lua = self.lua();
        // Convert the args first so a failing `IntoLua` (e.g. a bad argument)
        // surfaces *before* we touch any Lua stack — matching mlua.
        let args: MultiValue = args.into_lua_multi(&lua)?;

        if !matches!(self.status(), ThreadStatus::Resumable) {
            return Err(Error::CoroutineUnresumable);
        }

        let parent = lua.state();
        let parent = parent.get();
        let co = self.thread_state;
        unsafe {
            let nargs = args.len() as c_int;
            if lua_checkstack(co, nargs.saturating_add(2)) == 0 {
                return Err(Error::RuntimeError(
                    "stack overflow: too many arguments to coroutine resume".to_string(),
                ));
            }
            // Push args onto the parent, then move them to the coroutine.
            for v in args.iter() {
                lua.push_value(v)?;
            }
            if nargs > 0 {
                lua_xmove(parent, co, nargs);
            }

            self.resume_inner::<R>(&lua, nargs)
        }
    }

    /// Resume the coroutine, immediately raising `error` inside it.
    /// Mirrors `mlua::Thread::resume_error` (a Luau extension).
    pub fn resume_error<R: FromLuaMulti>(&self, error: impl IntoLua) -> Result<R> {
        let lua = self.lua();
        let err_value = error.into_lua(&lua)?;

        if !matches!(self.status(), ThreadStatus::Resumable) {
            return Err(Error::CoroutineUnresumable);
        }

        let parent = lua.state();
        let parent = parent.get();
        let co = self.thread_state;
        unsafe {
            if lua_checkstack(co, 2) == 0 {
                return Err(Error::RuntimeError("stack overflow".to_string()));
            }
            lua.push_value(&err_value)?;
            lua_xmove(parent, co, 1);
            // lua_resumeerror does the resume-with-error and returns the status.
            let status = lua_resumeerror(co, parent);
            self.finish_resume::<R>(&lua, status)
        }
    }

    /// Run `lua_resume` and collect/convert the results. Expects `nargs` already
    /// moved onto the coroutine stack.
    unsafe fn resume_inner<R: FromLuaMulti>(&self, lua: &Lua, nargs: c_int) -> Result<R> {
        let parent = lua.state();
        let parent = parent.get();
        let co = self.thread_state;
        let status = unsafe { lua_resume(co, parent, nargs) };
        unsafe { self.finish_resume::<R>(lua, status) }
    }

    /// Common tail of `resume`/`resume_error`: inspect the status, move results
    /// back to the parent, and convert.
    unsafe fn finish_resume<R: FromLuaMulti>(&self, lua: &Lua, status: c_int) -> Result<R> {
        let parent = lua.state();
        let parent = parent.get();
        let co = self.thread_state;
        unsafe {
            if status != status::OK && status != status::YIELD && status != status::BREAK {
                // Error: the coroutine left the error object on its own stack.
                let nres = lua_gettop(co);
                if nres > 0 {
                    lua_xmove(co, parent, nres);
                }
                let err = lua.pop_error(status);
                // Clear any extra values the coroutine left on the parent.
                return Err(err);
            }
            // `LUA_BREAK` is an interrupt-driven yield: the coroutine produced
            // no values and its entire register window is still *live* (it must
            // continue from the break point on the next resume). We must NOT
            // touch its stack — moving any values off would strip live
            // registers and corrupt the re-entry. Return an empty result.
            if status == status::BREAK {
                return R::from_lua_multi(MultiValue::with_capacity(0), lua);
            }
            // Success/yield: the produced values sit on the coroutine stack.
            let nres = lua_gettop(co);
            if lua_checkstack(parent, nres.saturating_add(1)) == 0 {
                return Err(Error::RuntimeError("stack overflow".to_string()));
            }
            let base = lua_gettop(parent);
            if nres > 0 {
                lua_xmove(co, parent, nres);
            }
            let mut results = MultiValue::with_capacity(nres.max(0) as usize);
            for i in 0..nres {
                results.push_back(lua.value_from_stack(base + 1 + i)?);
            }
            lua_settop(parent, base);
            R::from_lua_multi(results, lua)
        }
    }

    /// Low-level resume used by the async driver. Pushes `args` to the
    /// coroutine, resumes it once, and returns the raw outcome:
    ///
    /// * `Err(e)` — the coroutine raised an error.
    /// * `Ok(AsyncResume::Pending)` — the coroutine yielded the internal
    ///   "future pending" marker (a single light-userdata == `poll_pending()`);
    ///   the stack is left cleared.
    /// * `Ok(AsyncResume::Yielded(vals))` — the coroutine yielded `vals`
    ///   (a `coroutine.yield`, i.e. a Stream item).
    /// * `Ok(AsyncResume::Returned(vals))` — the coroutine finished, returning
    ///   `vals`.
    ///
    /// The coroutine stack is fully consumed/cleared on every path.
    #[cfg(feature = "async")]
    pub(crate) fn resume_for_async(&self, args: MultiValue) -> Result<AsyncResume> {
        let lua = self.lua();
        let parent = lua.state();
        let parent = parent.get();
        let co = self.thread_state;
        unsafe {
            let nargs = args.len() as c_int;
            if lua_checkstack(co, nargs.saturating_add(2)) == 0 {
                return Err(Error::RuntimeError(
                    "stack overflow: too many arguments to coroutine resume".to_string(),
                ));
            }
            for v in args.iter() {
                lua.push_value(v)?;
            }
            if nargs > 0 {
                lua_xmove(parent, co, nargs);
            }
            let status = lua_resume(co, parent, nargs);

            if status != status::OK && status != status::YIELD {
                let nres = lua_gettop(co);
                if nres > 0 {
                    lua_xmove(co, parent, nres);
                }
                return Err(lua.pop_error(status));
            }

            let yielded = status == status::YIELD;
            let nres = lua_gettop(co);

            // Detect the single-light-userdata pending marker (top of the
            // coroutine stack) on a yield.
            if yielded
                && nres == 1
                && crate::sys::lua_tolightuserdata(co, -1) == crate::async_support::poll_pending()
            {
                lua_settop(co, 0);
                return Ok(AsyncResume::Pending);
            }

            // Otherwise move the produced values to the parent and convert.
            if lua_checkstack(parent, nres.saturating_add(1)) == 0 {
                return Err(Error::RuntimeError("stack overflow".to_string()));
            }
            let base = lua_gettop(parent);
            if nres > 0 {
                lua_xmove(co, parent, nres);
            }
            let mut results = MultiValue::with_capacity(nres.max(0) as usize);
            for i in 0..nres {
                results.push_back(lua.value_from_stack(base + 1 + i)?);
            }
            lua_settop(parent, base);
            lua_settop(co, 0);

            if yielded {
                Ok(AsyncResume::Yielded(results))
            } else {
                Ok(AsyncResume::Returned(results))
            }
        }
    }

    /// Resume a yielded async coroutine with the "terminate" signal so it drops
    /// its in-flight future and parks. Best-effort; ignores errors. Used when an
    /// [`AsyncThread`](crate::async_support::AsyncThread) is dropped mid-flight.
    #[cfg(feature = "async")]
    pub(crate) fn terminate_async(&self) {
        if !self.is_resumable() {
            return;
        }
        let lua = self.lua();
        let parent = lua.state();
        let parent = parent.get();
        let co = self.thread_state;
        unsafe {
            if lua_checkstack(co, 2) == 0 {
                return;
            }
            crate::sys::lua_pushlightuserdatatagged(
                parent,
                crate::async_support::poll_terminate(),
                0,
            );
            lua_xmove(parent, co, 1);
            let _ = lua_resume(co, parent, 1);
            lua_settop(co, 0);
        }
    }

    /// The thread's status. Mirrors `mlua::Thread::status`.
    pub fn status(&self) -> ThreadStatus {
        let lua = self.lua();
        let parent = lua.state();
        let parent = parent.get();
        let co = self.thread_state;
        // A thread whose state is the currently-running state is "Running".
        if co == parent {
            return ThreadStatus::Running;
        }
        unsafe {
            // A coroutine yielded by an interrupt (`lua_break`) has raw status
            // `LUA_BREAK`; `lua_costatus` reports that as "normal", but the
            // coroutine is in fact resumable (it continues from the break point
            // on the next resume). Detect it directly.
            if lua_status(co) == status::BREAK {
                return ThreadStatus::Resumable;
            }
            let cos = lua_costatus(parent, co);
            match cos {
                costatus::SUSPENDED => ThreadStatus::Resumable,
                costatus::RUNNING => ThreadStatus::Running,
                costatus::NORMAL => ThreadStatus::Normal,
                costatus::FINISHED => ThreadStatus::Finished,
                costatus::ERROR => ThreadStatus::Error,
                _ => {
                    // Fall back to lua_status for any unexpected code.
                    let s = lua_status(co);
                    if s == status::YIELD {
                        ThreadStatus::Resumable
                    } else if s == status::OK {
                        // New (function on stack) vs finished (empty stack).
                        if lua_gettop(co) > 0 {
                            ThreadStatus::Resumable
                        } else {
                            ThreadStatus::Finished
                        }
                    } else {
                        ThreadStatus::Error
                    }
                }
            }
        }
    }

    /// Whether the thread can be resumed. Mirrors `mlua::Thread::is_resumable`.
    pub fn is_resumable(&self) -> bool {
        self.status() == ThreadStatus::Resumable
    }

    /// Whether the thread is currently running. Mirrors `mlua::Thread::is_running`.
    pub fn is_running(&self) -> bool {
        self.status() == ThreadStatus::Running
    }

    /// Whether the thread is active but not running. Mirrors
    /// `mlua::Thread::is_normal`.
    pub fn is_normal(&self) -> bool {
        self.status() == ThreadStatus::Normal
    }

    /// Whether the thread has finished executing. Mirrors
    /// `mlua::Thread::is_finished`.
    pub fn is_finished(&self) -> bool {
        self.status() == ThreadStatus::Finished
    }

    /// Whether the thread raised an error. Mirrors `mlua::Thread::is_error`.
    pub fn is_error(&self) -> bool {
        self.status() == ThreadStatus::Error
    }

    /// Reset the thread to a fresh state and install `func` as its body.
    /// Mirrors `mlua::Thread::reset` (Luau semantics: any non-running thread can
    /// be reset).
    pub fn reset(&self, func: Function) -> Result<()> {
        let status = self.status();
        match status {
            ThreadStatus::Running => {
                return Err(Error::runtime("cannot reset a running thread"));
            }
            ThreadStatus::Normal => {
                return Err(Error::runtime("cannot reset a normal thread"));
            }
            _ => {}
        }
        let lua = self.lua();
        let parent = lua.state();
        let parent = parent.get();
        let co = self.thread_state;
        unsafe {
            lua_resetthread(co);
            // Push the new body function onto the coroutine stack.
            func.push_to_stack();
            lua_xmove(parent, co, 1);
            // Re-inherit the *main* globals table into the coroutine, dropping
            // any sandbox proxy global a prior `Thread::sandbox` had installed
            // (matches mlua's Luau `reset`: a reset thread sees the main env).
            lua_pushvalue(parent, LUA_GLOBALSINDEX);
            lua_xmove(parent, co, 1);
            lua_replace(co, LUA_GLOBALSINDEX);
        }
        Ok(())
    }

    /// A raw pointer identifying this thread. Mirrors `mlua::Thread::to_pointer`.
    pub fn to_pointer(&self) -> *const c_void {
        let state = self.reference.state();
        let state = state.get();
        unsafe {
            self.reference.push();
            let p = lua_topointer(state, -1);
            lua_pop(state, 1);
            p
        }
    }
}

impl std::fmt::Debug for Thread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Thread")
    }
}

// ---------------------------------------------------------------------------
// Async: drive a coroutine as a Rust `Future` / `Stream` (the `async` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "async")]
impl Thread {
    /// Convert this (resumable) thread into an
    /// [`AsyncThread`](crate::AsyncThread) that implements
    /// [`Future`](std::future::Future) and
    /// [`Stream`](futures_util::stream::Stream).
    ///
    /// Mirrors `mlua::Thread::into_async`. `args` are passed to the coroutine on
    /// its first resume. As a `Future` the thread is driven to completion and
    /// resolves to its final return value(s); as a `Stream` each
    /// `coroutine.yield` produces an item.
    #[cfg_attr(docsrs, doc(cfg(feature = "async")))]
    pub fn into_async<R: FromLuaMulti>(
        self,
        args: impl IntoLuaMulti,
    ) -> Result<crate::async_support::AsyncThread<R>> {
        if !self.is_resumable() {
            return Err(Error::CoroutineUnresumable);
        }
        let lua = self.lua();
        let args = args.into_lua_multi(&lua)?;
        Ok(crate::async_support::AsyncThread::new(self, args))
    }
}

impl PartialEq for Thread {
    fn eq(&self, other: &Self) -> bool {
        self.to_pointer() == other.to_pointer()
    }
}

impl IntoLua for Thread {
    fn into_lua(self, _lua: &Lua) -> Result<crate::value::Value> {
        Ok(crate::value::Value::Thread(self))
    }
}

impl IntoLua for &Thread {
    fn into_lua(self, _lua: &Lua) -> Result<crate::value::Value> {
        Ok(crate::value::Value::Thread(self.clone()))
    }
}

impl FromLua for Thread {
    fn from_lua(value: crate::value::Value, _lua: &Lua) -> Result<Self> {
        match value {
            crate::value::Value::Thread(t) => Ok(t),
            other => Err(Error::FromLuaConversionError {
                from: other.type_name(),
                to: "Thread".to_string(),
                message: None,
            }),
        }
    }
}

use crate::traits::FromLua;

impl Lua {
    /// Create a new coroutine from a [`Function`]. Mirrors
    /// `mlua::Lua::create_thread`.
    pub fn create_thread(&self, func: Function) -> Result<Thread> {
        let state = self.state();
        let state = state.get();
        unsafe {
            // Create a new thread; it is pushed on the parent stack.
            let co = lua_newthread(state);
            if co.is_null() {
                return Err(Error::runtime("luaur-rt: failed to create thread"));
            }
            // Take a ref to the thread value (still on the parent stack top).
            let thread = Thread::from_ref(self.pop_ref());
            // Move the body function onto the coroutine's stack so the first
            // resume invokes it.
            func.push_to_stack(); // pushes onto parent stack
            lua_xmove(state, co, 1);
            Ok(thread)
        }
    }

    /// The currently-running thread. Mirrors `mlua::Lua::current_thread`.
    ///
    /// Inside a Rust callback this is the coroutine (or main thread) that
    /// invoked it. Under the `async` feature, a coroutine created implicitly by
    /// `call_async` is transparent: this returns its *owner* thread instead, so
    /// `current_thread()` is stable across the implicit-coroutine boundary
    /// (matching mlua).
    pub fn current_thread(&self) -> Thread {
        let state = self.state();
        let state = state.get();
        // If we are running on an implicit `call_async` coroutine, report the
        // owner thread that issued the call.
        #[cfg(feature = "async")]
        if let Some(owner) = crate::async_support::implicit_thread_owner(state) {
            unsafe {
                lua_pushthread(owner);
                // The owner-thread value is on the owner's stack; move it to this
                // state so we can take a ref to it from here.
                if owner != state {
                    lua_xmove(owner, state, 1);
                }
                return Thread::from_ref(self.pop_ref());
            }
        }
        unsafe {
            lua_pushthread(state);
            Thread::from_ref(self.pop_ref())
        }
    }
}
