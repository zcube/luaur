//! Luau interrupt support. Mirrors `mlua::Lua::set_interrupt` / `VmState`.
//!
//! Luau's VM calls a single global `interrupt` callback at safepoints (loop
//! back-edges, calls/returns, GC). mlua exposes this as `Lua::set_interrupt`,
//! taking a Rust closure that returns a [`VmState`] telling the VM whether to
//! continue or to **yield** the current coroutine.
//!
//! luaur's `lua_callbacks().interrupt` is a plain C function pointer, so we
//! install a fixed trampoline ([`interrupt_trampoline`]) and keep the Rust
//! closure in a per-VM store keyed by the VM's *global* pointer (shared by all
//! threads of one `Lua`) — thread-local without the `send` feature, a
//! process-global `RwLock` map with it (the VM may then run on any thread).
//! The trampoline looks up the closure, runs it with a borrowed [`Lua`], and:
//!
//! * `Ok(VmState::Continue)`  — returns normally; the VM keeps executing.
//! * `Ok(VmState::Yield)`     — calls `lua_break`, which sets the running
//!   thread's status so the VM unwinds back to `lua_resume` (a *yield* at a
//!   yieldable point; ignored otherwise, exactly like upstream Luau).
//! * `Err(e)`                 — raises `e` as a Lua error via `lua_error`.

use crate::error::{Error, Result};
use crate::state::Lua;
use crate::sync::XRc;
use crate::sys::*;

/// The action an interrupt callback asks the VM to take. Mirrors
/// `mlua::VmState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmState {
    /// Keep executing.
    Continue,
    /// Yield the currently running coroutine (no-op at a non-yieldable point).
    Yield,
}

/// The type-erased interrupt closure. `Send` under the `send` feature: the VM
/// (and hence the safepoint that invokes the closure) may run on any thread.
#[cfg(feature = "send")]
type InterruptFn = Box<dyn Fn(&Lua) -> Result<VmState> + Send + 'static>;
/// See the `send`-gated variant above.
#[cfg(not(feature = "send"))]
type InterruptFn = Box<dyn Fn(&Lua) -> Result<VmState> + 'static>;

/// The closure's shared slot. The trampoline clones the `XRc` out of the store
/// (read lock only) and invokes the clone, so a re-entrant `set_interrupt` /
/// `remove_interrupt` from inside the callback simply replaces the map entry —
/// no aliasing, no take-out/put-back dance.
struct InterruptSlot(InterruptFn);

// SAFETY (`send` only): the closure is only ever *invoked* from a VM safepoint,
// i.e. on a thread that is currently executing this VM and therefore holds the
// per-VM re-entrant lock; installs also run under that lock (`set_interrupt`
// takes `self.state()`). Invocations of one slot are thus serialized even
// though the `Arc` itself may be cloned/dropped on any thread, so sharing
// `&InterruptSlot` across threads never yields a concurrent call.
#[cfg(feature = "send")]
unsafe impl Sync for InterruptSlot {}

#[cfg(not(feature = "send"))]
mod interrupt_store {
    use super::{InterruptSlot, XRc};
    use std::cell::RefCell;
    use std::collections::HashMap;

    thread_local! {
        /// Per-VM interrupt closure, keyed by the `global_State` pointer
        /// (stable for the lifetime of the VM, shared by all of its threads).
        static INTERRUPTS: RefCell<HashMap<usize, XRc<InterruptSlot>>> =
            RefCell::new(HashMap::new());
    }

    pub(super) fn read<R>(f: impl FnOnce(&HashMap<usize, XRc<InterruptSlot>>) -> R) -> R {
        INTERRUPTS.with(|m| f(&m.borrow()))
    }

    pub(super) fn write<R>(f: impl FnOnce(&mut HashMap<usize, XRc<InterruptSlot>>) -> R) -> R {
        INTERRUPTS.with(|m| f(&mut m.borrow_mut()))
    }
}

#[cfg(feature = "send")]
mod interrupt_store {
    //! Under `send` the VM can run on any thread, so the closure must live in
    //! a process-global map. `RwLock`: the safepoint hot path only reads.
    use super::{InterruptSlot, XRc};
    use std::collections::HashMap;
    use std::sync::{LazyLock, RwLock};

    static INTERRUPTS: LazyLock<RwLock<HashMap<usize, XRc<InterruptSlot>>>> =
        LazyLock::new(|| RwLock::new(HashMap::new()));

    pub(super) fn read<R>(f: impl FnOnce(&HashMap<usize, XRc<InterruptSlot>>) -> R) -> R {
        let guard = INTERRUPTS.read().unwrap_or_else(|e| e.into_inner());
        f(&guard)
    }

    pub(super) fn write<R>(f: impl FnOnce(&mut HashMap<usize, XRc<InterruptSlot>>) -> R) -> R {
        let mut guard = INTERRUPTS.write().unwrap_or_else(|e| e.into_inner());
        f(&mut guard)
    }
}

/// The `global_State` pointer for `state` — the per-VM key shared by all
/// threads of one `Lua`.
unsafe fn vm_key(state: *mut lua_State) -> usize {
    unsafe { (*state).global as usize }
}

impl Lua {
    /// Install an interrupt callback. Mirrors `mlua::Lua::set_interrupt`.
    ///
    /// The callback runs at VM safepoints; returning [`VmState::Yield`] yields
    /// the running coroutine, and returning `Err` raises a Lua error.
    pub fn set_interrupt<F>(&self, callback: F)
    where
        F: Fn(&Lua) -> Result<VmState> + crate::sync::MaybeSend + 'static,
    {
        let state = self.state();
        let state = state.get();
        unsafe {
            let key = vm_key(state);
            interrupt_store::write(|m| {
                m.insert(key, XRc::new(InterruptSlot(Box::new(callback))));
            });
            let cb = lua_callbacks(state);
            (*cb).interrupt = Some(interrupt_trampoline);
        }
    }

    /// Remove a previously installed interrupt callback. Mirrors
    /// `mlua::Lua::remove_interrupt`.
    pub fn remove_interrupt(&self) {
        let state = self.state();
        let state = state.get();
        unsafe {
            let key = vm_key(state);
            interrupt_store::write(|m| {
                m.remove(&key);
            });
            let cb = lua_callbacks(state);
            (*cb).interrupt = None;
        }
    }
}

/// Drop this VM's interrupt closure. Called from `LuaInner::drop` so the closure
/// (and anything it captured) is released and the per-VM map entry does not leak
/// one slot per state created. (If a closure captured Lua handles it would pin
/// the VM and this never runs — but the common case captures non-Lua state.)
pub(crate) fn clear_interrupt(state: *mut lua_State) {
    let key = unsafe { vm_key(state) };
    interrupt_store::write(|m| {
        m.remove(&key);
    });
}

/// The fixed C trampoline installed as `lua_callbacks().interrupt`.
///
/// `gc` is non-negative only for GC interrupts; mlua ignores GC interrupts in
/// the user callback path, and so do we (return immediately) so the user
/// closure only sees real instruction safepoints.
unsafe extern "C-unwind" fn interrupt_trampoline(state: *mut lua_State, gc: c_int) {
    if gc >= 0 {
        // GC step interrupt — not surfaced to the user callback.
        return;
    }
    let key = unsafe { vm_key(state) };
    // Clone the slot out (read lock only) and invoke the clone: a re-entrant
    // `set_interrupt` inside the callback replaces the map entry without
    // touching this invocation.
    let cb = interrupt_store::read(|m| m.get(&key).cloned());
    let Some(cb) = cb else { return };

    let lua = unsafe { Lua::from_borrowed(state) };
    let result = (cb.0)(&lua);

    match result {
        Ok(VmState::Continue) => {}
        Ok(VmState::Yield) => unsafe {
            // Request a yield — but only at a yieldable point. Inside a
            // metamethod / C-call boundary Luau's `lua_break` would raise
            // "attempt to break across metamethod/C-call boundary"; upstream
            // (and mlua) silently ignore the yield request there, so we gate it
            // on `lua_isyieldable` and otherwise just continue.
            if lua_isyieldable(state) != 0 {
                let _ = luaur_vm::functions::lua_break::lua_break(state);
            }
        },
        Err(e) => unsafe {
            // Raise the error as a Lua error. Push the message and longjmp.
            raise_error(state, &e);
        },
    }
}

/// Push `e`'s message as a string error object and `lua_error` it (does not
/// return).
unsafe fn raise_error(state: *mut lua_State, e: &Error) -> ! {
    // Use the bare message for a runtime error (so it round-trips back through
    // `pop_error` as `RuntimeError(msg)` without a doubled "runtime error: "
    // prefix); fall back to the full Display for other error kinds.
    let msg = match e {
        Error::RuntimeError(m) => m.clone(),
        other => other.to_string(),
    };
    unsafe {
        // The interrupt fires at an arbitrary VM safepoint where `L->top` may be
        // flush against the call-info top; make room before pushing so the
        // `api_incr_top` stack invariant in `lua_pushlstring` holds.
        lua_rawcheckstack(state, 1);
        lua_pushlstring(state, msg.as_ptr() as *const c_char, msg.len());
        lua_error(state)
    }
}

#[cfg(test)]
pub(crate) fn interrupts_len() -> usize {
    interrupt_store::read(|m| m.len())
}
