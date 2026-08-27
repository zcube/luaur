//! Send/Sync-portability primitives, gated on the `send` feature.
//!
//! This module mirrors mlua's `XRc` / `MaybeSend` / `MaybeSync` machinery
//! ([`mlua-ref/src/types.rs`] + `types/sync.rs`). Under the `send` feature a
//! [`Lua`](crate::Lua) and all of its handles become **`Send + Sync`**, exactly
//! like mlua's `send` build: every entry into the VM (any operation that
//! obtains the raw `*mut lua_State`) first acquires a per-VM **re-entrant
//! mutex** ([`ReentrantMutex`]), so concurrent access through shared handles is
//! serialized, while callbacks re-entering the VM on the same thread do not
//! deadlock. This is the same design mlua uses (`ReentrantMutex<RawLua>`).
//!
//! ## What `send` changes
//!
//! - [`XRc<T>`] aliases `Arc<T>` (atomically reference-counted, `Send`) instead
//!   of `Rc<T>`. The shared `LuaInner` / `LuaRef` handles use it, so cloning a
//!   handle across a thread boundary keeps the refcount sound.
//! - [`MaybeSend`] gains a `Send` super-bound, so every type-erased callback /
//!   userdata closure box stored inside the VM is `Send`, and the captured
//!   environment can hold `Send` data moved in from another thread.
//! - The raw `*mut lua_State` pointers held by `LuaInner` (and, transitively,
//!   every handle) are made `Send + Sync` with documented `unsafe impl`s. This
//!   is sound because the raw state is only reachable through
//!   [`Lua::state`](crate::Lua) / `LuaRef::state`, which return a lock guard
//!   ([`crate::state::StateRef`]) holding the per-VM [`ReentrantMutex`].
//! - Per-VM side stores (app data, interrupt, compiler, memory control, serde
//!   sentinels) move from thread-local maps to process-global mutex-protected
//!   maps, so a VM used from several threads observes one coherent store.
//!
//! Without the feature everything is byte-for-byte identical to the original
//! single-threaded build: `XRc` = `Rc`, `MaybeSend` / `MaybeSync` are empty
//! marker traits blanket-implemented for every type, and no lock exists
//! (zero bound, zero cost).

/// Reference-counted shared pointer used for the VM's shared interior
/// (`LuaInner`) and the registry references (`LuaRef`) every handle clones.
///
/// `Arc` under the `send` feature (so handles can cross a thread boundary),
/// `Rc` otherwise (single-threaded, the default — byte-identical to before).
#[cfg(feature = "send")]
pub(crate) type XRc<T> = std::sync::Arc<T>;

/// See the `send`-gated variant above.
#[cfg(not(feature = "send"))]
pub(crate) type XRc<T> = std::rc::Rc<T>;

/// The weak counterpart of [`XRc`]: `Weak` from `Arc`/`Rc` depending on `send`.
/// Used by [`WeakLua`](crate::state::WeakLua) to hold a non-owning reference to
/// the VM's shared interior. Mirrors mlua's `XWeak`.
#[cfg(feature = "send")]
pub(crate) type XWeak<T> = std::sync::Weak<T>;

/// See the `send`-gated variant above.
#[cfg(not(feature = "send"))]
pub(crate) type XWeak<T> = std::rc::Weak<T>;

/// A trait that adds a `Send` requirement **iff** the `send` feature is enabled.
///
/// Mirrors `mlua::MaybeSend`. It is applied to every Rust closure that gets
/// type-erased and stored inside the VM (the `create_function` closure, every
/// userdata method/field/function closure, and — under `async` — the async
/// callback) and to the userdata payload type `T`. Under `send` that forces the
/// stored boxes (and their captured environment) to be `Send`; without the
/// feature it is an empty marker implemented for all types, so the extra bound
/// is a no-op and the default build is unchanged.
#[cfg(feature = "send")]
pub trait MaybeSend: Send {}
#[cfg(feature = "send")]
impl<T: Send> MaybeSend for T {}

/// See the `send`-gated variant above.
#[cfg(not(feature = "send"))]
pub trait MaybeSend {}
#[cfg(not(feature = "send"))]
impl<T> MaybeSend for T {}

/// A trait that adds a `Sync` requirement **iff** the `send` feature is enabled.
///
/// Mirrors `mlua::MaybeSync`. Applied to userdata payload types
/// (`T: UserData + MaybeSend + MaybeSync`), matching mlua's bounds.
#[cfg(feature = "send")]
pub trait MaybeSync: Sync {}
#[cfg(feature = "send")]
impl<T: Sync> MaybeSync for T {}

/// See the `send`-gated variant above.
#[cfg(not(feature = "send"))]
pub trait MaybeSync {}
#[cfg(not(feature = "send"))]
impl<T> MaybeSync for T {}

/// Define a per-VM side-store map (keyed by the `global_State` pointer as
/// `usize`) with `read` / `write` accessors.
///
/// Expands to a module `$name` backed by a **thread-local** `RefCell` map
/// without the `send` feature, and by a **process-global `RwLock`** map with it
/// (a `send` VM may run on any thread, so a thread-local store would come up
/// empty after the VM crosses threads; lookups take the read lock so hot
/// lookup paths don't serialize against each other).
///
/// Under `send` the value type must be `Send + Sync` (it sits in a `RwLock`
/// map shared across threads); POD pointer-bag values that are only *used*
/// under the per-VM lock document that with their own `unsafe impl`s.
macro_rules! vm_state_map {
    ($(#[$attr:meta])* $name:ident : $value:ty) => {
        #[cfg(not(feature = "send"))]
        $(#[$attr])*
        mod $name {
            #[allow(unused_imports)]
            use super::*;
            use std::cell::RefCell;
            use std::collections::HashMap;

            thread_local! {
                static MAP: RefCell<HashMap<usize, $value>> = RefCell::new(HashMap::new());
            }

            #[allow(dead_code)]
            pub(crate) fn read<R>(f: impl FnOnce(&HashMap<usize, $value>) -> R) -> R {
                MAP.with(|m| f(&m.borrow()))
            }

            #[allow(dead_code)]
            pub(crate) fn write<R>(f: impl FnOnce(&mut HashMap<usize, $value>) -> R) -> R {
                MAP.with(|m| f(&mut m.borrow_mut()))
            }

            /// Like `write`, but silently a no-op when the thread-local has
            /// already been torn down (drops that run during thread exit).
            #[allow(dead_code)]
            pub(crate) fn try_write(f: impl FnOnce(&mut HashMap<usize, $value>)) {
                let _ = MAP.try_with(|m| f(&mut m.borrow_mut()));
            }
        }

        #[cfg(feature = "send")]
        $(#[$attr])*
        mod $name {
            #[allow(unused_imports)]
            use super::*;
            use std::collections::HashMap;
            use std::sync::{LazyLock, RwLock};

            static MAP: LazyLock<RwLock<HashMap<usize, $value>>> =
                LazyLock::new(|| RwLock::new(HashMap::new()));

            #[allow(dead_code)]
            pub(crate) fn read<R>(f: impl FnOnce(&HashMap<usize, $value>) -> R) -> R {
                let guard = MAP.read().unwrap_or_else(|e| e.into_inner());
                f(&guard)
            }

            #[allow(dead_code)]
            pub(crate) fn write<R>(f: impl FnOnce(&mut HashMap<usize, $value>) -> R) -> R {
                let mut guard = MAP.write().unwrap_or_else(|e| e.into_inner());
                f(&mut guard)
            }

            /// A process-global static never tears down, so this is just
            /// `write` (present for signature parity with the TLS backend).
            #[allow(dead_code)]
            pub(crate) fn try_write(f: impl FnOnce(&mut HashMap<usize, $value>)) {
                let mut guard = MAP.write().unwrap_or_else(|e| e.into_inner());
                f(&mut guard);
            }
        }
    };
}

pub(crate) use vm_state_map;

/// A re-entrant mutex serializing all access to one VM (`send` feature only).
///
/// This is the lock behind [`Lua`](crate::Lua)'s `Sync`-ness, mirroring mlua's
/// `ReentrantMutex<RawLua>`: any thread may acquire it, and the *same* thread
/// may acquire it again while already holding it (a callback invoked by the VM
/// re-enters through the public API on the same thread). Different threads
/// block until the owner releases its outermost guard.
///
/// Hand-rolled on `std::sync::Mutex` + `Condvar` (std's `ReentrantLock` is
/// unstable and luaur-rt takes no external dependencies). The recursion count
/// lives inside the mutex-protected state, so no atomics are needed.
#[cfg(feature = "send")]
pub(crate) struct ReentrantMutex {
    inner: std::sync::Mutex<LockState>,
    cond: std::sync::Condvar,
}

#[cfg(feature = "send")]
struct LockState {
    /// The thread currently holding the lock, if any.
    owner: Option<std::thread::ThreadId>,
    /// How many nested guards the owner holds.
    count: usize,
}

#[cfg(feature = "send")]
impl ReentrantMutex {
    pub(crate) fn new() -> ReentrantMutex {
        ReentrantMutex {
            inner: std::sync::Mutex::new(LockState {
                owner: None,
                count: 0,
            }),
            cond: std::sync::Condvar::new(),
        }
    }

    /// Acquire the lock (re-entrantly on the owning thread), blocking other
    /// threads until every guard of the current owner is dropped.
    pub(crate) fn lock(&self) -> ReentrantGuard<'_> {
        let me = std::thread::current().id();
        // A panic while holding the inner mutex can only happen between the
        // wait/notify bookkeeping below (no user code runs under it), so a
        // poisoned state is still consistent — recover instead of propagating.
        let mut st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match st.owner {
                None => {
                    st.owner = Some(me);
                    st.count = 1;
                    break;
                }
                Some(owner) if owner == me => {
                    st.count += 1;
                    break;
                }
                Some(_) => {
                    st = self.cond.wait(st).unwrap_or_else(|e| e.into_inner());
                }
            }
        }
        ReentrantGuard {
            mutex: self,
            _not_send: std::marker::PhantomData,
        }
    }
}

/// RAII guard for [`ReentrantMutex`]; releasing the outermost guard wakes one
/// waiting thread.
///
/// `!Send` (like every mutex guard): ownership is tracked per thread, so a
/// guard must be dropped on the thread that acquired it. The raw-pointer
/// `PhantomData` suppresses the auto impl.
#[cfg(feature = "send")]
pub(crate) struct ReentrantGuard<'a> {
    mutex: &'a ReentrantMutex,
    _not_send: std::marker::PhantomData<*mut ()>,
}

#[cfg(feature = "send")]
impl Drop for ReentrantGuard<'_> {
    fn drop(&mut self) {
        let mut st = self.mutex.inner.lock().unwrap_or_else(|e| e.into_inner());
        debug_assert_eq!(st.owner, Some(std::thread::current().id()));
        st.count -= 1;
        if st.count == 0 {
            st.owner = None;
            drop(st);
            self.mutex.cond.notify_one();
        }
    }
}

/// Process-global registry mapping a VM (keyed by its `global_State` pointer)
/// to its [`ReentrantMutex`] (`send` feature only).
///
/// The lock must be **per-VM, not per-`LuaInner`**: a trampoline builds a
/// *borrowed* `Lua` (its own `LuaInner`) around the calling state, and handles
/// created through it can outlive the callback and be used from other threads.
/// If each `LuaInner` had its own mutex, those handles would lock a different
/// mutex than the owning `Lua`'s and the two could race on the same
/// `lua_State`. Keying by the `global_State` pointer gives every wrapper of the
/// same VM (including all of its coroutine threads, which share the global
/// state) the same lock.
///
/// The owning `LuaInner` registers the lock at construction and removes it
/// after `lua_close` (see `LuaInner::drop`). Pointer reuse by the allocator is
/// therefore harmless: a key is always unregistered before the address can be
/// recycled into a new VM.
#[cfg(feature = "send")]
pub(crate) mod vm_lock_registry {
    use super::ReentrantMutex;
    use std::collections::HashMap;
    use std::sync::{Arc, LazyLock, RwLock};

    static LOCKS: LazyLock<RwLock<HashMap<usize, Arc<ReentrantMutex>>>> =
        LazyLock::new(|| RwLock::new(HashMap::new()));

    /// The lock for the VM with global-state key `key`, creating it if absent
    /// (the borrowed-`Lua` path can, in principle, see a state luaur-rt did not
    /// create; giving it a fresh lock keeps that path sound too).
    ///
    /// The hot path (a borrowed `LuaInner` built by a trampoline entry) is a
    /// pure lookup, so it takes the read lock; only a genuinely new key
    /// upgrades to the write lock.
    pub(crate) fn lock_for(key: usize) -> Arc<ReentrantMutex> {
        if let Some(lock) = LOCKS.read().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return lock.clone();
        }
        let mut map = LOCKS.write().unwrap_or_else(|e| e.into_inner());
        map.entry(key)
            .or_insert_with(|| Arc::new(ReentrantMutex::new()))
            .clone()
    }

    /// Remove the registry entry for a closed VM. Outstanding `Arc` clones
    /// (borrowed `LuaInner`s that survive the owner) keep the mutex itself
    /// alive; only the key mapping is dropped.
    pub(crate) fn unregister(key: usize) {
        let mut map = LOCKS.write().unwrap_or_else(|e| e.into_inner());
        map.remove(&key);
    }
}
