//! Per-VM **application data** — a typed, borrow-checked side store keyed by
//! Rust `TypeId`. Mirrors `mlua::Lua`'s app-data surface (`set_app_data`,
//! `app_data_ref`, `app_data_mut`, `remove_app_data`, and the `try_*` variants).
//!
//! Each `Lua` instance has its own store. Values are kept behind a per-entry
//! borrow-tracked cell, so a `&T` ([`AppDataRef`]) and a `&mut T`
//! ([`AppDataRefMut`]) of **different** types can coexist (the usual aliasing
//! rules apply only within a single type). A VM-wide borrow counter (matching
//! mlua's `AppData`) additionally makes *any* outstanding borrow block
//! `set_app_data` / `remove_app_data` of *any* type, so the container is never
//! mutated while a guard is live.
//!
//! The store is keyed by the VM's global-state pointer (the same pattern
//! `luau_ext` uses for the per-VM compiler), since `LuaInner` itself is shared
//! immutably behind an `XRc`. Without the `send` feature it lives in a
//! thread-local map with `Rc<RefCell>` entries; under `send` — where a VM (and
//! its handles) may be used from several threads — it lives in a process-global
//! `RwLock`-protected map (lookups take the read lock), entries are
//! `Arc`s with **atomic** borrow flags, and
//! stored values must be `Send` (mlua's bound under its `send` feature too).

use std::any::TypeId;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use crate::error::{Error, Result};
use crate::state::Lua;
use crate::sync::MaybeSend;
use crate::sys::lua_State;

unsafe fn vm_key(state: *mut lua_State) -> usize {
    unsafe { (*state).global as usize }
}

// ---------------------------------------------------------------------------
// Store backends.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "send"))]
mod store {
    //! Single-threaded backend: thread-local map, `Rc` entries with a
    //! plain-`Cell` borrow flag (the same shape as the `send` backend's
    //! atomic cell, so the guard/API code above the backends is identical).
    use super::*;
    use std::cell::{Cell, RefCell, UnsafeCell};
    use std::rc::Rc;

    /// The type-erased stored value (no `Send` bound needed single-threaded).
    pub(super) type AnyBox = Box<dyn std::any::Any>;
    pub(super) type Entry = Rc<BorrowCell>;
    /// The VM-wide outstanding-borrow counter.
    pub(super) type BorrowCounter = Rc<Cell<usize>>;

    /// A `RefCell`-alike whose borrow flag the guards manage manually:
    /// `0` = free, `> 0` = that many readers, `-1` = one writer.
    pub(super) struct BorrowCell {
        flag: Cell<isize>,
        value: UnsafeCell<AnyBox>,
    }

    impl BorrowCell {
        pub(super) fn try_borrow(&self) -> Option<()> {
            let f = self.flag.get();
            (f >= 0).then(|| self.flag.set(f + 1))
        }
        pub(super) fn try_borrow_mut(&self) -> Option<()> {
            (self.flag.get() == 0).then(|| self.flag.set(-1))
        }
        pub(super) fn release_read(&self) {
            self.flag.set(self.flag.get() - 1);
        }
        pub(super) fn release_write(&self) {
            self.flag.set(0);
        }
        /// The stored box. Only sound while a matching borrow flag is held.
        pub(super) fn value_ptr(&self) -> *mut AnyBox {
            self.value.get()
        }
    }

    #[derive(Default)]
    pub(super) struct Store {
        pub(super) entries: HashMap<TypeId, Entry>,
        pub(super) borrow: BorrowCounter,
    }

    thread_local! {
        /// Per-VM application-data store, keyed by global-state pointer.
        static APP_DATA: RefCell<HashMap<usize, Store>> = RefCell::new(HashMap::new());
    }

    pub(super) fn read<R>(f: impl FnOnce(&HashMap<usize, Store>) -> R) -> R {
        APP_DATA.with(|m| f(&m.borrow()))
    }

    pub(super) fn write<R>(f: impl FnOnce(&mut HashMap<usize, Store>) -> R) -> R {
        APP_DATA.with(|m| f(&mut m.borrow_mut()))
    }

    pub(super) fn new_entry(value: AnyBox) -> Entry {
        Rc::new(BorrowCell {
            flag: Cell::new(0),
            value: UnsafeCell::new(value),
        })
    }

    pub(super) fn into_value(entry: Entry) -> Option<AnyBox> {
        Rc::try_unwrap(entry)
            .ok()
            .map(|cell| cell.value.into_inner())
    }

    pub(super) fn counter_get(c: &BorrowCounter) -> usize {
        c.get()
    }
    pub(super) fn counter_inc(c: &BorrowCounter) {
        c.set(c.get() + 1);
    }
    pub(super) fn counter_dec(c: &BorrowCounter) {
        c.set(c.get().saturating_sub(1));
    }
}

#[cfg(feature = "send")]
mod store {
    //! Multi-threaded backend (`send`): process-global mutex-protected map,
    //! `Arc` entries with an **atomic** borrow flag — a guard may be dropped on
    //! a different thread than the one that took it, so the borrow bookkeeping
    //! cannot rely on the VM lock alone.
    use super::*;
    use std::cell::UnsafeCell;
    use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
    use std::sync::{Arc, LazyLock, RwLock};

    /// The type-erased stored value. `Send` because it may be created, used
    /// and dropped on different threads (matching mlua's `send` bound).
    pub(super) type AnyBox = Box<dyn std::any::Any + Send>;
    pub(super) type Entry = Arc<AtomicCell>;
    pub(super) type BorrowCounter = Arc<AtomicUsize>;

    /// A `RefCell` with atomic borrow flags: `0` = free, `> 0` = that many
    /// readers, `-1` = one writer. The value itself is only dereferenced
    /// through guards that hold a flag, so aliasing follows the usual
    /// shared-xor-mutable rule across threads.
    pub(super) struct AtomicCell {
        flag: AtomicIsize,
        value: UnsafeCell<AnyBox>,
    }

    // SAFETY: access to `value` is gated by `flag` (shared-xor-mutable), and
    // the stored box is `Send`; the cell may therefore be shared and its
    // guards dropped across threads.
    unsafe impl Send for AtomicCell {}
    unsafe impl Sync for AtomicCell {}

    impl AtomicCell {
        pub(super) fn try_borrow(&self) -> Option<()> {
            self.flag
                .fetch_update(Ordering::Acquire, Ordering::Relaxed, |f| {
                    (f >= 0).then_some(f + 1)
                })
                .ok()
                .map(|_| ())
        }
        pub(super) fn try_borrow_mut(&self) -> Option<()> {
            self.flag
                .compare_exchange(0, -1, Ordering::Acquire, Ordering::Relaxed)
                .ok()
                .map(|_| ())
        }
        pub(super) fn release_read(&self) {
            self.flag.fetch_sub(1, Ordering::Release);
        }
        pub(super) fn release_write(&self) {
            self.flag.store(0, Ordering::Release);
        }
        /// The stored box. Only sound while a matching borrow flag is held.
        pub(super) fn value_ptr(&self) -> *mut AnyBox {
            self.value.get()
        }
    }

    #[derive(Default)]
    pub(super) struct Store {
        pub(super) entries: HashMap<TypeId, Entry>,
        pub(super) borrow: BorrowCounter,
    }

    static APP_DATA: LazyLock<RwLock<HashMap<usize, Store>>> =
        LazyLock::new(|| RwLock::new(HashMap::new()));

    pub(super) fn read<R>(f: impl FnOnce(&HashMap<usize, Store>) -> R) -> R {
        let guard = APP_DATA.read().unwrap_or_else(|e| e.into_inner());
        f(&guard)
    }

    pub(super) fn write<R>(f: impl FnOnce(&mut HashMap<usize, Store>) -> R) -> R {
        let mut guard = APP_DATA.write().unwrap_or_else(|e| e.into_inner());
        f(&mut guard)
    }

    pub(super) fn new_entry(value: AnyBox) -> Entry {
        Arc::new(AtomicCell {
            flag: AtomicIsize::new(0),
            value: UnsafeCell::new(value),
        })
    }

    pub(super) fn into_value(entry: Entry) -> Option<AnyBox> {
        Arc::try_unwrap(entry)
            .ok()
            .map(|cell| cell.value.into_inner())
    }

    pub(super) fn counter_get(c: &BorrowCounter) -> usize {
        c.load(Ordering::Acquire)
    }
    pub(super) fn counter_inc(c: &BorrowCounter) {
        c.fetch_add(1, Ordering::AcqRel);
    }
    pub(super) fn counter_dec(c: &BorrowCounter) {
        c.fetch_sub(1, Ordering::AcqRel);
    }
}

use store::{AnyBox, BorrowCounter, Entry};

// ---------------------------------------------------------------------------
// Guards.
// ---------------------------------------------------------------------------
//
// Both backends share the same guard shape: the entry (kept alive), a raw
// pointer to the `T` inside its box (resolved once at construction — the box
// address is stable while the entry is alive), and the VM-wide borrow counter
// to decrement on drop. The per-entry borrow flag is released by `Drop`.

/// An immutable borrow of an application-data value of type `T`. Mirrors
/// `mlua::AppDataRef`.
pub struct AppDataRef<T: 'static> {
    entry: Entry,
    ptr: *const T,
    borrow: BorrowCounter,
}

impl<T: 'static> Drop for AppDataRef<T> {
    fn drop(&mut self) {
        release_read(&self.entry);
        store::counter_dec(&self.borrow);
    }
}

impl<T: 'static> Deref for AppDataRef<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: the read-borrow flag taken at construction is still held and
        // `entry` keeps the box alive.
        unsafe { &*self.ptr }
    }
}

impl<T: std::fmt::Debug + 'static> std::fmt::Debug for AppDataRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

impl<T: std::fmt::Display + 'static> std::fmt::Display for AppDataRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

impl<T: PartialEq + 'static> PartialEq<T> for AppDataRef<T> {
    fn eq(&self, other: &T) -> bool {
        (**self) == *other
    }
}

/// A mutable borrow of an application-data value of type `T`. Mirrors
/// `mlua::AppDataRefMut`.
pub struct AppDataRefMut<T: 'static> {
    entry: Entry,
    ptr: *mut T,
    borrow: BorrowCounter,
}

impl<T: 'static> Drop for AppDataRefMut<T> {
    fn drop(&mut self) {
        release_write(&self.entry);
        store::counter_dec(&self.borrow);
    }
}

impl<T: 'static> Deref for AppDataRefMut<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: the write-borrow flag taken at construction is still held
        // and `entry` keeps the box alive.
        unsafe { &*self.ptr }
    }
}

impl<T: 'static> DerefMut for AppDataRefMut<T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as above; the flag is exclusive.
        unsafe { &mut *self.ptr }
    }
}

impl<T: std::fmt::Debug + 'static> std::fmt::Debug for AppDataRefMut<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

impl<T: PartialEq + 'static> PartialEq<T> for AppDataRefMut<T> {
    fn eq(&self, other: &T) -> bool {
        (**self) == *other
    }
}

// ---------------------------------------------------------------------------
// Backend-specific borrow/release plumbing used by the shared guard/API code.
// ---------------------------------------------------------------------------

fn try_take_read(entry: &Entry) -> Option<*const AnyBox> {
    entry.try_borrow()?;
    Some(entry.value_ptr() as *const AnyBox)
}

fn try_take_write(entry: &Entry) -> Option<*mut AnyBox> {
    entry.try_borrow_mut()?;
    Some(entry.value_ptr())
}

fn release_read(entry: &Entry) {
    entry.release_read();
}

fn release_write(entry: &Entry) {
    entry.release_write();
}

// ---------------------------------------------------------------------------
// Public API.
// ---------------------------------------------------------------------------

impl Lua {
    /// Insert (or replace) a value of type `T` in this VM's application-data
    /// store. Mirrors `mlua::Lua::set_app_data`.
    ///
    /// # Panics
    /// Panics if **any** app-data value is currently borrowed.
    pub fn set_app_data<T: MaybeSend + 'static>(&self, data: T) {
        self.try_set_app_data(data)
            .expect("cannot mutably borrow app data container");
    }

    /// Try to insert (or replace) a value of type `T`. Returns the previous
    /// value, or an error if any app-data value is currently borrowed. Mirrors
    /// `mlua::Lua::try_set_app_data`.
    pub fn try_set_app_data<T: MaybeSend + 'static>(&self, data: T) -> Result<Option<T>> {
        let key = unsafe { vm_key(self.state_ptr()) };
        store::write(|outer| {
            let s = outer.entry(key).or_default();
            // Any outstanding borrow blocks mutation of the container (mlua).
            if store::counter_get(&s.borrow) != 0 {
                return Err(Error::runtime("cannot mutably borrow app data container"));
            }
            let old = s
                .entries
                .insert(TypeId::of::<T>(), store::new_entry(Box::new(data)));
            Ok(old
                .and_then(store::into_value)
                .and_then(|b| b.downcast::<T>().ok().map(|b| *b)))
        })
    }

    /// Borrow the application-data value of type `T` immutably, if present.
    /// Mirrors `mlua::Lua::app_data_ref`.
    ///
    /// # Panics
    /// Panics if the value is currently mutably borrowed.
    pub fn app_data_ref<T: 'static>(&self) -> Option<AppDataRef<T>> {
        match self.try_app_data_ref::<T>() {
            Ok(opt) => opt,
            Err(_) => panic!("already mutably borrowed"),
        }
    }

    /// Try to borrow the application-data value of type `T` immutably. Returns
    /// `Ok(None)` if absent, `Err` if it is currently mutably borrowed. Mirrors
    /// `mlua::Lua::try_app_data_ref`.
    pub fn try_app_data_ref<T: 'static>(&self) -> Result<Option<AppDataRef<T>>> {
        let (entry, borrow) = match self.app_data_entry::<T>() {
            Some(pair) => pair,
            None => return Ok(None),
        };
        let any = try_take_read(&entry)
            .ok_or_else(|| Error::runtime("app data is currently mutably borrowed"))?;
        // SAFETY: the read flag is held; `entry` keeps the box alive.
        let ptr =
            unsafe { (*any).downcast_ref::<T>() }.expect("app data type mismatch") as *const T;
        store::counter_inc(&borrow);
        Ok(Some(AppDataRef { entry, ptr, borrow }))
    }

    /// Borrow the application-data value of type `T` mutably, if present.
    /// Mirrors `mlua::Lua::app_data_mut`.
    ///
    /// # Panics
    /// Panics if the value is currently borrowed (immutably or mutably).
    pub fn app_data_mut<T: 'static>(&self) -> Option<AppDataRefMut<T>> {
        match self.try_app_data_mut::<T>() {
            Ok(opt) => opt,
            Err(_) => panic!("already borrowed"),
        }
    }

    /// Try to borrow the application-data value of type `T` mutably. Returns
    /// `Ok(None)` if absent, `Err` if it is currently borrowed. Mirrors
    /// `mlua::Lua::try_app_data_mut`.
    pub fn try_app_data_mut<T: 'static>(&self) -> Result<Option<AppDataRefMut<T>>> {
        let (entry, borrow) = match self.app_data_entry::<T>() {
            Some(pair) => pair,
            None => return Ok(None),
        };
        let any = try_take_write(&entry)
            .ok_or_else(|| Error::runtime("app data is currently borrowed"))?;
        // SAFETY: the exclusive flag is held; `entry` keeps the box alive.
        let ptr = unsafe { (*any).downcast_mut::<T>() }.expect("app data type mismatch") as *mut T;
        store::counter_inc(&borrow);
        Ok(Some(AppDataRefMut { entry, ptr, borrow }))
    }

    /// Remove and return the application-data value of type `T`, if present.
    /// Mirrors `mlua::Lua::remove_app_data`.
    ///
    /// # Panics
    /// Panics if **any** app-data value is currently borrowed.
    pub fn remove_app_data<T: 'static>(&self) -> Option<T> {
        let key = unsafe { vm_key(self.state_ptr()) };
        store::write(|outer| {
            let s = outer.get_mut(&key)?;
            if store::counter_get(&s.borrow) != 0 {
                panic!("cannot mutably borrow app data container");
            }
            let entry = s.entries.remove(&TypeId::of::<T>())?;
            store::into_value(entry).and_then(|b| b.downcast::<T>().ok().map(|b| *b))
        })
    }

    /// This VM's entry + VM-wide borrow counter for type `T`, if present.
    fn app_data_entry<T: 'static>(&self) -> Option<(Entry, BorrowCounter)> {
        let key = unsafe { vm_key(self.state_ptr()) };
        store::read(|outer| {
            outer.get(&key).and_then(|s| {
                s.entries
                    .get(&TypeId::of::<T>())
                    .map(|e| (e.clone(), s.borrow.clone()))
            })
        })
    }
}

/// Drop this VM's entire application-data store. Called from `LuaInner::drop`.
pub(crate) fn clear_app_data(state: *mut lua_State) {
    let key = unsafe { vm_key(state) };
    store::write(|outer| {
        outer.remove(&key);
    });
}
