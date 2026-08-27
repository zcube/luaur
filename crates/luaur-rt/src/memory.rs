//! Memory limit + memory-category control. Mirrors mlua's `Lua::set_memory_limit`
//! and `Lua::set_memory_category`.
//!
//! ## Memory limit
//!
//! Luau routes every allocation through `global_State::frealloc(ud, ...)`. We
//! install a limit-enforcing allocator ([`limited_alloc`]) over the default one
//! by overwriting `frealloc`/`ud` on the global state, keyed by a heap-boxed
//! [`MemoryControl`] that holds the cap and a pointer back to the global state
//! (so the allocator can compare the would-be `totalbytes` to the cap). When a
//! growing allocation would exceed the cap the allocator returns null, which the
//! VM turns into a `LUA_ERRMEM` longjmp — surfaced by luaur-rt as
//! [`Error::MemoryError`](crate::Error::MemoryError).
//!
//! A limit of `0` means "unlimited" (mlua's convention).
//!
//! ## Memory categories
//!
//! Luau tags allocations with an 8-bit *category* (`global_State::activememcat`,
//! `memcatbytes[256]`). mlua exposes named categories; we keep a per-VM
//! name→id table (max 255 user categories: id 0 is reserved for `"main"`),
//! validate names (`[A-Za-z0-9_]+`), and call `lua_setmemcat`.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::state::Lua;
use crate::sys::*;

/// The per-VM allocator control block, pointed to by `global_State::ud` once a
/// memory limit is installed.
struct MemoryControl {
    /// The byte cap (`0` = unlimited).
    limit: usize,
    /// The global state, so the allocator can read the live `totalbytes`.
    global: *mut core::ffi::c_void,
    /// The original allocator we delegate to (libc realloc/free).
    base: unsafe extern "C" fn(
        *mut core::ffi::c_void,
        *mut core::ffi::c_void,
        usize,
        usize,
    ) -> *mut core::ffi::c_void,
    /// The original allocator's userdata.
    base_ud: *mut core::ffi::c_void,
}

// SAFETY (`send` only): `MemoryControl` holds raw pointers into the VM's
// global state and its base allocator. It is only ever read/written while the
// per-VM re-entrant lock is held (the allocator runs inside VM operations; the
// public API functions lock via `self.state()`), so sharing the box across
// threads through the store below cannot race.
#[cfg(feature = "send")]
unsafe impl Send for MemoryControl {}
#[cfg(feature = "send")]
unsafe impl Sync for MemoryControl {}

crate::sync::vm_state_map!(
    /// One `MemoryControl` per VM, keyed by the global-state pointer. Boxed so
    /// its address (handed to the VM as `ud`) is stable.
    memory_controls: Box<MemoryControl>
);
crate::sync::vm_state_map!(
    /// Per-VM memory-category name→id table (id 0 is reserved for `"main"`).
    memory_categories: HashMap<String, u8>
);

/// The global-state pointer for `state` — the per-VM key.
unsafe fn global_key(state: *mut lua_State) -> usize {
    unsafe { (*state).global as usize }
}

/// The global-state pointer for `state`, exposed for `LuaInner::drop` to capture
/// the memory-map key BEFORE `lua_close` frees the state.
pub(crate) unsafe fn memory_key(state: *mut lua_State) -> usize {
    unsafe { global_key(state) }
}

/// Drop this VM's memory-control + category entries so they don't leak one slot
/// per state created. Called from `LuaInner::drop` **after** `lua_close`: the
/// `MemoryControl` whose address was handed to the VM as the allocator `ud` must
/// stay live for the entire close (which frees every object through it), so this
/// takes `key` — the global-state pointer captured *before* close, since the
/// state is freed by the time this runs.
pub(crate) fn clear_memory(key: usize) {
    memory_controls::write(|m| {
        m.remove(&key);
    });
    memory_categories::write(|m| {
        m.remove(&key);
    });
}

/// The limit-enforcing allocator. Reads the live `totalbytes` from the global
/// state and refuses any growing allocation that would push it past the cap.
unsafe extern "C" fn limited_alloc(
    ud: *mut core::ffi::c_void,
    ptr: *mut core::ffi::c_void,
    osize: usize,
    nsize: usize,
) -> *mut core::ffi::c_void {
    let ctrl = unsafe { &*(ud as *const MemoryControl) };
    if ctrl.limit != 0 && nsize > osize {
        let g = ctrl.global as *const luaur_vm::records::global_state::global_State;
        let used = unsafe { (*g).totalbytes };
        // The would-be new total once this (re)allocation is accounted for.
        let projected = used.saturating_sub(osize).saturating_add(nsize);
        if projected > ctrl.limit {
            return core::ptr::null_mut();
        }
    }
    unsafe { (ctrl.base)(ctrl.base_ud, ptr, osize, nsize) }
}

impl Lua {
    /// Set the VM's memory limit in bytes (`0` = unlimited). Mirrors
    /// `mlua::Lua::set_memory_limit`.
    ///
    /// Once installed, an allocation that would exceed the cap fails with
    /// [`Error::MemoryError`](crate::Error::MemoryError), both during execution
    /// and during chunk loading.
    pub fn set_memory_limit(&self, limit: usize) -> Result<usize> {
        let state = self.state();
        let state = state.get();
        unsafe {
            let key = global_key(state);
            let g = (*state).global;
            let prev = memory_controls::write(|map| {
                if let Some(ctrl) = map.get_mut(&key) {
                    // Already installed: just update the cap.
                    let prev = ctrl.limit;
                    ctrl.limit = limit;
                    Some(prev)
                } else {
                    None
                }
            });
            if let Some(prev) = prev {
                return Ok(prev);
            }
            // First install: capture the existing allocator and wrap it.
            let base = (*g).frealloc.expect("VM allocator must be set");
            let base_ud = (*g).ud;
            let ctrl = Box::new(MemoryControl {
                limit,
                global: g as *mut core::ffi::c_void,
                base,
                base_ud,
            });
            let ctrl_ptr = (&*ctrl) as *const MemoryControl as *mut core::ffi::c_void;
            memory_controls::write(|m| {
                m.insert(key, ctrl);
            });
            (*g).ud = ctrl_ptr;
            (*g).frealloc = Some(limited_alloc);
            Ok(0)
        }
    }

    /// Set the active memory category by name. Mirrors
    /// `mlua::Lua::set_memory_category`.
    ///
    /// Category names must be non-empty and consist only of `[A-Za-z0-9_]`.
    /// At most 255 distinct user categories can be created (id 0 is reserved
    /// for the implicit `"main"` category); creating a 256th fails.
    pub fn set_memory_category(&self, name: &str) -> Result<()> {
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return Err(Error::runtime(format!(
                "invalid memory category name: {name:?}"
            )));
        }
        let state = self.state();
        let state = state.get();
        let key = unsafe { global_key(state) };
        let id = memory_categories::write(|map| -> Result<u8> {
            let cats = map.entry(key).or_insert_with(|| {
                let mut h = HashMap::new();
                // id 0 is the implicit "main" category.
                h.insert("main".to_string(), 0u8);
                h
            });
            if let Some(&id) = cats.get(name) {
                return Ok(id);
            }
            // Assign the next free id. Luau has 256 category slots (ids
            // 0..=255); id 0 is the implicit "main" and the top slot (255) is
            // reserved, so at most 255 distinct categories exist (ids 0..=254 —
            // "main" plus 254 user categories). Creating a 256th fails.
            let next = cats.len();
            if next >= 255 {
                return Err(Error::runtime(
                    "too many memory categories (limit 255)".to_string(),
                ));
            }
            let id = next as u8;
            cats.insert(name.to_string(), id);
            Ok(id)
        })?;
        lua_setmemcat(state, id as c_int);
        Ok(())
    }

    /// The number of bytes accounted to the named memory category, or `None` if
    /// the category was never set on this VM. A luaur-rt extension (mlua tracks
    /// this only via `heap_dump`, which luaur cannot back — see the module).
    pub fn memory_category_bytes(&self, name: &str) -> Option<usize> {
        let state = self.state();
        let state = state.get();
        let key = unsafe { global_key(state) };
        let id = memory_categories::read(|m| m.get(&key).and_then(|c| c.get(name).copied()))?;
        unsafe {
            let g = (*state).global;
            Some((*g).memcatbytes[id as usize])
        }
    }
}

#[cfg(test)]
pub(crate) fn memory_controls_len() -> usize {
    memory_controls::read(|m| m.len())
}
