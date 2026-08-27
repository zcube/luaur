//! [`RegistryKey`] — long-term storage of a Lua value in the registry.
//!
//! Mirrors `mlua::RegistryKey`. A registry key holds a value reachable by the
//! GC for as long as the key (or a clone of it) is alive. It is created with
//! [`Lua::create_registry_value`] and read back with [`Lua::registry_value`].
//! Dropping the key (or calling [`Lua::remove_registry_value`]) releases the
//! registry slot.
//!
//! Under the hood a `RegistryKey` is just a public wrapper around the same
//! `lua_ref`/`lua_unref` machinery the internal handles already use
//! ([`crate::state::LuaRef`]): `create_registry_value` pushes the value and
//! takes a registry ref; `registry_value` re-pushes it. Each key remembers
//! which [`Lua`] minted it so a key used with the wrong instance is rejected
//! with [`Error::MismatchedRegistryKey`].

use crate::error::{Error, Result};
use crate::state::{Lua, LuaRef};
use crate::sync::XRc;
use crate::traits::{FromLua, IntoLua};
use crate::value::Value;

/// An owned reference to a value stored in the Lua registry.
///
/// Mirrors `mlua::RegistryKey`. Cloning produces another handle to the **same**
/// stored value (the slot is shared via `Rc`). The value stays alive until the
/// last clone is dropped or it is explicitly removed.
///
/// Under the `send` feature it is `Send + Sync` (all VM access is
/// serialized by the per-VM lock — see [`crate::state::StateRef`]).
#[derive(Clone)]
pub struct RegistryKey {
    pub(crate) reference: XRc<LuaRef>,
}

impl RegistryKey {
    pub(crate) fn from_ref(reference: LuaRef) -> RegistryKey {
        RegistryKey {
            reference: XRc::new(reference),
        }
    }

    /// Push the stored value onto the owning state's stack.
    pub(crate) fn push(&self) {
        self.reference.push();
    }
}

impl std::fmt::Debug for RegistryKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Include the registry slot id so two keys referring to different slots
        // print differently (mlua's `RegistryKey` Debug exposes the slot too).
        write!(f, "RegistryKey({})", self.reference.id())
    }
}

// `RegistryKey` is usable as a hash-map key (mlua's `test_lua_registry_hash`).
// Identity is the (state, registry-slot) pair: a clone shares the same slot, so
// it hashes/compares equal; keys for distinct values use distinct slots.
impl PartialEq for RegistryKey {
    fn eq(&self, other: &Self) -> bool {
        self.reference.state_ptr() == other.reference.state_ptr()
            && self.reference.id() == other.reference.id()
    }
}

impl Eq for RegistryKey {}

impl std::hash::Hash for RegistryKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.reference.state_ptr() as usize).hash(state);
        self.reference.id().hash(state);
    }
}

impl Lua {
    /// Store a value in the registry and return a [`RegistryKey`] that keeps it
    /// alive. Mirrors `mlua::Lua::create_registry_value`.
    pub fn create_registry_value(&self, value: impl IntoLua) -> Result<RegistryKey> {
        let v = value.into_lua(self)?;
        // Hold the VM lock across the push + pop_ref pair (see `StateRef`).
        let _vm = self.state();
        self.push_value(&v)?;
        Ok(RegistryKey::from_ref(self.pop_ref()))
    }

    /// Read back a value previously stored with [`Lua::create_registry_value`],
    /// converting it to `T`. Mirrors `mlua::Lua::registry_value`.
    pub fn registry_value<T: FromLua>(&self, key: &RegistryKey) -> Result<T> {
        if !self.owns_registry_value(key) {
            return Err(Error::MismatchedRegistryKey);
        }
        let state = self.state();
        let state = state.get();
        let value = unsafe {
            key.push();
            let v = self.value_from_stack(-1)?;
            crate::sys::lua_pop(state, 1);
            v
        };
        T::from_lua(value, self)
    }

    /// Remove a value from the registry, releasing its slot. Mirrors
    /// `mlua::Lua::remove_registry_value`.
    pub fn remove_registry_value(&self, key: RegistryKey) -> Result<()> {
        if !self.owns_registry_value(&key) {
            return Err(Error::MismatchedRegistryKey);
        }
        // Dropping the key releases the underlying `lua_ref` slot.
        drop(key);
        Ok(())
    }

    /// Replace the value stored under an existing key. Mirrors
    /// `mlua::Lua::replace_registry_value`.
    pub fn replace_registry_value(&self, key: &mut RegistryKey, value: impl IntoLua) -> Result<()> {
        if !self.owns_registry_value(key) {
            return Err(Error::MismatchedRegistryKey);
        }
        *key = self.create_registry_value(value)?;
        Ok(())
    }

    /// Whether this `Lua` instance owns `key` (i.e. `key` was minted by this VM,
    /// not a different one). Mirrors `mlua::Lua::owns_registry_value`.
    pub fn owns_registry_value(&self, key: &RegistryKey) -> bool {
        // Two `Lua` handles share the same VM iff their inner state pointers are
        // equal (cloning a `Lua` shares the `Rc<LuaInner>`; a separate
        // `Lua::new()` has a distinct state).
        key.reference.state_ptr() == self.state_ptr()
    }

    /// Expire any [`RegistryKey`]s whose strong handles have all been dropped.
    ///
    /// Mirrors `mlua::Lua::expire_registry_values`. luaur-rt releases a
    /// registry slot eagerly when the last clone of its `RegistryKey` is dropped
    /// (via [`crate::state::LuaRef`]'s `Drop` calling `lua_unref`), so there is
    /// no deferred-expiry queue to drain; this is a no-op kept for parity.
    pub fn expire_registry_values(&self) {}

    /// Store a value in the registry under the string `name`. Mirrors
    /// `mlua::Lua::set_named_registry_value`.
    pub fn set_named_registry_value(&self, name: &str, value: impl IntoLua) -> Result<()> {
        let v = value.into_lua(self)?;
        let state = self.state();
        let state = state.get();
        let cname = std::ffi::CString::new(name)
            .map_err(|_| Error::runtime("registry name contains a NUL byte"))?;
        unsafe {
            self.push_value(&v)?;
            // lua_setfield pops the value and stores registry[name] = value.
            crate::sys::lua_setfield(state, crate::sys::LUA_REGISTRYINDEX, cname.as_ptr());
        }
        Ok(())
    }

    /// Read back a value previously stored with
    /// [`Lua::set_named_registry_value`], converting it to `T`. A name that was
    /// never set (or was unset) reads back as `nil`. Mirrors
    /// `mlua::Lua::named_registry_value`.
    pub fn named_registry_value<T: FromLua>(&self, name: &str) -> Result<T> {
        let state = self.state();
        let state = state.get();
        let cname = std::ffi::CString::new(name)
            .map_err(|_| Error::runtime("registry name contains a NUL byte"))?;
        let value = unsafe {
            crate::sys::lua_getfield(state, crate::sys::LUA_REGISTRYINDEX, cname.as_ptr());
            let v = self.value_from_stack(-1)?;
            crate::sys::lua_pop(state, 1);
            v
        };
        T::from_lua(value, self)
    }

    /// Remove a value stored under the string `name`. Mirrors
    /// `mlua::Lua::unset_named_registry_value`.
    pub fn unset_named_registry_value(&self, name: &str) -> Result<()> {
        self.set_named_registry_value(name, Value::Nil)
    }
}

// ---------------------------------------------------------------------------
// Conversions: a RegistryKey behaves like the value it stores when packed, and
// `Chunk::eval::<RegistryKey>()` stores the chunk's result.
// ---------------------------------------------------------------------------

impl IntoLua for RegistryKey {
    fn into_lua(self, lua: &Lua) -> Result<Value> {
        (&self).into_lua(lua)
    }
}

impl IntoLua for &RegistryKey {
    fn into_lua(self, lua: &Lua) -> Result<Value> {
        if self.reference.state_ptr() != lua.state_ptr() {
            return Err(Error::MismatchedRegistryKey);
        }
        let state = lua.state();
        let state = state.get();
        unsafe {
            self.push();
            let v = lua.value_from_stack(-1)?;
            crate::sys::lua_pop(state, 1);
            Ok(v)
        }
    }
}

impl FromLua for RegistryKey {
    fn from_lua(value: Value, lua: &Lua) -> Result<Self> {
        lua.create_registry_value(value)
    }
}
