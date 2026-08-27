//! Empty lib target; this crate exists for its `tests/` — a drop-in-for-mlua
//! smoke suite where luaur-rt is consumed under the dependency name `mlua`
//! (`mlua = { package = "luaur-rt", .. }`), the exact shape an mlua embedder
//! (e.g. the bevy_mod_scripting Lua layer) uses to swap the runtime with no
//! source changes. See `tests/mlua_drop_in.rs`.
