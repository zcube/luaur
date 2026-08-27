//! Stack inspection. Mirrors the Luau-feasible subset of `mlua::Lua::inspect_stack`
//! and `mlua::debug::Debug`.
//!
//! ## What Luau can back
//!
//! Luau's debug model is **not** the Lua 5.x line/count hook. It exposes
//! `lua_getinfo(L, level, what, ar)` for activation records and `lua_singlestep`
//! + the interrupt callback for stepping. We surface the *informational* part —
//! resolving a stack level into a [`Debug`] record (current line, source, name,
//! what kind of function) — which maps cleanly onto `lua_getinfo`.
//!
//! ## What is deferred (and why)
//!
//! The full `mlua::Lua::set_hook(HookTriggers, ...)` API (per-line / per-N-
//! instruction / on-call / on-return hooks with a `Debug` event) is a Lua 5.x
//! construct. Luau has no equivalent multiplexed hook: it has a *single* global
//! interrupt callback (see [`Lua::set_interrupt`](crate::Lua::set_interrupt))
//! and `lua_singlestep`. mlua itself gates `tests/hooks.rs` and `tests/debug.rs`
//! behind `#![cfg(not(feature = "luau"))]` for exactly this reason. We therefore
//! do **not** fake a 5.x hook surface; the interrupt API is the Luau-native
//! analog and is implemented separately.

use std::ffi::CStr;

use crate::state::Lua;
use crate::sys::*;

/// What kind of function an activation record refers to. Mirrors the relevant
/// part of `mlua::debug::DebugSource::what`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugWhat {
    /// A Lua function.
    Lua,
    /// The main chunk.
    Main,
    /// A C / Rust (native) function.
    C,
    /// Unknown / unavailable.
    Unknown,
}

/// A snapshot of one activation record, resolved from a stack level via
/// `lua_getinfo`. Mirrors the informational subset of `mlua::debug::Debug`.
#[derive(Debug, Clone)]
pub struct Debug {
    name: Option<String>,
    what: DebugWhat,
    source: Option<String>,
    short_src: Option<String>,
    current_line: Option<i64>,
    line_defined: Option<i64>,
}

impl Debug {
    /// The function's name, if known (`(n)`).
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// What kind of function this record refers to (`(s)`).
    pub fn what(&self) -> DebugWhat {
        self.what
    }

    /// The chunk source (`(s)`).
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// A short, human-readable source description (`(s)`).
    pub fn short_src(&self) -> Option<&str> {
        self.short_src.as_deref()
    }

    /// The currently executing line (`(l)`), if available.
    pub fn current_line(&self) -> Option<i64> {
        self.current_line
    }

    /// The line where the function was defined (`(s)`).
    pub fn line_defined(&self) -> Option<i64> {
        self.line_defined
    }
}

impl Lua {
    /// Inspect the activation record `level` frames up the call stack (0 = the
    /// currently running function), passing it to `f`. Returns `None` if there
    /// is no function at that level, otherwise `Some(f(&debug))`.
    ///
    /// Mirrors the Luau-feasible part of `mlua::Lua::inspect_stack` —
    /// including its mlua-0.10+ closure shape, so mlua code calling
    /// `lua.inspect_stack(1, |debug| ..)` ports verbatim. (In mlua the closure
    /// bounds the activation record's lifetime; luaur's `Debug` snapshot is
    /// owned, but the signature is kept identical for drop-in compatibility.)
    pub fn inspect_stack<R>(&self, level: usize, f: impl FnOnce(&Debug) -> R) -> Option<R> {
        self.inspect_stack_raw(level).map(|d| f(&d))
    }

    /// The record-returning form behind [`Lua::inspect_stack`].
    fn inspect_stack_raw(&self, level: usize) -> Option<Debug> {
        let state = self.state();
        let state = state.get();
        unsafe {
            let mut ar: LuaDebug = core::mem::zeroed();
            // `lua_getinfo` with a non-negative level walks call-info depth.
            let opt = c"nsl";
            let ok = lua_getinfo(
                state,
                level as c_int,
                opt.as_ptr() as *const c_char,
                &mut ar,
            );
            if ok == 0 {
                return None;
            }
            let cstr = |p: *const c_char| -> Option<String> {
                if p.is_null() {
                    None
                } else {
                    Some(CStr::from_ptr(p).to_string_lossy().into_owned())
                }
            };
            let what_str = cstr(ar.what).unwrap_or_default();
            let what = match what_str.as_str() {
                "Lua" => DebugWhat::Lua,
                "main" => DebugWhat::Main,
                "C" => DebugWhat::C,
                _ => DebugWhat::Unknown,
            };
            let current_line = if ar.currentline >= 0 {
                Some(ar.currentline as i64)
            } else {
                None
            };
            let line_defined = if ar.linedefined > 0 {
                Some(ar.linedefined as i64)
            } else {
                None
            };
            Some(Debug {
                name: cstr(ar.name),
                what,
                source: cstr(ar.source),
                short_src: cstr(ar.short_src),
                current_line,
                line_defined,
            })
        }
    }
}
