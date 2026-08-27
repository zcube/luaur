//! Procedural derive macros for [`luaur-rt`](https://docs.rs/luaur-rt) — the
//! safe, mlua-style API of luaur (pure-Rust Luau).
//!
//! Two derives are provided, mirroring `mlua_derive`:
//!
//! - [`macro@UserData`] — `#[derive(UserData)]` generates an `impl UserData`
//!   that exposes the type's named struct fields to Lua (getters/setters),
//!   honoring the `#[lua(skip)]` / `#[lua(get)]` / `#[lua(set)]` /
//!   `#[lua(name = "...")]` field attributes — exactly mlua's field surface.
//! - [`macro@FromLua`] — `#[derive(FromLua)]` generates an `impl FromLua` that
//!   recovers a `T: Clone` value out of a Lua userdata of that type.
//!
//! These are intended to be used via luaur-rt's `macros` feature, which
//! re-exports them so users write `#[derive(luaur_rt::UserData)]` /
//! `#[derive(luaur_rt::FromLua)]`.
//!
//! ## Not implemented (deferred)
//!
//! mlua additionally ships a `#[userdata_impl]` **attribute** macro that turns
//! a whole `impl` block into method/meta/field registrations through an
//! `inventory`-based registry (`UserDataRegistry`) and `Lua::create_proxy`.
//! luaur-rt does not (yet) have that registry / proxy / inventory machinery, so
//! the method side of mlua's derive system is out of scope here; only the
//! field-deriving `#[derive(UserData)]` and `#[derive(FromLua)]` are provided.
//! mlua's `chunk!` and `lua_module` macros are likewise out of scope.

use proc_macro::TokenStream;

mod attr;
mod from_lua;
mod userdata;

/// The path to the luaur-rt crate **as the deriving crate names it**.
///
/// `#[derive]` output must name luaur-rt items with an absolute path, but the
/// user may have renamed the dependency (`mlua = { package = "luaur-rt", .. }`
/// is the supported mlua-drop-in shape). Resolve the real name from the user's
/// Cargo.toml; fall back to `::luaur_rt` when resolution is unavailable (e.g.
/// building outside cargo).
pub(crate) fn runtime_path() -> proc_macro2::TokenStream {
    use proc_macro_crate::{crate_name, FoundCrate};
    match crate_name("luaur-rt") {
        Ok(FoundCrate::Itself) => quote::quote!(::luaur_rt),
        Ok(FoundCrate::Name(name)) => {
            let ident = proc_macro2::Ident::new(&name, proc_macro2::Span::call_site());
            quote::quote!(::#ident)
        }
        Err(_) => quote::quote!(::luaur_rt),
    }
}

/// Derive macro implementing `UserData` for a struct, exposing its named fields
/// to Lua.
///
/// Field attributes (mirroring mlua):
/// - `#[lua(skip)]` — do not expose this field.
/// - `#[lua(get)]` — expose only a getter (read-only field).
/// - `#[lua(set)]` — expose only a setter (write-only field).
/// - `#[lua(name = "...")]` — use a custom Lua-visible name.
/// - bare `#[lua]` or no attribute — getter **and** setter (the default).
///
/// Getters clone the field, so the field type must be `Clone + IntoLua`;
/// setters move the assigned value in, so it must be `FromLua`.
#[proc_macro_derive(UserData, attributes(lua))]
pub fn userdata(item: TokenStream) -> TokenStream {
    userdata::userdata_type(item)
}

/// Derive macro implementing `FromLua` for a `Clone` type, recovering the value
/// out of a Lua userdata of that type.
#[proc_macro_derive(FromLua)]
pub fn from_lua(input: TokenStream) -> TokenStream {
    from_lua::from_lua(input)
}
