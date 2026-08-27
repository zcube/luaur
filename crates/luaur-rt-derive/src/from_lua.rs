//! `#[derive(FromLua)]` — generates an `impl FromLua for T` that extracts the
//! value from a Lua userdata of type `T` (by `borrow` + `clone`).
//!
//! Faithful port of mlua's `from_lua` derive (`mlua_derive/src/from_lua.rs`):
//! same `Self: 'static + Clone` bound, same match on the userdata value, same
//! `FromLuaConversionError` shape for any other value type. Only the crate path
//! (luaur-rt, resolved via `crate::runtime_path` so dependency renames work)
//! and luaur-rt's slightly different `Value`/`Error` field layout differ.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, parse_quote, DeriveInput};

pub fn from_lua(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident,
        mut generics,
        ..
    } = parse_macro_input!(input as DeriveInput);

    let ident_str = ident.to_string();
    generics
        .make_where_clause()
        .predicates
        .push(parse_quote!(Self: 'static + Clone));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let rt = crate::runtime_path();

    quote! {
        impl #impl_generics #rt::FromLua for #ident #ty_generics #where_clause {
            #[inline]
            fn from_lua(value: #rt::Value, _: &#rt::Lua) -> #rt::Result<Self> {
                match value {
                    #rt::Value::UserData(ud) => Ok(ud.borrow::<Self>()?.clone()),
                    _ => Err(#rt::Error::FromLuaConversionError {
                        from: value.type_name(),
                        to: #ident_str.to_string(),
                        message: None,
                    }),
                }
            }
        }
    }
    .into()
}
