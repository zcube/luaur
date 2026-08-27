//! `#[derive(UserData)]` — generates an `impl UserData for T` that exposes the
//! struct's fields to Lua.
//!
//! ## Relation to mlua
//!
//! mlua's `#[derive(UserData)]` (`mlua_derive/src/userdata/mod.rs`) handles the
//! struct's **named fields** directly — registering an `add_field_method_get`
//! (and `add_field_method_set` unless the field is read-only) per field — and
//! defers *methods* to a separate `#[userdata_impl]` attribute macro plus an
//! `inventory`-based registry. luaur-rt has neither `inventory` nor a
//! `UserDataRegistry`, and its `UserData` trait uses the `add_fields` /
//! `add_methods` shape (mirroring mlua 0.9). So this derive emits a direct
//! `impl UserData for T` whose `add_fields` registers the field getters/setters
//! — faithful to the **field** behaviour of mlua's derive, with the same
//! `#[lua(...)]` field attributes (`skip`, `get`, `set`, `name = "..."`).
//!
//! The default (no `#[lua(get/set)]`) is get + set, exactly like mlua. A field
//! getter clones the field (`Ok(this.field.clone())`); a setter moves the
//! incoming value into the field. Both require the field type to be `Clone`
//! (getter) / `FromLua` (setter) / `IntoLua` (getter) — the same requirements
//! mlua's generated getters/setters impose.

use proc_macro::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Error, Fields, FieldsNamed, Meta};

use crate::attr::LuaAttr;

/// Parse and merge all `#[lua(...)]` attributes on a field into one [`LuaAttr`].
/// Mirrors mlua's `parse_field_lua_attr`.
fn parse_field_lua_attr(attrs: &[Attribute]) -> syn::Result<LuaAttr> {
    let mut lua_attr = LuaAttr::default();
    for attr in attrs {
        if !attr.path().is_ident("lua") {
            continue;
        }
        match &attr.meta {
            // `#[lua(...)]`
            Meta::List(_) => {
                lua_attr.span = Some(attr.span());
                attr.parse_nested_meta(|meta| lua_attr.parse_inner(meta))?;
            }
            // bare `#[lua]` — equivalent to default get + set.
            Meta::Path(_) => {
                lua_attr.span = Some(attr.span());
            }
            Meta::NameValue(_) => {
                return Err(syn::Error::new_spanned(
                    attr,
                    "`#[lua = \"...\"]` is not supported: use `#[lua(name = \"...\")]`",
                ));
            }
        }
    }
    Ok(lua_attr)
}

pub fn userdata_type(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let type_name = &input.ident;

    // Generic type parameters are not supported (mlua rejects them too — the
    // registry needs a concrete type). Lifetimes/const generics share the same
    // limitation here.
    if !input.generics.params.is_empty() {
        return Error::new_spanned(
            &input.generics,
            "`#[derive(UserData)]` does not support generic parameters; \
             wrap the generic type in a concrete newtype instead",
        )
        .to_compile_error()
        .into();
    }

    let named_fields: Option<&FieldsNamed> = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => Some(fields),
            // Tuple structs / unit structs / enums simply expose no fields. They
            // still get a valid (empty-field) `UserData` impl, matching mlua,
            // where field exposure only applies to named-field structs.
            Fields::Unnamed(_) | Fields::Unit => None,
        },
        Data::Enum(_) => None,
        Data::Union(_) => {
            return Error::new_spanned(&input, "`#[derive(UserData)]` cannot be applied to unions")
                .to_compile_error()
                .into();
        }
    };

    let mut field_registrations = Vec::new();
    if let Some(fields) = named_fields {
        for field in &fields.named {
            let field_name = field.ident.as_ref().unwrap();

            let lua_attr = match parse_field_lua_attr(&field.attrs) {
                Ok(a) => a,
                Err(err) => return err.to_compile_error().into(),
            };
            // `skip` is meaningless combined with `get`/`set`/`name`.
            if lua_attr.skip && (lua_attr.get || lua_attr.set || lua_attr.name.is_some()) {
                return Error::new(
                    lua_attr.span(),
                    "`skip` cannot be combined with `get`, `set`, or `name`",
                )
                .to_compile_error()
                .into();
            }
            if lua_attr.skip {
                continue;
            }

            let lua_name = lua_attr
                .name
                .clone()
                .unwrap_or_else(|| field_name.to_string());

            // Default (neither `get` nor `set` given) is get + set, like mlua.
            let (has_get, has_set) = if lua_attr.get || lua_attr.set {
                (lua_attr.get, lua_attr.set)
            } else {
                (true, true)
            };

            if has_get {
                field_registrations.push(quote! {
                    fields.add_field_method_get(#lua_name, |_lua, this| Ok(this.#field_name.clone()));
                });
            }
            if has_set {
                field_registrations.push(quote! {
                    fields.add_field_method_set(#lua_name, |_lua, this, val| {
                        this.#field_name = val;
                        Ok(())
                    });
                });
            }
        }
    }

    let rt = crate::runtime_path();

    let output = quote! {
        impl #rt::UserData for #type_name {
            fn add_fields<__LuaurUDF: #rt::UserDataFields<Self>>(fields: &mut __LuaurUDF) {
                #(#field_registrations)*
            }
        }
    };

    output.into()
}
