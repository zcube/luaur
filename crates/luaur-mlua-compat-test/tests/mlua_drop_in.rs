//! luaur-rt consumed as `mlua` (dependency rename) — the drop-in shape used by
//! mlua embedders such as the bevy_mod_scripting Lua layer. Every test here is
//! written the way the *embedder's* source is written (`use mlua::…`, mlua
//! signatures with `?`, `Arc<dyn Fn + Send + Sync>` callback containers), so a
//! regression in any of these means the swap stops being source-compatible.

#[cfg(feature = "send")]
use std::sync::Arc;

use mlua::{
    Function, IntoLua, Lua, MetaMethod, MultiValue, Result, UserData, UserDataMethods, Value,
    Variadic,
};

/// `create_string` returns `Result` (mlua signature) — `?` composes.
#[test]
fn create_string_is_fallible_like_mlua() -> Result<()> {
    let lua = Lua::new();
    let s = lua.create_string("hello")?;
    assert_eq!(s.to_str()?, "hello");
    Ok(())
}

/// `load` accepts raw bytes (`&[u8]`), like mlua's `AsChunk`; invalid UTF-8
/// surfaces as a syntax-class error at execution, not a panic.
#[test]
fn load_accepts_bytes() -> Result<()> {
    let lua = Lua::new();
    let content: &[u8] = b"return 40 + 2";
    let n: i64 = lua.load(content).eval()?;
    assert_eq!(n, 42);

    let bad: &[u8] = b"return 1 --\xff\xfe";
    let err = lua.load(bad).exec().unwrap_err();
    assert!(
        matches!(err, mlua::Error::SyntaxError { .. }),
        "invalid UTF-8 should be a syntax error, got: {err}"
    );
    Ok(())
}

/// `inspect_stack` has mlua-0.10+'s closure signature.
#[test]
fn inspect_stack_closure_form() -> Result<()> {
    let lua = Lua::new();
    let probe = lua.create_function(|lua, ()| {
        // The exact line shape bevy_mod_scripting uses.
        let line = lua.inspect_stack(1, |debug| debug.current_line().unwrap_or_default());
        Ok(line.unwrap_or_default())
    })?;
    lua.globals().set("probe", probe)?;
    let line: i64 = lua.load("return probe()").eval()?;
    assert!(line >= 1);
    Ok(())
}

/// `Lua` is `Debug` (mlua derives it), so wrappers can `#[derive(Debug)]`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LuaContext(Lua);

#[test]
fn lua_implements_debug() {
    let ctx = LuaContext(Lua::new());
    let printed = format!("{ctx:?}");
    assert!(printed.contains("Lua"), "got: {printed}");
}

/// The derive macros must resolve the renamed crate path themselves (no
/// `extern crate mlua as luaur_rt;` shim in the embedder).
#[derive(Clone, mlua::FromLua)]
struct Marker;

impl UserData for Marker {}

#[test]
fn derive_resolves_renamed_crate() -> Result<()> {
    let lua = Lua::new();
    lua.globals().set("m", lua.create_userdata(Marker)?)?;
    let _roundtrip: Marker = lua.globals().get("m")?;
    Ok(())
}

/// A Lua `Function` captured in an `Arc<dyn Fn + Send + Sync>` — the
/// `DynamicScriptFunction` shape that requires `Sync` handles (`send` feature,
/// exercised by CI's feature-matrix step).
#[cfg(feature = "send")]
#[test]
fn function_in_host_callback_container() -> Result<()> {
    let lua = Lua::new();
    let f: Function = lua.load("return function(x) return x + 1 end").eval()?;
    let cb: Arc<dyn Fn(i64) -> i64 + Send + Sync> = Arc::new(move |x| f.call(x).unwrap());
    let out = std::thread::spawn({
        let cb = cb.clone();
        move || cb(41)
    })
    .join()
    .expect("worker panicked");
    assert_eq!(out, 42);
    Ok(())
}

/// `add_meta_function` (all-args form) exists — the shape behind
/// bevy_mod_scripting's 14 binary-operator registrations.
struct Vec2(f64, f64);

impl UserData for Vec2 {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_meta_function(MetaMethod::Add, |lua, (a, b): (Value, Value)| {
            let get = |v: &Value| -> Result<(f64, f64)> {
                match v {
                    Value::UserData(u) => {
                        let v = u.borrow::<Vec2>()?;
                        Ok((v.0, v.1))
                    }
                    other => Err(mlua::Error::runtime(format!(
                        "not a Vec2: {}",
                        other.type_name()
                    ))),
                }
            };
            let (ax, ay) = get(&a)?;
            let (bx, by) = get(&b)?;
            lua.create_userdata(Vec2(ax + bx, ay + by))?.into_lua(lua)
        });
        methods.add_method("x", |_, this, ()| Ok(this.0));
    }
}

#[test]
fn add_meta_function_binary_op() -> Result<()> {
    let lua = Lua::new();
    lua.globals()
        .set("a", lua.create_userdata(Vec2(1.0, 2.0))?)?;
    lua.globals()
        .set("b", lua.create_userdata(Vec2(3.0, 4.0))?)?;
    let x: f64 = lua.load("return (a + b):x()").eval()?;
    assert_eq!(x, 4.0);
    Ok(())
}

/// `register_callback`'s exact upstream shape: a `(String, Function)` callback
/// pair stored into an `Arc` container behind `Send + Sync` bounds (`send`
/// feature, exercised by CI's feature-matrix step).
#[cfg(feature = "send")]
#[test]
fn register_callback_shape() -> Result<()> {
    type Callback = Arc<dyn Fn(Vec<i64>) -> Result<i64> + Send + Sync>;
    let lua = Lua::new();
    let store: Arc<std::sync::Mutex<Vec<(String, Callback)>>> = Arc::default();

    let store_c = store.clone();
    let register = lua.create_function(move |_, (name, func): (String, Function)| {
        store_c.lock().unwrap().push((
            name,
            Arc::new(move |args: Vec<i64>| func.call::<i64>(Variadic::from_iter(args))),
        ));
        Ok(())
    })?;
    lua.globals().set("register_callback", register)?;

    lua.load(r#"register_callback("sum", function(a, b) return a + b end)"#)
        .exec()?;
    let cbs = store.lock().unwrap();
    let (name, cb) = &cbs[0];
    assert_eq!(name, "sum");
    assert_eq!(cb(vec![20, 22])?, 42);
    Ok(())
}

/// `MultiValue` + `Variadic` under the renamed crate (used by the embedder's
/// dispatch glue).
#[test]
fn multivalue_roundtrip() -> Result<()> {
    let lua = Lua::new();
    let f: Function = lua.load("return function(...) return ... end").eval()?;
    let out: MultiValue = f.call((1, 2, 3))?;
    assert_eq!(out.len(), 3);
    Ok(())
}
