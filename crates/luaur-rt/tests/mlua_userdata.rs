// Adapted from mlua (https://github.com/mlua-rs/mlua), MIT License,
// © 2019 Aleksandr Orlenko / mlua authors. See tests/ATTRIBUTION.md.
//
// luaur-rt's userdata supports *constructing* userdata, *using it from Lua*
// (methods, mutable methods, plain functions, meta-methods, and fields), and
// Rust-side typed read-back (`borrow`/`borrow_mut`/`take`/`is`/`type_id`) via
// the `MetaMethod` enum and `UserDataFields` trait — all exercised below.
//
// Still deferred (later phases): `ObjectLike`, userdata user-values
// (`set_nth_user_value`/`user_value`), `create_ser_userdata` (serde), and
// `destroy`/once-methods. The mlua tests relying on those are dropped or
// trimmed with a one-line note.
//
// The `#[derive(UserData)]` / `#[derive(FromLua)]` procedural derives are now
// implemented (behind the `macros` feature, crate `luaur-rt-derive`); mlua's
// `test_userdata_derive` (and the derive's field surface) are ported in
// `tests/mlua_userdata_macro.rs`, gated `#![cfg(feature = "macros")]`. mlua's
// exact `test_userdata_derive` body additionally relies on
// `register_userdata_type` + `AnyUserData::wrap` (no luaur-rt equivalent), so
// the *derive* behaviour is proven there over luaur-rt's `create_userdata`.

use std::sync::Arc;

use luaur_rt::{
    AnyUserData, Error, Function, Lua, MetaMethod, MultiValue, Result, UserData, UserDataFields,
    UserDataMethods, Value, Variadic,
};

#[test]
fn test_methods() -> Result<()> {
    struct MyUserData(i64);

    impl UserData for MyUserData {
        fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
            methods.add_method("get_value", |_, data, ()| Ok(data.0));
            methods.add_method_mut("set_value", |_, data, args: i64| {
                data.0 = args;
                Ok(())
            });
        }
    }

    let lua = Lua::new();
    let globals = lua.globals();
    let userdata = lua.create_userdata(MyUserData(42))?;
    globals.set("userdata", &userdata)?;
    lua.load(
        r#"
        function get_it()
            return userdata:get_value()
        end

        function set_it(i)
            return userdata:set_value(i)
        end
    "#,
    )
    .exec()?;
    let get = globals.get::<Function>("get_it")?;
    let set = globals.get::<Function>("set_it")?;
    assert_eq!(get.call::<i64>(())?, 42);
    // Mutate the wrapped value through a typed Rust-side borrow.
    userdata.borrow_mut::<MyUserData>()?.0 = 64;
    assert_eq!(get.call::<i64>(())?, 64);
    set.call::<()>(100)?;
    assert_eq!(get.call::<i64>(())?, 100);

    Ok(())
}

#[test]
fn test_userdata() -> Result<()> {
    use std::any::TypeId;

    struct UserData1(i64);
    struct UserData2(Box<i64>);

    impl UserData for UserData1 {}
    impl UserData for UserData2 {}

    let lua = Lua::new();
    let userdata1 = lua.create_userdata(UserData1(1))?;
    let userdata2 = lua.create_userdata(UserData2(Box::new(2)))?;

    assert!(userdata1.is::<UserData1>());
    assert!(userdata1.type_id() == Some(TypeId::of::<UserData1>()));
    assert!(!userdata1.is::<UserData2>());
    assert!(userdata2.is::<UserData2>());
    assert!(!userdata2.is::<UserData1>());
    assert!(userdata2.type_id() == Some(TypeId::of::<UserData2>()));

    assert_eq!(userdata1.borrow::<UserData1>()?.0, 1);
    assert_eq!(*userdata2.borrow::<UserData2>()?.0, 2);

    Ok(())
}

#[test]
fn test_userdata_take() -> Result<()> {
    // Adapted from mlua's `test_userdata_take` (the user-value parts are
    // dropped). Exercises borrow-blocks-take, take, post-take destructed state,
    // and that `take` drops the value.
    #[derive(Debug)]
    struct MyUserdata(Arc<i64>);

    impl UserData for MyUserdata {
        fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
            methods.add_method("num", |_, this, ()| Ok(*this.0))
        }
    }

    let lua = Lua::new();
    let rc = Arc::new(18);
    let userdata = lua.create_userdata(MyUserdata(rc.clone()))?;
    lua.globals().set("userdata", &userdata)?;
    assert_eq!(Arc::strong_count(&rc), 2);

    {
        let _value = userdata.borrow::<MyUserdata>()?;
        // We should not be able to take userdata while it's borrowed.
        match userdata.take::<MyUserdata>() {
            Err(Error::UserDataBorrowMutError) => {}
            r => panic!("expected `UserDataBorrowMutError` error, got {:?}", r),
        }
    }

    let value = userdata.take::<MyUserdata>()?;
    assert_eq!(*value.0, 18);
    drop(value);
    assert_eq!(Arc::strong_count(&rc), 1);

    match userdata.borrow::<MyUserdata>() {
        Err(Error::UserDataDestructed) => {}
        r => panic!("expected `UserDataDestructed` error, got {:?}", r),
    }
    // Calling a method on the destructed userdata surfaces the destructed state.
    // Matches mlua's `test_userdata_take`: a `UserDataDestructed` returned from a
    // Rust callback now travels across the Lua boundary as the structured
    // `CallbackError { cause: UserDataDestructed }` (added with `Lua::scope`).
    match lua.load("userdata:num()").exec() {
        Err(Error::CallbackError { ref cause, .. }) => match cause.as_ref() {
            Error::UserDataDestructed => {}
            err => panic!("expected `UserDataDestructed`, got {:?}", err),
        },
        r => panic!("improper return for destructed userdata: {:?}", r),
    }

    assert!(!userdata.is::<MyUserdata>());

    Ok(())
}

#[test]
fn test_fields() -> Result<()> {
    // Adapted from mlua's `test_fields`: covers `add_field`,
    // `add_field_method_get`/`set`, and `add_field_function_get`/`set`. The
    // user-value and `add_meta_field` parts of the mlua test are dropped
    // (deferred subsystems).
    let lua = Lua::new();
    let globals = lua.globals();

    #[derive(Copy, Clone)]
    struct MyUserData(i64);

    impl UserData for MyUserData {
        fn add_fields<F: UserDataFields<Self>>(fields: &mut F) {
            fields.add_field("static", "constant");
            fields.add_field_method_get("val", |_, data| Ok(data.0));
            fields.add_field_method_set("val", |_, data, val| {
                data.0 = val;
                Ok(())
            });

            // Field that emulates a method by returning a closure.
            fields.add_field_function_get("val_fget", |lua, ud| {
                lua.create_function(move |_, ()| Ok(ud.borrow::<MyUserData>()?.0))
            });
        }

        fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
            methods.add_method("dummy", |_, _, ()| Ok(()));
        }
    }

    globals.set("ud", lua.create_userdata(MyUserData(7))?)?;
    lua.load(
        r#"
        assert(ud.static == "constant")
        assert(ud.val == 7)
        ud.val = 10
        assert(ud.val == 10)
        assert(ud:val_fget() == 10)
        ud:dummy()
    "#,
    )
    .exec()?;

    Ok(())
}

#[test]
fn test_method_variadic() -> Result<()> {
    struct MyUserData(i64);

    impl UserData for MyUserData {
        fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
            methods.add_method("get", |_, data, ()| Ok(data.0));
            methods.add_method_mut("add", |_, data, vals: Variadic<i64>| {
                data.0 += vals.into_iter().sum::<i64>();
                Ok(())
            });
        }
    }

    let lua = Lua::new();
    let globals = lua.globals();
    globals.set("userdata", lua.create_userdata(MyUserData(0))?)?;
    lua.load("userdata:add(1, 5, -10)").exec()?;
    let total: i64 = lua.load("return userdata:get()").eval()?;
    assert_eq!(total, -4);

    Ok(())
}

#[test]
fn test_metamethods() -> Result<()> {
    // Arithmetic/comparison meta-methods used from Lua. luaur-rt meta-methods
    // receive `&self` and the other operand; we return a number, keeping mlua's
    // intent of "the `__add`/`__sub` metamethods fire and compute the result".
    // (A *custom* `__index` alongside methods/fields is exercised separately in
    // `test_meta_index_*` below.)
    struct MyUserData(i64);

    impl UserData for MyUserData {
        fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
            methods.add_method("get", |_, data, ()| Ok(data.0));
            // The `MetaMethod` enum is accepted by `add_meta_method` (it also
            // accepts a raw `"__add"` string, exercised by `__sub` below).
            methods.add_meta_method(MetaMethod::Add, |_, data, other: i64| Ok(data.0 + other));
            methods.add_meta_method("__sub", |_, data, other: i64| Ok(data.0 - other));
            methods.add_meta_method(MetaMethod::ToString, |_, data, ()| {
                Ok(format!("MyUserData({})", data.0))
            });
        }
    }

    let lua = Lua::new();
    let globals = lua.globals();
    globals.set("userdata1", lua.create_userdata(MyUserData(7))?)?;

    assert_eq!(lua.load("return userdata1 + 3").eval::<i64>()?, 10);
    assert_eq!(lua.load("return userdata1 - 2").eval::<i64>()?, 5);
    assert_eq!(lua.load("return userdata1:get()").eval::<i64>()?, 7);
    assert_eq!(
        lua.load("return tostring(userdata1)").eval::<String>()?,
        "MyUserData(7)"
    );

    Ok(())
}

#[test]
fn test_functions() -> Result<()> {
    // `add_function` registers a plain function in the userdata namespace
    // (no `self`), callable as `ud.func(...)`.
    struct MyUserData(i64);

    impl UserData for MyUserData {
        fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
            methods.add_method("get_value", |_, data, ()| Ok(data.0));
            methods.add_function("get_constant", |_, ()| Ok(7));
        }
    }

    let lua = Lua::new();
    let globals = lua.globals();
    globals.set("userdata", lua.create_userdata(MyUserData(42))?)?;
    lua.load(
        r#"
        function get_it()
            return userdata:get_value()
        end
        function get_constant()
            return userdata.get_constant()
        end
    "#,
    )
    .exec()?;
    assert_eq!(globals.get::<Function>("get_it")?.call::<i64>(())?, 42);
    assert_eq!(globals.get::<Function>("get_constant")?.call::<i64>(())?, 7);

    Ok(())
}

#[test]
fn test_gc_userdata_access_after_collect() -> Result<()> {
    // DEVIATION: mlua's `test_gc_userdata` resurrects a userdata from a table's
    // `__gc` and asserts the resurrected handle is unusable. luaur's base
    // library does not expose `collectgarbage` (only `gcinfo`), and `__gc` on
    // plain tables is not part of the supported surface, so the resurrection
    // scenario cannot be expressed. We instead assert the supported invariant:
    // a userdata accessed via a *live* Rust handle keeps working across an
    // explicit `gc_collect()`.
    struct MyUserdata {
        id: u8,
    }

    impl UserData for MyUserdata {
        fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
            methods.add_method("access", |_, this, ()| {
                assert_eq!(this.id, 123);
                Ok(this.id)
            });
        }
    }

    let lua = Lua::new();
    let ud = lua.create_userdata(MyUserdata { id: 123 })?;
    lua.globals().set("userdata", ud.clone())?;

    // A GC cycle must not collect a userdata still reachable from a handle/global.
    lua.gc_collect()?;
    let id: u8 = lua.load("return userdata:access()").eval()?;
    assert_eq!(id, 123);

    Ok(())
}

#[test]
fn test_userdata_drop_runs_destructor() -> Result<()> {
    // The wrapped value's `Drop` must run when the userdata is collected.
    // (Uses the Rust `gc_collect` API since luaur lacks `collectgarbage`.)
    // `Arc<AtomicBool>` (rather than `Rc<Cell<bool>>`) so this also compiles
    // under the `send` feature, where the userdata payload `T` must be `Send`.
    // Behaviorally identical in the single-threaded default build.
    use std::sync::atomic::{AtomicBool, Ordering};
    struct Tracked(Arc<AtomicBool>);
    impl UserData for Tracked {}
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    let dropped = Arc::new(AtomicBool::new(false));
    let lua = Lua::new();
    lua.globals()
        .set("ud", lua.create_userdata(Tracked(dropped.clone()))?)?;
    assert!(!dropped.load(Ordering::SeqCst));

    // Make the userdata unreachable, then collect.
    lua.load("ud = nil").exec()?;
    lua.gc_collect()?;
    lua.gc_collect()?;
    assert!(
        dropped.load(Ordering::SeqCst),
        "userdata destructor should have run"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Custom `__index` / `__newindex` meta-methods must compose with the method
// table and field getters/setters instead of being overwritten by them (mlua's
// chain: field accessor → method table → user handler). Regression tests for
// the dispatcher in `create_userdata` / `create_scoped_userdata` — without it,
// `add_meta_method(MetaMethod::Index, ..)` was silently replaced by the method
// table and dynamic lookups came back as "attempt to call a nil value"
// (bevy_mod_scripting's `world.get_type_by_name` died exactly this way).
// ---------------------------------------------------------------------------

#[test]
fn test_meta_index_without_fields() -> Result<()> {
    struct Dynamic;

    impl UserData for Dynamic {
        fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
            methods.add_method("real", |_, _, ()| Ok("method"));
            methods.add_meta_method(MetaMethod::Index, |_, _, key: String| {
                Ok(format!("dyn:{key}"))
            });
        }
    }

    let lua = Lua::new();
    lua.globals().set("ud", lua.create_userdata(Dynamic)?)?;

    // The method table still wins for a real method...
    assert_eq!(lua.load("return ud:real()").eval::<String>()?, "method");
    // ...and an unknown key falls through to the user `__index` handler.
    assert_eq!(
        lua.load("return ud.get_type_by_name").eval::<String>()?,
        "dyn:get_type_by_name"
    );
    Ok(())
}

#[test]
fn test_meta_index_with_fields() -> Result<()> {
    struct Dynamic(i64);

    impl UserData for Dynamic {
        fn add_fields<F: luaur_rt::UserDataFields<Self>>(fields: &mut F) {
            fields.add_field_method_get("n", |_, this| Ok(this.0));
        }
        fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
            methods.add_method("real", |_, this, ()| Ok(this.0 * 10));
            methods.add_meta_method(MetaMethod::Index, |_, _, key: String| {
                Ok(format!("dyn:{key}"))
            });
        }
    }

    let lua = Lua::new();
    lua.globals().set("ud", lua.create_userdata(Dynamic(4))?)?;

    // Chain: field getter → method table → user `__index` handler.
    assert_eq!(lua.load("return ud.n").eval::<i64>()?, 4);
    assert_eq!(lua.load("return ud:real()").eval::<i64>()?, 40);
    assert_eq!(lua.load("return ud.other").eval::<String>()?, "dyn:other");
    Ok(())
}

#[test]
fn test_meta_newindex_handler() -> Result<()> {
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct Sink(Arc<Mutex<Vec<(String, i64)>>>);

    impl UserData for Sink {
        fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
            methods.add_meta_method(
                MetaMethod::NewIndex,
                |_, this, (key, val): (String, i64)| {
                    this.0.lock().unwrap().push((key, val));
                    Ok(())
                },
            );
        }
    }

    let lua = Lua::new();
    let sink = Sink(Arc::new(Mutex::new(Vec::new())));
    lua.globals()
        .set("ud", lua.create_userdata(sink.clone())?)?;

    lua.load("ud.x = 1; ud.y = 2").exec()?;
    let seen = sink.0.lock().unwrap().clone();
    assert_eq!(seen, vec![("x".to_string(), 1), ("y".to_string(), 2)]);
    Ok(())
}

#[test]
fn test_meta_index_scoped_userdata() -> Result<()> {
    struct Dynamic;

    impl UserData for Dynamic {
        fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
            methods.add_method("real", |_, _, ()| Ok("method"));
            methods.add_meta_method(MetaMethod::Index, |_, _, key: String| {
                Ok(format!("dyn:{key}"))
            });
        }
    }

    let lua = Lua::new();
    lua.scope(|scope| {
        let ud = scope.create_userdata(Dynamic)?;
        lua.globals().set("ud", ud)?;
        assert_eq!(lua.load("return ud:real()").eval::<String>()?, "method");
        assert_eq!(lua.load("return ud.zzz").eval::<String>()?, "dyn:zzz");
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// `add_meta_function` / `add_meta_function_mut` (mlua parity): the meta-method
// registered as a *plain function* — the receiver is not split off, every
// argument (receiver included) reaches the closure. This is the surface
// bevy_mod_scripting's Lua layer uses for its 14 `add_meta_function` call
// sites (previously recreated downstream via an `add_meta_method` shim).
// ---------------------------------------------------------------------------

#[test]
fn test_add_meta_function() -> Result<()> {
    struct Val(i64);

    impl UserData for Val {
        fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
            // Receiver-first shape: `ud - n`.
            methods.add_meta_function(MetaMethod::Sub, |_, (a, b): (AnyUserData, i64)| {
                Ok(a.borrow::<Val>()?.0 - b)
            });
            // Raw shape: Lua may call `__concat` with the userdata as either
            // operand; take both as `Value` and sort it out.
            methods.add_meta_function(MetaMethod::Concat, |lua, (a, b): (Value, Value)| {
                let side = |v: &Value| match v {
                    Value::UserData(u) => u.borrow::<Val>().map(|v| v.0.to_string()),
                    other => Ok(other.to_string().unwrap_or_default()),
                };
                let joined = format!("{}{}", side(&a)?, side(&b)?);
                Ok(lua.create_string(joined)?)
            });
            // The luaur_compat probe: `__index` registered as a meta function
            // must actually be consulted for unknown keys.
            methods.add_meta_function(MetaMethod::Index, |_, (_this, key): (Value, String)| {
                Ok(format!("seen:{key}"))
            });
        }
    }

    let lua = Lua::new();
    lua.globals().set("v", lua.create_userdata(Val(10))?)?;

    assert_eq!(lua.load("return v - 3").eval::<i64>()?, 7);
    assert_eq!(lua.load("return v .. \"!\"").eval::<String>()?, "10!");
    assert_eq!(lua.load("return \"=\" .. v").eval::<String>()?, "=10");
    assert_eq!(
        lua.load("return v.get_type_by_name").eval::<String>()?,
        "seen:get_type_by_name"
    );
    Ok(())
}

#[test]
fn test_add_meta_function_mut() -> Result<()> {
    struct Counted;

    impl UserData for Counted {
        fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
            let mut calls = 0i64;
            methods.add_meta_function_mut(MetaMethod::Call, move |_, _args: MultiValue| {
                calls += 1;
                Ok(calls)
            });
        }
    }

    let lua = Lua::new();
    lua.globals().set("c", lua.create_userdata(Counted)?)?;
    assert_eq!(lua.load("return c()").eval::<i64>()?, 1);
    assert_eq!(lua.load("return c()").eval::<i64>()?, 2);
    Ok(())
}
