// Adapted from mlua (https://github.com/mlua-rs/mlua), MIT License,
// © 2019 Aleksandr Orlenko / mlua authors. See tests/ATTRIBUTION.md.
//
// Port of mlua's `tests/send.rs`, gated on the `send` feature.
//
// mlua's `send` feature makes `Lua` and every handle **`Send + Sync`**: the
// interior is serialized by a re-entrant mutex (`ReentrantMutex<RawLua>` in
// mlua), so handles can be shared and used from several threads, one at a time.
// luaur-rt mirrors that exactly: `XRc` (= `Arc` under `send`) + a per-VM
// `ReentrantMutex` acquired by every operation that touches the raw
// `*mut lua_State`. See `crates/luaur-rt/src/sync.rs` and `state.rs`.
//
// Locked down below: compile-time `Send + Sync` assertions for every handle, a
// move-the-VM-to-another-thread test, shared-`&Lua`-across-scoped-threads
// tests (including a contention stress test that would crash or corrupt the VM
// if the lock were absent), and a `Function` captured in an
// `Arc<dyn Fn + Send + Sync>` — the embedding shape (e.g. bevy_mod_scripting's
// `DynamicScriptFunction`) that requires `Sync` handles.

#![cfg(feature = "send")]

use std::thread;

use luaur_rt::{
    AnyUserData, Buffer, Function, Lua, LuaString, RegistryKey, Result, Table, Thread, UserData,
    UserDataMethods, UserDataRef, Value, Vector,
};

// ---------------------------------------------------------------------------
// Compile-time assertions: under the `send` feature `Lua` and every handle is
// `Send`. (`static_assertions::assert_impl_all!` in mlua; written by hand here
// to avoid a new dev-dependency.)
// ---------------------------------------------------------------------------

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn test_lua_is_send_and_sync() {
    // mlua parity: under `send`, `Lua` is both `Send` and `Sync` (all VM
    // access is serialized by the internal per-VM re-entrant mutex).
    assert_send::<Lua>();
    assert_sync::<Lua>();
}

#[test]
fn test_lua_and_handles_are_send_and_sync() {
    assert_send::<Lua>();
    assert_send::<Table>();
    assert_send::<Function>();
    assert_send::<LuaString>();
    assert_send::<AnyUserData>();
    assert_send::<Thread>();
    assert_send::<Buffer>();
    assert_send::<Vector>();
    assert_send::<Value>();
    assert_send::<RegistryKey>();
    assert_send::<luaur_rt::MultiValue>();
    assert_send::<luaur_rt::Error>();
    assert_sync::<Table>();
    assert_sync::<Function>();
    assert_sync::<LuaString>();
    assert_sync::<AnyUserData>();
    assert_sync::<Thread>();
    assert_sync::<Buffer>();
    assert_sync::<Vector>();
    assert_sync::<Value>();
    assert_sync::<RegistryKey>();
    // NOTE: `UserDataRef<'_, T>` is intentionally **not** asserted `Send`. It
    // wraps a `std::cell::Ref` borrow guard (which is `!Send`), and it is a
    // short-lived RAII borrow that must not outlive — let alone cross threads
    // away from — the VM it borrows. Only the long-lived *handles* above are
    // `Send + Sync`. (mlua's `UserDataRef` is `Send` because it uses a
    // `parking_lot` arc-guard; luaur-rt deliberately keeps the simpler
    // `RefCell` borrow.)
}

// ---------------------------------------------------------------------------
// The reason `send` exists: a `Lua` (with everything reachable from it) can be
// constructed on one thread and *moved* to another to be driven there.
// ---------------------------------------------------------------------------

#[test]
fn test_move_lua_to_another_thread() -> Result<()> {
    let lua = Lua::new();

    // A Rust callback capturing `Send` data, registered before the move.
    let captured = String::from("from the spawning thread");
    let f = lua.create_function(move |_, ()| Ok(captured.clone()))?;
    lua.globals().set("greet", f)?;
    lua.globals().set("n", 41i64)?;

    // Move the whole VM (and its handles) into a fresh thread and run there.
    let handle = thread::spawn(move || -> Result<(i64, String)> {
        let n: i64 = lua.load("return n + 1").eval()?;
        let g: String = lua.load("return greet()").eval()?;
        Ok((n, g))
    });

    let (n, g) = handle.join().expect("worker thread panicked")?;
    assert_eq!(n, 42);
    assert_eq!(g, "from the spawning thread");
    Ok(())
}

// ---------------------------------------------------------------------------
// A callback can capture data moved across a thread boundary into the closure
// environment (the `MaybeSend` bound makes the stored box `Send`).
// ---------------------------------------------------------------------------

#[test]
fn test_callback_captures_send_data() -> Result<()> {
    // Build the captured value on a worker thread, then move it into the VM
    // (which lives on the main thread) — proving the closure env is `Send`.
    let payload: Vec<i64> = thread::spawn(|| vec![1, 2, 3, 4])
        .join()
        .expect("worker panicked");

    let lua = Lua::new();
    let sum_fn = lua.create_function(move |_, ()| Ok(payload.iter().sum::<i64>()))?;
    lua.globals().set("sum", sum_fn)?;

    let total: i64 = lua.load("return sum()").eval()?;
    assert_eq!(total, 10);
    Ok(())
}

// ---------------------------------------------------------------------------
// Spirit-port of mlua's `test_userdata_multithread_access_sync` (single-thread
// half): a userdata method that, while borrowing `this`, reaches back into
// globals to call a *second* userdata method — the nested re-entrant access
// the mlua test exercises. The cross-thread half lives in the `Sync` tests
// below.
// ---------------------------------------------------------------------------

struct MyUserData(String);

// This type is `Send + Sync`, exactly like mlua's `MyUserData`.
fn _assert_my_userdata_send_sync()
where
    MyUserData: Send + Sync,
{
}

impl UserData for MyUserData {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("method", |lua, this, ()| {
            // Reach back into globals and invoke another method, while `this`
            // is borrowed — the re-entrant pattern from the mlua test.
            let ud = lua.globals().get::<AnyUserData>("ud")?;
            let method2 = lua
                .load("return function(u) return u:method2() end")
                .eval::<Function>()?;
            method2.call::<()>(ud)?;
            Ok(this.0.clone())
        });

        methods.add_method("method2", |_, _, ()| Ok(()));
    }
}

#[test]
fn test_userdata_nested_method_call() -> Result<()> {
    let lua = Lua::new();

    let ud = lua.create_userdata(MyUserData("hello".to_string()))?;
    lua.globals().set("ud", ud)?;

    // Acquire a shared Rust-side reference (mirrors the mlua test's
    // `UserDataRef` acquisition before driving the VM).
    {
        let any = lua.globals().get::<AnyUserData>("ud")?;
        let r: UserDataRef<'_, MyUserData> = any.borrow::<MyUserData>()?;
        assert_eq!(r.0, "hello");
    }

    // Drive the re-entrant method from Lua.
    let out: String = lua.load("return ud:method()").eval()?;
    assert_eq!(out, "hello");
    Ok(())
}

// ---------------------------------------------------------------------------
// The `Sync` half: sharing one `&Lua` (and handles) across threads. These are
// the tests that would crash, corrupt the VM stack, or trip debug assertions
// if the per-VM lock were absent.
// ---------------------------------------------------------------------------

/// Share `&Lua` across scoped threads; each thread runs chunks and touches the
/// same global table concurrently. The internal lock must serialize them.
#[test]
fn test_shared_lua_across_scoped_threads() -> Result<()> {
    let lua = Lua::new();
    lua.globals().set("counter", 0i64)?;

    thread::scope(|scope| {
        for _ in 0..4 {
            let lua = &lua;
            scope.spawn(move || {
                for _ in 0..250 {
                    lua.load("counter = counter + 1").exec().unwrap();
                }
            });
        }
    });

    let n: i64 = lua.globals().get("counter")?;
    assert_eq!(n, 1000);
    Ok(())
}

/// Hammer one `Table` handle from several threads through the Rust-side API
/// (get/set/len), interleaved with GC cycles — pure handle traffic, no chunks.
#[test]
fn test_shared_table_handle_stress() -> Result<()> {
    let lua = Lua::new();
    let table = lua.create_table();

    thread::scope(|scope| {
        for t in 0..4i64 {
            let table = &table;
            let lua = &lua;
            scope.spawn(move || {
                for i in 0..200i64 {
                    table.set(t * 1000 + i, i).unwrap();
                    let _: i64 = table.get(t * 1000 + i).unwrap();
                    if i % 50 == 0 {
                        lua.gc_collect().unwrap();
                    }
                }
            });
        }
    });

    // Every thread's final write must be present and uncorrupted.
    for t in 0..4i64 {
        let v: i64 = table.get(t * 1000 + 199)?;
        assert_eq!(v, 199);
    }
    Ok(())
}

/// The embedding shape that *requires* `Sync` handles: a Lua `Function` value
/// captured inside an `Arc<dyn Fn + Send + Sync>` host callback container
/// (bevy_mod_scripting's `DynamicScriptFunction`), invoked from another thread.
#[test]
fn test_function_in_send_sync_container() -> Result<()> {
    let lua = Lua::new();
    let double: Function = lua.load("return function(x) return x * 2 end").eval()?;

    let host_cb: std::sync::Arc<dyn Fn(i64) -> i64 + Send + Sync> =
        std::sync::Arc::new(move |x| double.call::<i64>(x).unwrap());

    let from_other_thread = {
        let host_cb = host_cb.clone();
        thread::spawn(move || host_cb(21))
            .join()
            .expect("worker panicked")
    };
    assert_eq!(from_other_thread, 42);
    assert_eq!(host_cb(5), 10);
    Ok(())
}

/// A Rust callback registered in the VM calls back *into* the VM (re-entrant
/// lock on the same thread) while other threads contend for the lock.
#[test]
fn test_reentrant_callback_under_contention() -> Result<()> {
    let lua = Lua::new();
    let f = lua.create_function(|lua, x: i64| {
        // Re-enter the VM from inside the callback: the calling thread already
        // holds the lock, so this must not deadlock.
        let doubled: i64 = lua.load("return ...").call(x * 2)?;
        Ok(doubled)
    })?;
    lua.globals().set("dbl", f)?;

    thread::scope(|scope| {
        for _ in 0..3 {
            let lua = &lua;
            scope.spawn(move || {
                for i in 0..100i64 {
                    let out: i64 = lua.load("return dbl(...)").call(i).unwrap();
                    assert_eq!(out, i * 2);
                }
            });
        }
    });
    Ok(())
}

/// Handles may be *dropped* on a different thread than the VM currently runs
/// on; `LuaRef::drop` takes the lock, so this must neither race nor deadlock.
#[test]
fn test_handle_drop_from_other_thread() -> Result<()> {
    let lua = Lua::new();
    let handles: Vec<Table> = (0..100).map(|_| lua.create_table()).collect();

    let dropper = thread::spawn(move || drop(handles));
    for _ in 0..100 {
        lua.gc_collect()?;
    }
    dropper.join().expect("dropper panicked");
    Ok(())
}

/// App data set on one thread must be visible from another (under `send` the
/// store is process-global, not thread-local).
#[test]
fn test_app_data_visible_across_threads() -> Result<()> {
    let lua = Lua::new();
    lua.set_app_data(7usize);

    thread::scope(|scope| {
        let lua = &lua;
        scope
            .spawn(move || {
                let v = lua
                    .app_data_ref::<usize>()
                    .expect("app data must be visible");
                assert_eq!(*v, 7);
            })
            .join()
            .expect("worker panicked");
    });
    assert_eq!(*lua.app_data_ref::<usize>().unwrap(), 7);
    Ok(())
}
