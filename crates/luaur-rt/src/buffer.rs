//! The [`Buffer`] handle (Luau `buffer` type). Mirrors `mlua::Buffer`.
//!
//! A buffer is a fixed-size, mutable, GC-managed byte array — Luau's answer to
//! a `Vec<u8>` / `ArrayBuffer`. Like the other reference-typed handles
//! ([`Table`](crate::Table), [`Function`](crate::Function)) it holds a registry
//! reference ([`LuaRef`]) that keeps the underlying object alive and lets us
//! re-push it onto the stack on demand.
//!
//! The data lives in VM-managed memory; [`Buffer::as_slice`] / [`as_slice_mut`]
//! borrow the raw bytes directly through luaur's `lua_tobuffer`, so reads and
//! writes go straight to the buffer with no copy.

use std::io;

use crate::error::Result;
use crate::state::{Lua, LuaRef};
use crate::sync::XRc;
use crate::sys::*;

/// A Luau buffer type.
///
/// See the buffer [documentation] for more information.
///
/// Mirrors `mlua::Buffer`. Holds a registry reference keeping the buffer alive.
///
/// Under the `send` feature it is `Send + Sync` (all VM access is
/// serialized by the per-VM lock — see [`crate::state::StateRef`]).
///
/// [documentation]: https://luau.org/library#buffer-library
#[derive(Clone)]
pub struct Buffer {
    pub(crate) reference: XRc<LuaRef>,
}

impl Buffer {
    pub(crate) fn from_ref(reference: LuaRef) -> Buffer {
        Buffer {
            reference: XRc::new(reference),
        }
    }

    /// Push this buffer onto the owning state's stack.
    pub(crate) unsafe fn push_to_stack(&self) {
        self.reference.push();
    }

    /// The owning [`Lua`].
    pub(crate) fn lua(&self) -> Lua {
        self.reference.lua()
    }

    /// Copies the buffer data into a new `Vec<u8>`.
    ///
    /// Mirrors `mlua::Buffer::to_vec`.
    pub fn to_vec(&self) -> Vec<u8> {
        self.as_slice().to_vec()
    }

    /// Returns the length of the buffer.
    ///
    /// Mirrors `mlua::Buffer::len`.
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Returns `true` if the buffer is empty.
    ///
    /// Mirrors `mlua::Buffer::is_empty`.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reads given number of bytes from the buffer at the given offset.
    ///
    /// Offset is 0-based.
    ///
    /// Mirrors `mlua::Buffer::read_bytes`.
    #[track_caller]
    pub fn read_bytes<const N: usize>(&self, offset: usize) -> [u8; N] {
        let data = self.as_slice();
        let mut bytes = [0u8; N];
        bytes.copy_from_slice(&data[offset..offset + N]);
        bytes
    }

    /// Writes given bytes to the buffer at the given offset.
    ///
    /// Offset is 0-based.
    ///
    /// Mirrors `mlua::Buffer::write_bytes`.
    #[track_caller]
    pub fn write_bytes(&self, offset: usize, bytes: &[u8]) {
        let data = self.as_slice_mut();
        data[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    /// Returns an adaptor implementing [`io::Read`], [`io::Write`] and
    /// [`io::Seek`] over the buffer.
    ///
    /// Buffer operations are infallible, none of the read/write functions will
    /// return an `Err`.
    ///
    /// Mirrors `mlua::Buffer::cursor`.
    pub fn cursor(self) -> impl io::Read + io::Write + io::Seek {
        BufferCursor(self, 0)
    }

    /// A raw pointer identifying this buffer (for identity comparison).
    /// Mirrors `Value::Buffer(_).to_pointer()`.
    pub(crate) fn to_pointer(&self) -> *const std::ffi::c_void {
        let state = self.reference.state();
        let state = state.get();
        unsafe {
            self.reference.push();
            let p = lua_topointer(state, -1);
            lua_pop(state, 1);
            p
        }
    }

    /// Borrow the buffer's bytes directly (no copy).
    pub(crate) fn as_slice(&self) -> &[u8] {
        unsafe {
            let (buf, size) = self.as_raw_parts();
            std::slice::from_raw_parts(buf, size)
        }
    }

    /// Mutably borrow the buffer's bytes directly (no copy).
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn as_slice_mut(&self) -> &mut [u8] {
        unsafe {
            let (buf, size) = self.as_raw_parts();
            std::slice::from_raw_parts_mut(buf, size)
        }
    }

    /// The raw `(ptr, len)` of the underlying buffer object via luaur's
    /// `lua_tobuffer`. Pushes the buffer, reads the parts, then pops — the
    /// pointer remains valid because the registry ref keeps the object alive.
    unsafe fn as_raw_parts(&self) -> (*mut u8, usize) {
        let state = self.reference.state();
        let state = state.get();
        unsafe {
            self.reference.push();
            let mut size = 0usize;
            let buf = lua_tobuffer(state, -1, &mut size);
            lua_pop(state, 1);
            assert!(!buf.is_null(), "invalid Luau buffer");
            (buf as *mut u8, size)
        }
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Mirror mlua: a buffer renders as its byte contents (a byte slice).
        write!(f, "Buffer({:?})", self.as_slice())
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        // Reference (pointer) identity, matching mlua: two handles are equal iff
        // they point at the *same* buffer object (NOT byte-wise content).
        self.to_pointer() == other.to_pointer()
    }
}

/// Cursor adapter returned by [`Buffer::cursor`]. The `usize` is the current
/// 0-based offset into the buffer. Mirrors mlua's `BufferCursor`.
struct BufferCursor(Buffer, usize);

impl io::Read for BufferCursor {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let data = self.0.as_slice();
        if self.1 == data.len() {
            return Ok(0);
        }
        let len = buf.len().min(data.len() - self.1);
        buf[..len].copy_from_slice(&data[self.1..self.1 + len]);
        self.1 += len;
        Ok(len)
    }
}

impl io::Write for BufferCursor {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let data = self.0.as_slice_mut();
        if self.1 == data.len() {
            return Ok(0);
        }
        let len = buf.len().min(data.len() - self.1);
        data[self.1..self.1 + len].copy_from_slice(&buf[..len]);
        self.1 += len;
        Ok(len)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl io::Seek for BufferCursor {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        let data = self.0.as_slice();
        let new_offset = match pos {
            io::SeekFrom::Start(offset) => offset as i64,
            io::SeekFrom::End(offset) => data.len() as i64 + offset,
            io::SeekFrom::Current(offset) => self.1 as i64 + offset,
        };
        if new_offset < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid seek to a negative position",
            ));
        }
        if new_offset as usize > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid seek to a position beyond the end of the buffer",
            ));
        }
        self.1 = new_offset as usize;
        Ok(self.1 as u64)
    }
}

// ---------------------------------------------------------------------------
// Buffer creation
//
// `lua_newbuffer` allocates a GC buffer and pushes it. When `size` exceeds the
// VM's `MAX_BUFFER_SIZE` (1GB) the underlying `luaM_toobig` *raises* a Lua
// error (longjmp) rather than returning. Unwinding that across Rust frames is
// UB, so we run `lua_newbuffer` inside `lua_pcall` via a small C trampoline:
// the requested size is passed as a number argument, and a raising allocation
// is reported as an ordinary non-zero status with the error on the stack.
// ---------------------------------------------------------------------------

/// C trampoline: stack is `[size]` (a number). Allocates a buffer of that many
/// bytes via `lua_newbuffer`, leaving the buffer object on top.
unsafe fn c_newbuffer(state: *mut lua_State) -> c_int {
    unsafe {
        let size = lua_tonumberx(state, 1, core::ptr::null_mut()) as usize;
        lua_settop(state, 0);
        lua_newbuffer(state, size);
        1
    }
}

/// Create a buffer of `size` zero-initialized bytes, catching an over-limit
/// allocation as an `Err` rather than letting the VM longjmp.
pub(crate) fn create_buffer_with_capacity(lua: &Lua, size: usize) -> Result<Buffer> {
    let state = lua.state();
    let state = state.get();
    unsafe {
        lua_pushcclosurek(
            state,
            Some(c_newbuffer),
            c"luaur-rt-newbuffer".as_ptr(),
            0,
            None,
        );
        lua_pushnumber(state, size as f64);
        let status = lua_pcall(state, 1, 1, 0);
        if status != 0 {
            return Err(lua.pop_error(status));
        }
        Ok(Buffer::from_ref(lua.pop_ref()))
    }
}
