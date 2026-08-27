//! The [`Chunk`] builder returned by [`Lua::load`]. Mirrors `mlua::Chunk`.
//!
//! Compilation reuses the same machinery as the umbrella `luaur` crate's
//! `compile`/`eval` helpers: source -> `luaur_compiler::compile` -> bytecode ->
//! `luau_load` -> a Lua function on the stack -> `lua_pcall`.

use crate::error::{Error, Result};
use crate::function::Function;
use crate::state::Lua;
use crate::sys::*;
use crate::traits::FromLuaMulti;

use luaur_ast::records::parse_options::ParseOptions;
use luaur_bytecode::records::bytecode_encoder::BytecodeEncoder;
use luaur_compiler::functions::compile::compile as compiler_compile;
use luaur_compiler::records::compile_options::CompileOptions;

/// No-op bytecode encoder (same as the umbrella crate's `NoopEncoder`).
struct NoopEncoder;
impl BytecodeEncoder for NoopEncoder {
    fn encode(&mut self, _data: &mut [u32]) {}
}

/// Input accepted by [`Lua::load`]: source text as a string or as raw bytes.
///
/// Mirrors the text half of `mlua::AsChunk` (mlua additionally accepts
/// precompiled bytecode and file paths; luaur-rt's high-level loader is
/// text-only, see [`ChunkMode`]). Bytes are validated as UTF-8 — luaur's
/// parser operates on `str` — and an invalid sequence is deferred into the
/// [`Chunk`], surfacing as a [`Error::SyntaxError`] when the chunk is
/// compiled (exactly where mlua's Luau parser would reject it).
pub trait AsChunk {
    /// The chunk's source text, or the position of the first invalid UTF-8
    /// byte (reported when the chunk is compiled).
    fn into_chunk_source(self) -> std::result::Result<String, std::str::Utf8Error>;
}

impl AsChunk for &str {
    fn into_chunk_source(self) -> std::result::Result<String, std::str::Utf8Error> {
        Ok(self.to_string())
    }
}

impl AsChunk for String {
    fn into_chunk_source(self) -> std::result::Result<String, std::str::Utf8Error> {
        Ok(self)
    }
}

impl AsChunk for &String {
    fn into_chunk_source(self) -> std::result::Result<String, std::str::Utf8Error> {
        Ok(self.clone())
    }
}

impl AsChunk for std::borrow::Cow<'_, str> {
    fn into_chunk_source(self) -> std::result::Result<String, std::str::Utf8Error> {
        Ok(self.into_owned())
    }
}

impl AsChunk for &[u8] {
    fn into_chunk_source(self) -> std::result::Result<String, std::str::Utf8Error> {
        std::str::from_utf8(self).map(str::to_string)
    }
}

impl AsChunk for &Vec<u8> {
    fn into_chunk_source(self) -> std::result::Result<String, std::str::Utf8Error> {
        self.as_slice().into_chunk_source()
    }
}

impl AsChunk for Vec<u8> {
    fn into_chunk_source(self) -> std::result::Result<String, std::str::Utf8Error> {
        match String::from_utf8(self) {
            Ok(s) => Ok(s),
            Err(e) => Err(e.utf8_error()),
        }
    }
}

impl AsChunk for std::borrow::Cow<'_, [u8]> {
    fn into_chunk_source(self) -> std::result::Result<String, std::str::Utf8Error> {
        match self {
            std::borrow::Cow::Borrowed(b) => b.into_chunk_source(),
            std::borrow::Cow::Owned(v) => v.into_chunk_source(),
        }
    }
}

/// A not-yet-executed piece of Lua source.
///
/// Mirrors `mlua::Chunk`. Produced by [`Lua::load`]; finalized with
/// [`Chunk::exec`], [`Chunk::eval`], or [`Chunk::into_function`].
pub struct Chunk {
    pub(crate) lua: Lua,
    /// The source text, or the deferred invalid-UTF-8 error from a bytes
    /// [`AsChunk`] input (reported by [`Chunk::compile`]).
    pub(crate) source: std::result::Result<String, std::str::Utf8Error>,
    pub(crate) name: String,
    /// Optional environment table applied to the loaded function. Mirrors the
    /// per-chunk environment set by `mlua::Chunk::set_environment`.
    pub(crate) environment: Option<crate::table::Table>,
    /// Optional per-chunk compiler. Mirrors `mlua::Chunk::set_compiler`. When
    /// `None`, the VM-default compiler (`Lua::set_compiler`) is used, falling
    /// back to luaur's default options.
    pub(crate) compiler: Option<crate::compiler::Compiler>,
}

impl Chunk {
    /// Override the chunk name shown in error messages / tracebacks.
    ///
    /// Mirrors `mlua::Chunk::set_name`.
    pub fn set_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the environment (globals table) the loaded chunk runs in.
    ///
    /// Mirrors `mlua::Chunk::set_environment`. Applied to the function produced
    /// by [`Chunk::into_function`] / [`Chunk::exec`] / [`Chunk::eval`].
    pub fn set_environment(mut self, env: crate::table::Table) -> Self {
        self.environment = Some(env);
        self
    }

    /// Set the [`Compiler`](crate::Compiler) used to compile this chunk.
    /// Mirrors `mlua::Chunk::set_compiler`.
    pub fn set_compiler(mut self, compiler: crate::compiler::Compiler) -> Self {
        self.compiler = Some(compiler);
        self
    }

    /// Compile the chunk and call it with `args`, converting the result to `R`.
    /// Mirrors `mlua::Chunk::call`.
    pub fn call<R: FromLuaMulti>(self, args: impl crate::traits::IntoLuaMulti) -> Result<R> {
        self.into_function()?.call::<R>(args)
    }

    /// The current chunk mode is always text here (luaur-rt loads source).
    /// Mirrors `mlua::Chunk::set_mode` as a no-op accepting `mlua::ChunkMode`'s
    /// role: luaur-rt auto-detects, so this is provided only for signature
    /// parity and returns `self` unchanged.
    pub fn set_mode(self, _mode: ChunkMode) -> Self {
        self
    }

    /// The chunk name used for error messages / tracebacks.
    ///
    /// Mirrors `mlua::Chunk::name`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Compile the source to bytecode (or return a [`Error::SyntaxError`]).
    fn compile(&self) -> Result<Vec<u8>> {
        // Pick the effective compiler: a per-chunk one wins over the VM-default
        // one (`Lua::set_compiler`); otherwise use luaur's default options.
        let effective = self.compiler.clone().or_else(|| self.lua.vm_compiler());
        let mut scratch: Vec<*const core::ffi::c_char> = Vec::new();
        let options = match &effective {
            Some(c) => c.to_options(&mut scratch),
            None => CompileOptions::default(),
        };
        let parse_options = ParseOptions::default();
        let mut encoder = NoopEncoder;
        let owned = match &self.source {
            Ok(s) => s.clone(),
            Err(e) => return Err(invalid_utf8_error(e)),
        };
        let blob = compiler_compile(
            &owned,
            &options,
            &parse_options,
            &mut encoder as *mut dyn BytecodeEncoder,
        );
        let bytes = blob.into_bytes();
        // A leading 0 byte is the compiler's error marker.
        if bytes.first() == Some(&0u8) {
            let message = String::from_utf8_lossy(&bytes[1..]).into_owned();
            // Luau reports a syntax error caused by hitting end-of-input (an
            // unterminated block/expression) with a "got <eof>" suffix; mlua
            // surfaces that as `incomplete_input: true` so a REPL can keep
            // reading. Detect it the same way.
            let incomplete_input = message.contains("<eof>");
            return Err(Error::SyntaxError {
                message,
                incomplete_input,
            });
        }
        Ok(bytes)
    }

    /// Load the compiled chunk and leave the resulting function on top of the
    /// stack, returning a [`Function`] handle.
    ///
    /// Mirrors `mlua::Chunk::into_function`.
    pub fn into_function(self) -> Result<Function> {
        let bytecode = self.compile()?;
        let state = self.lua.state();
        let state = state.get();
        unsafe {
            let chunkname = std::ffi::CString::new(format!("={}", self.name))
                .unwrap_or_else(|_| std::ffi::CString::new("=chunk").unwrap());
            let rc = luau_load(
                state,
                chunkname.as_ptr(),
                bytecode.as_ptr() as *const c_char,
                bytecode.len(),
                0,
            );
            if rc != 0 {
                // luau_load failure leaves an error message on the stack.
                return Err(self.lua.pop_error(rc));
            }
            let func = Function::from_ref(self.lua.pop_ref());
            if let Some(env) = &self.environment {
                func.set_environment(env.clone())?;
            }
            Ok(func)
        }
    }

    /// Statically type-check this chunk's source against the owning [`Lua`]'s
    /// accumulated host definitions (the `typecheck` feature).
    ///
    /// Returns `Ok(())` when the source type-checks clean, or
    /// [`Error::TypeError`](crate::Error::TypeError) carrying the structured
    /// diagnostics otherwise. Because Luau is dynamically typed, the check is
    /// advisory — it composes with `?` ahead of [`Chunk::exec`] / [`Chunk::eval`]
    /// without changing what running the chunk does:
    ///
    /// ```
    /// # #[cfg(feature = "typecheck")] {
    /// # use luaur_rt::Lua;
    /// let lua = Lua::new();
    /// let c = lua.load("local x: number = 1\nreturn x");
    /// c.check().unwrap();
    /// # }
    /// ```
    #[cfg(feature = "typecheck")]
    #[cfg_attr(docsrs, doc(cfg(feature = "typecheck")))]
    pub fn check(&self) -> Result<()> {
        match &self.source {
            Ok(s) => self.lua.check(s),
            Err(e) => Err(invalid_utf8_error(e)),
        }
    }

    /// Run the chunk for its side effects, discarding return values.
    ///
    /// Mirrors `mlua::Chunk::exec`.
    pub fn exec(self) -> Result<()> {
        let f = self.into_function()?;
        f.call::<()>(())
    }

    /// Run the chunk and convert its return value(s) to `R`.
    ///
    /// Mirrors `mlua::Chunk::eval`. Like the Lua REPL (and mlua), the source is
    /// first tried as an *expression* (by prepending `return `); if that
    /// compiles it is used, otherwise the chunk is run as a statement block.
    /// This is what lets `lua.load("coroutine.create(f)").eval::<Thread>()` and
    /// `lua.load("function() ... end").eval::<Function>()` work.
    pub fn eval<R: FromLuaMulti>(self) -> Result<R> {
        // Try the expression form first.
        let expr = Chunk {
            lua: self.lua.clone(),
            source: self.source.clone().map(|s| format!("return {s}")),
            name: self.name.clone(),
            environment: self.environment.clone(),
            compiler: self.compiler.clone(),
        };
        if let Ok(f) = expr.into_function() {
            return f.call::<R>(());
        }
        // Fall back to statement-block mode.
        let f = self.into_function()?;
        f.call::<R>(())
    }

    /// Asynchronously load the chunk and call it with `args` (the `async`
    /// feature). Mirrors `mlua::Chunk::call_async`.
    #[cfg(feature = "async")]
    #[cfg_attr(docsrs, doc(cfg(feature = "async")))]
    pub async fn call_async<R>(self, args: impl crate::traits::IntoLuaMulti) -> Result<R>
    where
        R: FromLuaMulti,
    {
        self.into_function()?.call_async(args).await
    }

    /// Asynchronously run the chunk for its side effects (the `async` feature).
    /// Mirrors `mlua::Chunk::exec_async`.
    #[cfg(feature = "async")]
    #[cfg_attr(docsrs, doc(cfg(feature = "async")))]
    pub async fn exec_async(self) -> Result<()> {
        self.call_async(()).await
    }

    /// Asynchronously evaluate the chunk as an expression (or block) and convert
    /// the result to `R` (the `async` feature). Mirrors `mlua::Chunk::eval_async`.
    ///
    /// Like [`Chunk::eval`], the source is first tried as an expression (by
    /// prepending `return `); if that compiles it is driven, otherwise the chunk
    /// runs as a statement block.
    #[cfg(feature = "async")]
    #[cfg_attr(docsrs, doc(cfg(feature = "async")))]
    pub async fn eval_async<R>(self) -> Result<R>
    where
        R: FromLuaMulti,
    {
        // Try the expression form first (mirrors `eval`).
        let expr = Chunk {
            lua: self.lua.clone(),
            source: self.source.clone().map(|s| format!("return {s}")),
            name: self.name.clone(),
            environment: self.environment.clone(),
            compiler: self.compiler.clone(),
        };
        if let Ok(f) = expr.into_function() {
            return f.call_async::<R>(()).await;
        }
        let f = self.into_function()?;
        f.call_async::<R>(()).await
    }
}

/// How a chunk's bytes are interpreted. Mirrors `mlua::ChunkMode`. luaur-rt
/// always loads text source, so this exists for signature parity with
/// [`Chunk::set_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkMode {
    /// Text source (the default and only supported mode here).
    Text,
    /// Precompiled bytecode (not supported by luaur-rt's high-level loader).
    Binary,
}

/// The deferred error for byte chunks that are not valid UTF-8: shaped as a
/// [`Error::SyntaxError`], since that is where mlua's Luau parser (which takes
/// the raw bytes) would reject the same input.
fn invalid_utf8_error(e: &std::str::Utf8Error) -> Error {
    Error::SyntaxError {
        message: format!("chunk source is not valid UTF-8: {e}"),
        incomplete_input: false,
    }
}
