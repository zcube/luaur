use crate::functions::enumedge::enumedge;
use crate::functions::enumedges::enumedges;
use crate::functions::enumnode::enumnode;
use crate::macros::clvalue::clvalue;
use crate::macros::getstr::getstr;
use crate::macros::lua_idsize::LUA_IDSIZE;
use crate::macros::obj_2_gco::obj2gco;
use crate::macros::ttisfunction::ttisfunction;
use crate::records::call_info::CallInfo;
use crate::records::enum_context::EnumContext;
use crate::records::gc_object::GCObject;
use crate::records::lua_state::lua_State;
use crate::records::proto::Proto;
use crate::records::t_string::TString;
use crate::type_aliases::closure::Closure;
use crate::type_aliases::t_value::TValue;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::ffi::{c_char, CStr};

unsafe fn format_thread_label(
    debugname: *const c_char,
    linedefined: i32,
    source: *const c_char,
) -> Vec<u8> {
    let debugname = CStr::from_ptr(debugname).to_bytes();
    let source = CStr::from_ptr(source).to_bytes();
    let line = linedefined.to_string();

    let mut label = Vec::with_capacity(
        b"thread at ".len() + debugname.len() + 1 + line.len() + 1 + source.len() + 1,
    );
    label.extend_from_slice(b"thread at ");
    label.extend_from_slice(debugname);
    label.push(b':');
    label.extend_from_slice(line.as_bytes());
    label.push(b' ');
    label.extend_from_slice(source);
    label.push(0);
    label
}

#[allow(non_snake_case)]
pub unsafe fn enumthread(ctx: *mut EnumContext, th: *mut lua_State) {
    let size = core::mem::size_of::<lua_State>()
        + core::mem::size_of::<TValue>() * (*th).stacksize as usize
        + core::mem::size_of::<CallInfo>() * (*th).size_ci as usize;

    let mut tcl: *mut Closure = core::ptr::null_mut();
    let mut ci: *mut CallInfo = (*th).base_ci;
    while ci <= (*th).ci {
        if ttisfunction!((*ci).func) {
            tcl = clvalue!((*ci).func);
            break;
        }
        ci = ci.wrapping_add(1);
    }

    if !tcl.is_null() && (*tcl).isC == 0 {
        let tcl_l = core::ptr::addr_of!((*tcl).inner.l).cast::<crate::records::closure::LClosure>();
        let p: *mut Proto = (*tcl_l).p;
        if !p.is_null() {
            let src_str = if !(*p).source.is_null() {
                getstr((*p).source)
            } else {
                c"unnamed".as_ptr()
            };
            let debugname_str = if !(*p).debugname.is_null() {
                getstr((*p).debugname)
            } else {
                c"unnamed".as_ptr()
            };

            let label = format_thread_label(debugname_str, (*p).linedefined, src_str);

            enumnode(
                ctx,
                obj2gco!(th as *mut GCObject),
                size,
                label.as_ptr() as *const c_char,
            );
        } else {
            enumnode(ctx, obj2gco!(th as *mut GCObject), size, core::ptr::null());
        }
    } else {
        enumnode(ctx, obj2gco!(th as *mut GCObject), size, core::ptr::null());
    }

    enumedge(
        ctx,
        obj2gco!(th as *mut GCObject),
        obj2gco!((*th).gt as *mut GCObject),
        c"globals".as_ptr(),
    );

    if (*th).top > (*th).stack {
        enumedges(
            ctx,
            obj2gco!(th as *mut GCObject),
            (*th).stack,
            (*th).top.offset_from((*th).stack) as usize,
            c"stack".as_ptr(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::format_thread_label;

    #[test]
    fn thread_label_preserves_source_bytes_and_is_nul_terminated() {
        let debugname = b"render\xff\0";
        let source = b"canvas.luau\0";
        let label = unsafe {
            format_thread_label(
                debugname.as_ptr() as *const core::ffi::c_char,
                -42,
                source.as_ptr() as *const core::ffi::c_char,
            )
        };

        assert_eq!(label, b"thread at render\xff:-42 canvas.luau\0");
    }
}
