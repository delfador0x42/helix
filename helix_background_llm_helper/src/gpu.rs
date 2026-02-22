//! Zero-dependency Metal compute bindings. Direct objc_msgSend FFI.
//! Only what we need for GPU inference: device, buffers, shaders, dispatch.

#![allow(non_snake_case, clippy::missing_safety_doc)]

use std::ffi::{c_char, c_void, CStr};
use std::ptr;

// ── ObjC runtime FFI ────────────────────────────────────────────────

type Id = *mut c_void;
type Sel = *const c_void;
type Class = *mut c_void;

#[link(name = "objc", kind = "dylib")]
extern "C" {
    fn objc_getClass(name: *const c_char) -> Class;
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_msgSend();          // variadic — we cast to correct fn signature
}

#[link(name = "Metal", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> Id;
}

// ── Cached selectors ────────────────────────────────────────────────
// sel_registerName does a hash lookup (~50ns). We call selectors ~2400 times
// per forward pass. Caching eliminates all hash lookups after first use.

use std::sync::atomic::{AtomicPtr, Ordering};

macro_rules! cached_sel {
    ($name:ident, $sel_str:literal) => {
        #[inline(always)]
        fn $name() -> Sel {
            static CACHED: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
            let p = CACHED.load(Ordering::Relaxed);
            if !p.is_null() { return p as Sel; }
            let s = unsafe { sel_registerName(concat!($sel_str, "\0").as_ptr() as *const c_char) };
            CACHED.store(s as *mut c_void, Ordering::Relaxed);
            s
        }
    };
}

// Hot path selectors (called per encoder, ~200+ times per forward)
cached_sel!(sel_computeCommandEncoder, "computeCommandEncoder");
cached_sel!(sel_retain, "retain");
cached_sel!(sel_release, "release");
cached_sel!(sel_setPipelineState, "setComputePipelineState:");
cached_sel!(sel_setBuffer, "setBuffer:offset:atIndex:");
cached_sel!(sel_setBytes, "setBytes:length:atIndex:");
cached_sel!(sel_dispatchThreadgroups, "dispatchThreadgroups:threadsPerThreadgroup:");
cached_sel!(sel_dispatchThreads, "dispatchThreads:threadsPerThreadgroup:");
cached_sel!(sel_endEncoding, "endEncoding");
cached_sel!(sel_commit, "commit");
cached_sel!(sel_waitUntilCompleted, "waitUntilCompleted");
cached_sel!(sel_commandBuffer, "commandBuffer");
cached_sel!(sel_contents, "contents");
cached_sel!(sel_length, "length");

// Cold path: still use dynamic lookup (called once at init)
#[inline]
fn sel(name: &str) -> Sel {
    unsafe { sel_registerName(name.as_ptr() as *const c_char) }
}

#[inline]
fn cls(name: &str) -> Class {
    unsafe { objc_getClass(name.as_ptr() as *const c_char) }
}

// Cast objc_msgSend to various signatures. All unsafe, all necessary.
macro_rules! msg {
    // () return
    (void $obj:expr, $sel:expr) => {
        unsafe {
            let f: unsafe extern "C" fn(Id, Sel) =
                std::mem::transmute(objc_msgSend as *const ());
            f($obj, $sel)
        }
    };
    // For void with args, use the typed msg_void_* functions directly
    // Id return
    ($obj:expr, $sel:expr) => {
        unsafe {
            let f: unsafe extern "C" fn(Id, Sel) -> Id =
                std::mem::transmute(objc_msgSend as *const ());
            f($obj, $sel)
        }
    };
    // u64 return
    (u64 $obj:expr, $sel:expr) => {
        unsafe {
            let f: unsafe extern "C" fn(Id, Sel) -> u64 =
                std::mem::transmute(objc_msgSend as *const ());
            f($obj, $sel)
        }
    };
}

// Typed msg_send wrappers for specific signatures we actually use.
// Each one casts objc_msgSend to the exact calling convention needed.

unsafe fn msg_id(obj: Id, sel: Sel) -> Id {
    let f: unsafe extern "C" fn(Id, Sel) -> Id =
        std::mem::transmute(objc_msgSend as *const ());
    f(obj, sel)
}

unsafe fn msg_id_id(obj: Id, sel: Sel, a1: Id) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, Id) -> Id =
        std::mem::transmute(objc_msgSend as *const ());
    f(obj, sel, a1)
}

unsafe fn msg_id_id_id_ptr(obj: Id, sel: Sel, a1: Id, a2: Id, a3: *mut Id) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
        std::mem::transmute(objc_msgSend as *const ());
    f(obj, sel, a1, a2, a3)
}

unsafe fn msg_id_id_ptr(obj: Id, sel: Sel, a1: Id, a2: *mut Id) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
        std::mem::transmute(objc_msgSend as *const ());
    f(obj, sel, a1, a2)
}

unsafe fn msg_id_u64_u64(obj: Id, sel: Sel, a1: u64, a2: u64) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, u64, u64) -> Id =
        std::mem::transmute(objc_msgSend as *const ());
    f(obj, sel, a1, a2)
}

unsafe fn msg_id_ptr_u64_u64(obj: Id, sel: Sel, a1: *const c_void, a2: u64, a3: u64) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
        std::mem::transmute(objc_msgSend as *const ());
    f(obj, sel, a1, a2, a3)
}

unsafe fn msg_void(obj: Id, sel: Sel) {
    let f: unsafe extern "C" fn(Id, Sel) =
        std::mem::transmute(objc_msgSend as *const ());
    f(obj, sel)
}

unsafe fn msg_void_id(obj: Id, sel: Sel, a1: Id) {
    let f: unsafe extern "C" fn(Id, Sel, Id) =
        std::mem::transmute(objc_msgSend as *const ());
    f(obj, sel, a1)
}

unsafe fn msg_void_id_u64_u64(obj: Id, sel: Sel, a1: Id, a2: u64, a3: u64) {
    let f: unsafe extern "C" fn(Id, Sel, Id, u64, u64) =
        std::mem::transmute(objc_msgSend as *const ());
    f(obj, sel, a1, a2, a3)
}

unsafe fn msg_void_ptr_u64_u64(obj: Id, sel: Sel, a1: *const c_void, a2: u64, a3: u64) {
    let f: unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64) =
        std::mem::transmute(objc_msgSend as *const ());
    f(obj, sel, a1, a2, a3)
}

unsafe fn msg_void_2size(obj: Id, sel: Sel, a: MTLSize, b: MTLSize) {
    let f: unsafe extern "C" fn(Id, Sel, MTLSize, MTLSize) =
        std::mem::transmute(objc_msgSend as *const ());
    f(obj, sel, a, b)
}

unsafe fn msg_ptr(obj: Id, sel: Sel) -> *mut c_void {
    let f: unsafe extern "C" fn(Id, Sel) -> *mut c_void =
        std::mem::transmute(objc_msgSend as *const ());
    f(obj, sel)
}

unsafe fn msg_u64(obj: Id, sel: Sel) -> u64 {
    let f: unsafe extern "C" fn(Id, Sel) -> u64 =
        std::mem::transmute(objc_msgSend as *const ());
    f(obj, sel)
}

// ── NSString helpers ────────────────────────────────────────────────

unsafe fn nsstring(s: &str) -> Id {
    let cls = cls("NSString\0");
    let alloc: Id = msg_id(cls, sel("alloc\0"));
    let sel = sel("initWithBytes:length:encoding:\0");
    let f: unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
        std::mem::transmute(objc_msgSend as *const ());
    f(alloc, sel, s.as_ptr() as *const c_void, s.len() as u64, 4u64) // UTF8 = 4
}

unsafe fn nsstring_to_str(ns: Id) -> String {
    if ns.is_null() { return String::new(); }
    let cstr: *const c_char = std::mem::transmute(msg_ptr(ns, sel("UTF8String\0")));
    if cstr.is_null() { return String::new(); }
    CStr::from_ptr(cstr).to_string_lossy().into_owned()
}

unsafe fn release(obj: Id) {
    if !obj.is_null() { msg_void(obj, sel_release()); }
}

// ── MTLSize ─────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MTLSize {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

impl MTLSize {
    pub fn new(w: u64, h: u64, d: u64) -> Self {
        MTLSize { width: w, height: h, depth: d }
    }
}

// ── GPU Types (opaque wrappers with Drop) ───────────────────────────

macro_rules! gpu_type {
    ($name:ident) => {
        pub struct $name(Id);
        impl Drop for $name {
            fn drop(&mut self) { unsafe { release(self.0); } }
        }
        impl $name {
            pub fn raw(&self) -> Id { self.0 }
        }
    };
}

gpu_type!(Device);
gpu_type!(Library);
gpu_type!(Function);
gpu_type!(Pipeline);
gpu_type!(Buffer);
gpu_type!(CommandQueue);

// CommandBuffer and Encoder are autoreleased, NOT retained by us
pub struct CommandBuffer(Id);
pub struct ComputeEncoder(Id);

// ── Device ──────────────────────────────────────────────────────────

impl Device {
    pub fn system_default() -> Result<Self, String> {
        let d = unsafe { MTLCreateSystemDefaultDevice() };
        if d.is_null() { return Err("No Metal device".into()); }
        Ok(Device(d))
    }

    pub fn name(&self) -> String {
        unsafe { nsstring_to_str(msg_id(self.0, sel("name\0"))) }
    }

    pub fn recommended_working_set_size(&self) -> u64 {
        unsafe { msg_u64(self.0, sel("recommendedMaxWorkingSetSize\0")) }
    }

    pub fn new_library_with_source(&self, source: &str) -> Result<Library, String> {
        unsafe {
            let src = nsstring(source);
            let opts_cls = cls("MTLCompileOptions\0");
            let opts: Id = msg_id(opts_cls, sel("new\0"));
            let mut err: Id = ptr::null_mut();
            let lib = msg_id_id_id_ptr(
                self.0,
                sel("newLibraryWithSource:options:error:\0"),
                src, opts, &mut err
            );
            release(opts);
            release(src);
            // Only check lib.is_null() — Metal returns non-null error for
            // warnings even when compilation succeeds.
            if lib.is_null() {
                let desc = if !err.is_null() {
                    nsstring_to_str(msg_id(err, sel("localizedDescription\0")))
                } else { "unknown".into() };
                return Err(format!("Shader compile error: {desc}"));
            }
            Ok(Library(lib))
        }
    }

    pub fn new_compute_pipeline(&self, func: &Function) -> Result<Pipeline, String> {
        unsafe {
            let mut err: Id = ptr::null_mut();
            let p = msg_id_id_ptr(
                self.0,
                sel("newComputePipelineStateWithFunction:error:\0"),
                func.0, &mut err
            );
            if p.is_null() {
                let desc = if !err.is_null() {
                    nsstring_to_str(msg_id(err, sel("localizedDescription\0")))
                } else { "unknown".into() };
                return Err(format!("Pipeline error: {desc}"));
            }
            Ok(Pipeline(p))
        }
    }

    pub fn new_buffer(&self, size: u64) -> Buffer {
        unsafe {
            let b = msg_id_u64_u64(
                self.0, sel("newBufferWithLength:options:\0"),
                size, 0u64  // StorageModeShared
            );
            Buffer(b)
        }
    }

    pub fn new_buffer_with_data(&self, data: *const c_void, size: u64) -> Buffer {
        unsafe {
            let b = msg_id_ptr_u64_u64(
                self.0, sel("newBufferWithBytes:length:options:\0"),
                data, size, 0u64
            );
            Buffer(b)
        }
    }

    pub fn new_command_queue(&self) -> CommandQueue {
        unsafe { CommandQueue(msg_id(self.0, sel("newCommandQueue\0"))) }
    }
}

// ── Library ─────────────────────────────────────────────────────────

impl Library {
    pub fn get_function(&self, name: &str) -> Result<Function, String> {
        unsafe {
            let ns = nsstring(name);
            let f = msg_id_id(self.0, sel("newFunctionWithName:\0"), ns);
            release(ns);
            if f.is_null() {
                return Err(format!("Function '{name}' not found"));
            }
            Ok(Function(f))
        }
    }
}

// ── Pipeline ────────────────────────────────────────────────────────

impl Pipeline {
    pub fn thread_execution_width(&self) -> u64 {
        unsafe { msg_u64(self.0, sel("threadExecutionWidth\0")) }
    }

    pub fn max_total_threads_per_threadgroup(&self) -> u64 {
        unsafe { msg_u64(self.0, sel("maxTotalThreadsPerThreadgroup\0")) }
    }
}

// ── Buffer ──────────────────────────────────────────────────────────

impl Buffer {
    pub fn contents(&self) -> *mut c_void {
        unsafe { msg_ptr(self.0, sel_contents()) }
    }

    pub fn length(&self) -> u64 {
        unsafe { msg_u64(self.0, sel_length()) }
    }
}

// ── CommandQueue ────────────────────────────────────────────────────

impl CommandQueue {
    pub fn new_command_buffer(&self) -> CommandBuffer {
        unsafe {
            let cb = msg_id(self.0, sel_commandBuffer());
            let _: Id = msg_id(cb, sel_retain());
            CommandBuffer(cb)
        }
    }
}

// ── CommandBuffer ───────────────────────────────────────────────────

impl CommandBuffer {
    pub fn new_compute_encoder(&self) -> ComputeEncoder {
        unsafe {
            let enc = msg_id(self.0, sel_computeCommandEncoder());
            let _: Id = msg_id(enc, sel_retain());
            ComputeEncoder(enc)
        }
    }

    pub fn commit(&self) {
        unsafe { msg_void(self.0, sel_commit()); }
    }

    pub fn wait_until_completed(&self) {
        unsafe { msg_void(self.0, sel_waitUntilCompleted()); }
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) { unsafe { release(self.0); } }
}

// ── ComputeEncoder ──────────────────────────────────────────────────

impl ComputeEncoder {
    pub fn set_pipeline(&self, pipeline: &Pipeline) {
        unsafe { msg_void_id(self.0, sel_setPipelineState(), pipeline.0); }
    }

    pub fn set_buffer(&self, index: u64, buffer: &Buffer, offset: u64) {
        unsafe { msg_void_id_u64_u64(self.0, sel_setBuffer(), buffer.0, offset, index); }
    }

    pub fn set_bytes(&self, index: u64, data: *const c_void, len: u64) {
        unsafe { msg_void_ptr_u64_u64(self.0, sel_setBytes(), data, len, index); }
    }

    pub fn dispatch_threadgroups(&self, grid: MTLSize, threadgroup: MTLSize) {
        unsafe { msg_void_2size(self.0, sel_dispatchThreadgroups(), grid, threadgroup); }
    }

    pub fn dispatch_threads(&self, grid: MTLSize, threadgroup: MTLSize) {
        unsafe { msg_void_2size(self.0, sel_dispatchThreads(), grid, threadgroup); }
    }

    pub fn end_encoding(&self) {
        unsafe { msg_void(self.0, sel_endEncoding()); }
    }
}

impl Drop for ComputeEncoder {
    fn drop(&mut self) { unsafe { release(self.0); } }
}
