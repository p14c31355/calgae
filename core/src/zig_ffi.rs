use std::os::raw::{c_char, c_int, c_void};
use libc::c_float; // c_float を libc からインポート

#[link(name = "quantizer")]
unsafe extern "C" {
    pub fn zig_quantize_model(model_path: *const c_char, bits: i32, output_path: *const c_char) -> i32;
    pub fn zig_quantize_buffer(data: *const c_float, len: usize, bits: i32, output_buffer: *mut c_void) -> isize;
}

#[link(name = "runtime")]
unsafe extern "C" {
    pub fn zig_spawn_thread(entry_fn: extern "C" fn(*mut c_void) -> *mut c_void) -> *mut c_void;
    pub fn zig_join_thread(thread_handle: *mut c_void) -> c_int;
}
