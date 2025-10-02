use std::os::raw::{c_int, c_float};

#[link(name = "awq", kind = "dylib")] // awqライブラリからインポート
extern "C" {
    pub fn compute_smoothquant_scales_c( // Renamed from original AWQ kernel
        act_max: *const c_float,
        hidden_size: c_int,
        sparsity: c_float,
        bits: c_int,
        scales_out: *mut c_float,
    ) -> c_int; // Added return type for error handling
}
