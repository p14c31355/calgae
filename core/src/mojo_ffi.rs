use std::os::raw::{c_int, c_float};

#[link(name = "awq", kind = "dylib")] // awqライブラリからインポート
extern "C" {
    pub fn compute_smoothquant_scales_c( // 関数名を変更
        act_max: *const c_float,
        hidden_size: c_int,
        sparsity: c_float,
        bits: c_int,
        scales_out: *mut c_float,
    ) -> c_int; // 戻り値の型を追加
}
