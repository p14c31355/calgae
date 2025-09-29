use std::os::raw::{c_float, c_int};

#[link(name = "awq")]
extern "C" {
    fn per_channel_max_abs_c(
        abs_output: *mut c_float,
        batch_size: c_int,
        seq_len: c_int,
        hidden_size: c_int,
        out_max: *mut c_float,
    ) -> c_int;

    fn top_k_indices_c(
        act_max: *const c_float,
        hidden_size: c_int,
        top_k: c_int,
        indices: *mut c_int,
    ) -> c_int;

    fn compute_scale_c(
        act_max: *const c_float,
        hidden_size: c_int,
        salient_indices: *const c_int,
        num_salient: c_int,
        scale_out: *mut c_float,
    ) -> c_int;

    fn apply_awq_quantize_c(
        weight: *mut c_float,
        out_dim: c_int,
        in_dim: c_int,
        scale: c_float,
        salient_indices: *const c_int,
        num_salient: c_int,
        bits: c_int,
    ) -> c_int;

    fn compute_smoothquant_scales_c(
        act_max: *const c_float,
        hidden_size: c_int,
        sparsity: c_float,
        bits: c_int,
        scales_out: *mut c_float,
    ) -> c_int;

    fn apply_smoothquant_quantize_c(
        weight: *mut c_float,
        out_dim: c_int,
        in_dim: c_int,
        act_scales: *const c_float,
        bits: c_int,
        group_size: c_int,
    ) -> c_int;
}
