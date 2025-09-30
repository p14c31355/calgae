use std::os::raw::{c_char, c_float, c_int, c_void};

// #[link(name = "awq")]
// unsafe extern "C" {
//     // AWQ and SmoothQuant functions (commented for Candle-only build)
//     // fn per_channel_max_abs_c(
//     //     abs_output: *mut c_float,
//     //     batch_size: c_int,
//     //     seq_len: c_int,
//     //     hidden_size: c_int,
//     //     out_max: *mut c_float,
//     // ) -> c_int;
//     //
//     // fn top_k_indices_c(
//     //     act_max: *const c_float,
//     //     hidden_size: c_int,
//     //     top_k: c_int,
//     //     indices: *mut c_int,
//     // ) -> c_int;
//     //
//     // fn compute_scale_c(
//     //     act_max: *const c_float,
//     //     hidden_size: c_int,
//     //     salient_indices: *const c_int,
//     //     num_salient: c_int,
//     //     scale_out: *mut c_float,
//     // ) -> c_int;
//     //
//     // fn apply_awq_quantize_c(
//     //     weight: *mut c_float,
//     //     out_dim: c_int,
//     //     in_dim: c_int,
//     //     scale: c_float,
//     //     salient_indices: *const c_int,
//     //     num_salient: c_int,
//     //     bits: c_int,
//     // ) -> c_int;
//     //
//     // fn compute_smoothquant_scales_c(
//     //     act_max: *const c_float,
//     //     hidden_size: c_int,
//     //     sparsity: c_float,
//     //     bits: c_int,
//     //     scales_out: *mut c_float,
//     // ) -> c_int;
//     //
//     // fn apply_smoothquant_quantize_c(
//     //     weight: *mut c_float,
//     //     out_dim: c_int,
//     //     in_dim: c_int,
//     //     act_scales: *const c_float,
//     //     bits: c_int,
//     //     group_size: c_int,
//     // ) -> c_int;
// }

/* Zig runtime FFI - commented for Candle-only build
#[link(name = "runtime")]
unsafe extern "C" {
    fn zig_open_file(path: *const c_char, flags: u32) -> c_int;
    fn zig_close_file(fd: c_int);
    fn zig_read_file(fd: c_int, buffer: *mut c_void, len: usize) -> isize;
    fn zig_write_file(fd: c_int, buffer: *const c_void, len: usize) -> isize;
    fn zig_tcp_connect(host: *const c_char, port: u16) -> c_int;
    fn zig_tcp_close(fd: c_int);
    fn zig_spawn_thread(entry: extern "C" fn()) -> c_int;
    fn zig_join_thread(tid: c_int);
}
*/

#[link(name = "quantizer")]
unsafe extern "C" {
    fn zig_quantize_model(model_path: *const c_char, bits: i32, output_path: *const c_char) -> i32;
}

/* Zig llama wrapper FFI (opaque types) - commented for Candle-only build
#[link(name = "llama_wrapper")]
unsafe extern "C" {
    // extern type is unstable, use struct {} for opaques or skip
    type llama_context_params;
    type llama_model_params;
    type llama_context;
    type llama_model;
    type llama_token;

    fn llama_init_from_file(path: *const c_char, params: *mut llama_context_params) -> *mut llama_context;
    fn llama_load_model_from_file(path: *const c_char, params: *mut llama_model_params) -> *mut llama_model;
    fn llama_free_model(model: *mut llama_model);
    fn llama_free(ctx: *mut llama_context);
    fn llama_eval(ctx: *mut llama_context, tokens: *const llama_token, n_tokens: c_int, n_past: c_int, logits: *mut c_float) -> c_int;
    fn llama_token_to_str(ctx: *mut llama_context, token: llama_token, buf: *mut c_char, length: c_int) -> c_int;
    fn llama_tokenize(ctx: *mut llama_context, text: *const c_char, tokens: *mut llama_token, n_max_tokens: c_int, add_special: c_int) -> c_int;
}
*/
