const std = @import("std");

// Thin Zig wrapper for llama.cpp C API
// Assumes llama.cpp is built and llama.h is available.
// This provides C ABI exports for Rust to use via C FFI.

const c_llama = @cImport({
    @cInclude("llama.h");
});

pub export fn llama_init_from_file(path_ptr: [*:0]const u8, params_ptr: ?*c_llama.llama_context_params) ?*c_llama.llama_context {
    const path = std.mem.span(path_ptr);
    const params = if (params_ptr) |p| p else &c_llama.llama_context_default_params();
    return c_llama.llama_init_from_file(path, params);
}

pub export fn llama_load_model_from_file(path_ptr: [*:0]const u8, params_ptr: ?*c_llama.llama_model_params) ?*c_llama.llama_model {
    const path = std.mem.span(path_ptr);
    const params = if (params_ptr) |p| p else &c_llama.llama_model_default_params();
    return c_llama.llama_load_model_from_file(path, params);
}

pub export fn llama_free_model(model_ptr: ?*c_llama.llama_model) void {
    if (model_ptr) |model| c_llama.llama_free_model(model);
}

pub export fn llama_free(ctx_ptr: ?*c_llama.llama_context) void {
    if (ctx_ptr) |ctx| c_llama.llama_free(ctx);
}

pub export fn llama_eval(ctx_ptr: ?*c_llama.llama_context, tokens: [*]const c_llama.llama_token, n_tokens: c_int, n_past: c_int, logits: ?[*]f32) c_int {
    return c_llama.llama_eval(ctx_ptr orelse null, tokens, n_tokens, n_past, logits orelse null);
}

pub export fn llama_token_to_str(ctx_ptr: ?*c_llama.llama_context, token: c_llama.llama_token, buf: [*]u8, length: c_int) c_int {
    return c_llama.llama_token_to_str(ctx_ptr orelse null, token, @ptrCast(buf), length);
}

pub export fn llama_tokenize(ctx_ptr: ?*c_llama.llama_context, text: [*:0]const u8, tokens: [*]c_llama.llama_token, n_max_tokens: c_int, add_special: bool) c_int {
    return c_llama.llama_tokenize(ctx_ptr orelse null, text, tokens, n_max_tokens, if (add_special) 1 else 0);
}

// Additional wrappers can be added as needed for quantization, etc.
