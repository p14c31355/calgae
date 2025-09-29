const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.*.standardTargetOptions(.{});
    const optimize = b.*.standardOptimizeOption(.{});

    // Core runtime library
    const lib_runtime = b.*.addStaticLibrary(.{
        .name = "runtime",
        .root_source_file = b.path("src/runtime.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_runtime.export_symbol_names = &.{
        "zig_open_file", "zig_close_file", "zig_read_file", "zig_write_file",
        "zig_tcp_connect", "zig_tcp_close", "zig_spawn_thread", "zig_join_thread",
    };
    b.installArtifact(lib_runtime);

    // Quantizer library
    const lib_quantizer = b.*.addStaticLibrary(.{
        .name = "quantizer",
        .root_source_file = b.path("src/quantizer.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_quantizer.root_module.addImport("runtime", b.createModule(.{ .source_file = b.path("src/runtime.zig") }));
    lib_quantizer.export_symbol_names = &.{"zig_quantize_model"};
    b.installArtifact(lib_quantizer);

    // Llama.cpp integration (assume llama.cpp source is in external/llama.cpp/)
    // Add runCmake or exe for building llama as static lib
    const llama_mode = b.*.addStaticLibrary(.{
        .name = "ggml",
        .target = target,
        .optimize = optimize,
    });

    const llama_step = b.*.addRunArtifactStep(.{
        .artifact = b.*.addSystemCommand(.{
            .directories = .{ .cwd = .{ .path = "external/llama.cpp" } },
            .command = "make",
            .arguments = &.{"libggml", "-j"},
        }),
    });
    llama_step.step.dependOn(&b.getInstallStep());
    llama_mode.step.dependOn(&llama_step.step);

    lib_runtime.linkSystemLibrary("ggml");
    lib_quantizer.linkSystemLibrary("ggml");

    // Mojo integration (assume Mojo libs are prebuilt or compiled)
    // For now, link against system if available; extend for custom build
    // lib_runtime.linkSystemLibrary("mojo");

    // Engine wrapper (llama)
    const lib_llama_wrapper = b.*.addStaticLibrary(.{
        .name = "llama_wrapper",
        .root_source_file = .{ .path = "../engine/llama_wrapper.zig" },
        .target = target,
        .optimize = optimize,
    });
    lib_llama_wrapper.linkSystemLibrary("ggml");
    lib_llama_wrapper.export_symbol_names = &.{
        "llama_init_from_file", "llama_load_model_from_file", "llama_free_model",
        "llama_free", "llama_eval", "llama_token_to_str", "llama_tokenize",
    };
    b.installArtifact(lib_llama_wrapper);
}
