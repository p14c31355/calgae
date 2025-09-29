const std = @import("std");

pub fn build(b: *std.Build) void {

    // Core runtime library
    const runtime_module = b.createModule(.{
        .root_source_file = b.path("src/runtime.zig"),
    });
    const lib_runtime = b.addLibrary(.{
        .name = "runtime",
        .root_module = runtime_module,
        .linkage = .static,
    });
    lib_runtime.linkLibC();
    b.installArtifact(lib_runtime);

    // Quantizer library
    const quantizer_module = b.createModule(.{
        .root_source_file = b.path("src/quantizer.zig"),
    });
    quantizer_module.addImport("runtime", runtime_module);
    const lib_quantizer = b.addLibrary(.{
        .name = "quantizer",
        .root_module = quantizer_module,
        .linkage = .static,
    });
    lib_quantizer.linkLibC();
    b.installArtifact(lib_quantizer);

    // Test executable for quantizer
    const test_exe = b.addExecutable(.{
        .name = "test_quantizer",
        .root_module = b.createModule(.{}),
    });
    test_exe.addCSourceFile(.{
        .file = b.path("test.c"),
        .flags = &[_][]const u8{},
    });
    test_exe.linkLibC();
    test_exe.linkLibrary(lib_quantizer);
    test_exe.linkLibrary(lib_runtime);
    b.installArtifact(test_exe);

    const run_test = b.addRunArtifact(test_exe);
    const test_step = b.step("test", "Run the quantizer test");
    test_step.dependOn(&run_test.step);
    if (b.args) |_| {
        const test_arg = b.option(bool, "test", "Run the test");
        if (test_arg == true) {
            b.default_step = test_step;
        }
    } else {
        b.default_step = test_step;
    }
}
