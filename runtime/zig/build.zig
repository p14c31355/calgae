const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});
    const target = b.standardTargetOptions(.{});

    // Core runtime library
    const runtime_module = b.addModule("runtime", .{
        .root_source_file = b.path("src/runtime.zig"),
        .target = target,
        .optimize = optimize,
    });
    const lib_runtime = b.addLibrary(.{
        .name = "runtime",
        .root_module = runtime_module,
    });
    lib_runtime.linkLibC();
    b.installArtifact(lib_runtime);

    // Quantizer library
    const quantizer_module = b.addModule("quantizer", .{
        .root_source_file = b.path("src/quantizer.zig"),
        .target = target,
        .optimize = optimize,
    });
    const lib_quantizer = b.addLibrary(.{
        .name = "quantizer",
        .root_module = quantizer_module,
    });
    lib_quantizer.linkLibC();
    b.installArtifact(lib_quantizer);

    // Test executable for quantizer
    const test_exe = b.addExecutable(.{
        .name = "test_quantizer",
        .target = target,
        .optimize = optimize,
    });
    test_exe.addCSourceFile(.{
        .file = b.path("test.c"),
        .flags = &[_][]const u8{},
    });
    test_exe.addIncludePath(b.path("src"));
    test_exe.linkLibC();
    test_exe.linkLibrary(lib_quantizer);
    test_exe.linkLibrary(lib_runtime);
    b.installArtifact(test_exe);

    const run_test = b.addRunArtifact(test_exe);
    run_test.addArg("../../../models/dummy_model.bin");

    const test_step = b.step("test", "Run the quantizer test");
    test_step.dependOn(&run_test.step);

    // Set the default step to install all artifacts.
    // `zig build test` will run the tests.
    b.default_step.dependOn(b.getInstallStep());
}
