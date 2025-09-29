const std = @import("std");

pub fn build(b: *std.Build) void {

    // Core runtime library
    const runtime_module = b.createModule(.{
        .root_source_file = b.path("src/runtime.zig"),
        .target = b.resolveTargetQuery(.{}),
        .optimize = b.standardOptimizeOption(.{}),
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
        .target = b.resolveTargetQuery(.{}),
        .optimize = b.standardOptimizeOption(.{}),
    });
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
        // .target and .optimize are set below
    });
    test_exe.setTarget(b.resolveTargetQuery(.{}));
    test_exe.setOptimize(b.standardOptimizeOption(.{}));
    test_exe.linkLibC();
    test_exe.addCSourceFile(.{ .file = b.path("test.c"), .flags = &.{} });
    test_exe.linkLibrary(lib_quantizer);
    test_exe.linkLibrary(lib_runtime);
    b.installArtifact(test_exe);

    const run_test = b.addRunArtifact(test_exe, .{
        .args = &[_][]const u8{"../../../models/dummy_model.bin"},
    });

    const test_step = b.step("test", "Run the quantizer test");
    test_step.dependOn(&run_test.step);

    // Set the default step to install all artifacts.
    // `zig build test` will run the tests.
    b.default_step.dependOn(&b.getInstallStep().step);
}
