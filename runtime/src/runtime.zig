const std = @import("std");

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

pub export fn matrix_mult(n: usize, p: usize, q: usize, a: [*]const f32, b: [*]const f32, c: [*]f32) void {
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var j: usize = 0;
        while (j < p) : (j += 1) {
            var sum: f32 = 0.0;
            var k: usize = 0;
            while (k < q) : (k += 1) {
                sum += a[i * q + k] * b[k * p + j];
            }
            c[i * p + j] = sum;
        }
    }
}

test "add" {
    try std.testing.expectEqual(5, add(2, 3));
}

test "matrix_mult" {
    const a = [_]f32{1.0, 2.0, 3.0};
    const b = [_]f32{4.0, 5.0, 6.0};
    var c = [_]f32{0.0, 0.0, 0.0};
    matrix_mult(1, 3, 3, &a, &b, &c);
    try std.testing.expectApproxEqAbs(c[0], 32.0, 0.001);
    try std.testing.expectApproxEqAbs(c[1], 0.0, 0.001);
    try std.testing.expectApproxEqAbs(c[2], 0.0, 0.001);
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const stdout = std.io.getStdOut().writer();

    if (args.len > 1) {
        try stdout.print("Hello, {s}!\n", .{args[1]});
    } else {
        try stdout.print("Hello, world!\n", .{});
    }
}
