const std = @import("std");

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

// matrix_mult multiplies matrix A (m x n) by matrix B (n x p) and stores the result in C (m x p).
// All matrices are assumed to be in row-major order.
pub export fn matrix_mult(m: usize, n: usize, p: usize, a: [*]const f32, b: [*]const f32, c: [*]f32) void {
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j < p) : (j += 1) {
            var sum: f32 = 0.0;
            var k: usize = 0;
            while (k < n) : (k += 1) {
                sum = @mulAdd(f32, a[i * n + k], b[k * p + j], sum);
            }
            c[i * p + j] = sum;
        }
    }
}

test "add" {
    try std.testing.expectEqual(5, add(2, 3));
}

test "matrix_mult" {
    // Test a 1x3 matrix multiplied by a 3x2 matrix, resulting in a 1x2 matrix.
    const a = [_]f32{ 1.0, 2.0, 3.0 }; // 1x3
    const b = [_]f32{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }; // 3x2
    var c = [_]f32{ 0.0, 0.0 }; // 1x2
    matrix_mult(1, 3, 2, &a, &b, &c);

    // Expected: [1*7+2*9+3*11, 1*8+2*10+3*12] = [7+18+33, 8+20+36] = [58, 64]
    try std.testing.expectApproxEqAbs(c[0], 58.0, 0.001);
    try std.testing.expectApproxEqAbs(c[1], 64.0, 0.001);
}

pub fn main() !void {
    // Remove main for library use, add if needed for testing
}
