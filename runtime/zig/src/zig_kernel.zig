const std = @import("std");

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

// matrix_mult multiplies matrix A (m x n) by matrix B (n x p) and stores the result in C (m x p).
// All matrices are assumed to be in row-major order.
// This version attempts to use SIMD (AVX2) for f32 operations.
// Note: Actual SIMD intrinsics might require specific target features and more complex setup.
// This is a simplified example using Zig's vector types.
pub export fn matrix_mult(m: usize, n: usize, p: usize, a: [*]const f32, b: [*]const f32, c: [*]f32) void {
    const Vec8f = @Vector(8, f32); // 256-bit AVX2 vector for 8 f32s

    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j < p) : (j += 1) {
            var sum_vec = Vec8f{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            var k: usize = 0;
            while (k < n) : (k += 8) { // Process 8 elements at a time
                // Handle remaining elements if n is not a multiple of 8
                if (k + 8 > n) {
                    var sum_scalar: f32 = 0.0;
                    var k_scalar: usize = k;
                    while (k_scalar < n) : (k_scalar += 1) {
                        sum_scalar = @mulAdd(f32, a[i * n + k_scalar], b[k_scalar * p + j], sum_scalar);
                    }
                    c[i * p + j] = sum_scalar; // Store scalar result and break
                    break;
                }

                // Load 8 elements from A and broadcast one element from B
                // @ptrCastでポインタにキャストし、それをVec8fのポインタとして扱い、デリファレンスしてロード
                const a_vec = @as(Vec8f, (@as([*]const Vec8f, @ptrCast(a + (i * n + k))))[0]);
                const b_scalar = b[k * p + j];
                const b_vec = @splat(Vec8f, b_scalar); // Broadcast b_scalar to all elements of b_vec

                sum_vec = a_vec * b_vec + sum_vec;
            }
            // Sum up the elements in sum_vec to get the final scalar sum
            var final_sum: f32 = 0.0;
            for (sum_vec) |val| {
                final_sum += val;
            }
            c[i * p + j] = final_sum;
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
