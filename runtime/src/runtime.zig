const std = @import("std");

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "add" {
    try std.testing.expectEqual(5, add(2, 3));
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
