const std = @import("std");
const libc = @cImport({
    @cInclude("stdint.h");
});
const pthread = @cImport({
    @cInclude("pthread.h");
});

const kernel = @import("zig_kernel.zig");

// OS abstraction layer for FFI to Rust
// Provides file IO, networking, and threading primitives via C ABI

// File IO using std.os for direct fd handling
pub export fn zig_open_file(path_ptr: [*:0]const u8, flags: u32) i32 {
    const mode = std.os.mode_t(0o666);
    const fd = std.os.openatZ(std.math.maxInt(std.os.fd_t), path_ptr, flags, mode) catch return -1;
    return @as(i32, fd);
}

pub export fn zig_close_file(fd: i32) void {
    _ = std.os.close(@as(std.os.fd_t, fd));
}

pub export fn zig_read_file(fd: i32, buffer: [*]u8, len: usize) isize {
    const handle = @as(std.os.fd_t, fd);
    const n = std.os.read(handle, buffer[0..len]) catch return -1;
    return @intCast(n);
}

pub export fn zig_write_file(fd: i32, buffer: [*]const u8, len: usize) isize {
    const handle = @as(std.os.fd_t, fd);
    const n = std.os.write(handle, buffer[0..len]) catch return -1;
    return @intCast(n);
}

// Networking (resolve host and connect TCP)
pub export fn zig_tcp_connect(host_ptr: [*:0]const u8, port: u16) i32 {
    const allocator = std.heap.page_allocator;
    const stream = std.net.tcpConnectToHost(allocator, std.mem.span(host_ptr), port) catch return -1;
    const fd = stream.handle.?.fd;
    stream.close(); // Close the stream object to prevent resource leaks
    return @as(i32, fd);
}

pub export fn zig_tcp_close(fd: i32) void {
    _ = std.os.close(@as(std.os.fd_t, fd));
}

// Threading (simplified, using std.Thread.spawn - assumes caller provides compatible fn)
pub export fn zig_spawn_thread(entry_fn: *const fn() callconv(.C) void) isize {
    const thread = std.Thread.spawn(.{}, entry_fn.*, .{}) catch return -1;
    return @bitCast(thread.getHandle());
}

pub export fn zig_join_thread(handle: isize) void {
    const thread = std.Thread.get(std.thread.Thread.Id.init(@bitCast(handle))) catch return;
    thread.join();
}

pub export fn runtime_matmul(m: usize, n: usize, p: usize, a: [*]const f32, b: [*]const f32, c_out: [*]f32) void {
    kernel.matrix_mult(m, n, p, a, b, c_out);
}
