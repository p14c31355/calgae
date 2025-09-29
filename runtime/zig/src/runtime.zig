const std = @import("std");
const libc = @cImport({
    @cInclude("stdint.h");
    @cInclude("pthread.h");
});

const kernel = @import("zig_kernel.zig");

// OS abstraction layer for FFI to Rust
// Provides file IO, networking, and threading primitives via C ABI

// File IO using std.os for direct fd handling
pub export fn zig_open_file(path_ptr: [*:0]const u8, flags: u32) libc.c_int {
    const mode = std.os.mode_t(0o666);
    const fd = std.os.openatZ(std.math.maxInt(std.os.fd_t), path_ptr, flags, mode) catch |err| return -1;
    return @as(libc.c_int, fd);
}

pub export fn zig_close_file(fd: libc.c_int) void {
    _ = std.os.close(@as(std.os.fd_t, fd));
}

pub export fn zig_read_file(fd: libc.c_int, buffer: [*]u8, len: usize) isize {
    const handle = @as(std.os.fd_t, fd);
    const n = std.os.read(handle, buffer[0..len]) catch |err| return -1;
    return @intCast(isize, n);
}

pub export fn zig_write_file(fd: libc.c_int, buffer: [*]const u8, len: usize) isize {
    const handle = @as(std.os.fd_t, fd);
    const n = std.os.write(handle, buffer[0..len]) catch |err| return -1;
    return @intCast(isize, n);
}

// Networking (resolve host and connect TCP)
pub export fn zig_tcp_connect(host_ptr: [*:0]const u8, port: u16) libc.c_int {
    const allocator = std.heap.page_allocator;
    const stream = std.net.tcpConnectToHost(allocator, std.mem.span(host_ptr), port) catch |err| return -1;
    return @as(libc.c_int, stream.handle.?.fd);
}

pub export fn zig_tcp_close(fd: libc.c_int) void {
    _ = std.os.close(@as(std.os.fd_t, fd));
}

// Threading
pub export fn zig_spawn_thread(fn_ptr: *const fn () callconv(.C) void) isize {
    var native: libc.pthread_t = undefined;
    const ptr = @ptrCast(?*libc.c_void, fn_ptr);
    const rc: libc.c_int = libc.pthread_create(&native, null, struct {
        fn thread_main(ptr: ?*libc.c_void) ?*libc.c_void {
            const fn_main = @ptrCast(*const fn () callconv(.C) void, ptr.?);
            fn_main.*();
            return null;
        }
    }.thread_main, ptr);
    if (rc != 0) return -1;
    return @bitCast(isize, native);
}

pub export fn zig_join_thread(tid: isize) void {
    const native: libc.pthread_t = @bitCast(tid);
    _ = libc.pthread_join(native, null);
}

pub export fn runtime_matmul(m: usize, n: usize, p: usize, a: [*]const f32, b: [*]const f32, c: [*]f32) void {
    kernel.matrix_mult(m, n, p, a, b, c);
}
