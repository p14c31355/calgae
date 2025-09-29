const std = @import("std");
const c = @cImport({
    @cInclude("stdint.h");
});

// OS abstraction layer for FFI to Rust
// Provides file IO, networking, and threading primitives via C ABI

// File IO using std.os for direct fd handling
pub export fn zig_open_file(path_ptr: [*:0]const u8, flags: u32) c_int {
    const path = std.mem.span(path_ptr);
    const fd = std.os.open(path, flags, 0o666) catch return -1;
    return @intCast(fd);
}

pub export fn zig_close_file(fd: c_int) void {
    _ = std.os.close(@intCast(std.os.fd_t, fd));
}

pub export fn zig_read_file(fd: c_int, buffer: [*]u8, len: usize) isize {
    const handle = @intCast(std.os.fd_t, fd);
    const n = std.os.read(handle, buffer[0..len]) catch return -1;
    return @intCast(n);
}

pub export fn zig_write_file(fd: c_int, buffer: [*]const u8, len: usize) isize {
    const handle = @intCast(std.os.fd_t, fd);
    const n = std.os.write(handle, buffer[0..len]) catch return -1;
    return @intCast(n);
}

// Networking (resolve host and connect TCP)
pub export fn zig_tcp_connect(host_ptr: [*:0]const u8, port: u16) c_int {
    const allocator = std.heap.page_allocator;
    const host = std.mem.span(host_ptr);
    const addresses = std.net.getAddrInfo(allocator, host, port, null, null) catch return -1;
    defer addresses.deinit();
    const first = addresses.first orelse return -1;
    var stream = first.address.connect(allocator) catch return -1;
    return @intCast(stream.handle);
}

pub export fn zig_tcp_close(fd: c_int) void {
    _ = std.os.close(@intCast(std.os.fd_t, fd));
}

// Threading
pub export fn zig_spawn_thread(fn_ptr: *const fn() callconv(.C) void) c_int {
    const thread = std.Thread.spawn(.{}, struct {
        fn run(fn: *const fn() callconv(.C) void) void {
            @call(.auto, fn, .{});
        }
    }.run, .{fn_ptr}) catch return -1;
    return @intCast(thread.getHandle());
}

pub export fn zig_join_thread(tid: c_int) void {
    const id = @intCast(std.Thread.Id, tid);
    _ = std.Thread.join(id).catchVoid();
}
