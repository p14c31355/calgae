const std = @import("std");
const libc = @cImport({
    @cInclude("stdint.h");
});
const pthread = @cImport({
    @cInclude("pthread.h");
});

const windows = if (builtin.os.tag == .windows) @cImport({ @cInclude("windows.h"); }) else null;

const builtin = @import("builtin");
const kernel = @import("zig_kernel.zig");

// OS abstraction layer for FFI to Rust
// Provides file IO, networking, and threading primitives via C ABI

pub export fn zig_open_file(path_ptr: [*:0]const u8, flags: u32) i32 {
    if (builtin.os.tag == .linux or builtin.os.tag == .macos or builtin.os.tag == .freestanding) {
        const mode: std.fs.File.Mode = 0o666;
        const open_flags: std.posix.O = @as(std.posix.O, @bitCast(flags));
        const fd = std.posix.openZ(path_ptr, open_flags, mode) catch return -1;
        return fd;
    } else {
        @compileError("File I/O only supported on Unix-like systems for now.");
    }
}

pub export fn zig_close_file(fd: i32) void {
    if (builtin.os.tag == .linux or builtin.os.tag == .macos or builtin.os.tag == .freestanding) {
        _ = std.posix.close(fd);
    } else {
        @compileError("File I/O only supported on Unix-like systems for now.");
    }
}

pub export fn zig_read_file(fd: i32, buffer: [*]u8, len: usize) isize {
    if (builtin.os.tag == .linux or builtin.os.tag == .macos or builtin.os.tag == .freestanding) {
        const handle = @as(std.posix.fd_t, fd);
        const n = std.posix.read(handle, buffer[0..len]) catch return -1;
        return @intCast(n);
    } else {
        @compileError("File I/O only supported on Unix-like systems for now.");
    }
}

pub export fn zig_write_file(fd: i32, buffer: [*]const u8, len: usize) isize {
    if (builtin.os.tag == .linux or builtin.os.tag == .macos or builtin.os.tag == .freestanding) {
        const handle = @as(std.posix.fd_t, fd);
        const n = std.posix.write(handle, buffer[0..len]) catch return -1;
        return @intCast(n);
    } else {
        @compileError("File I/O only supported on Unix-like systems for now.");
    }
}

// Networking (resolve host and connect TCP)
pub export fn zig_tcp_connect(host_ptr: [*:0]const u8, port: u16) i32 {
    const allocator = std.heap.page_allocator;
    const stream = std.net.tcpConnectToHost(allocator, std.mem.span(host_ptr), port) catch return -1;
    const fd = stream.handle;
    // stream is dropped here, handle is returned
    return @intCast(fd);
}


pub export fn zig_spawn_thread(entry_fn: *const fn(?*anyopaque) callconv(.c) ?*anyopaque, arg: ?*anyopaque) ?*anyopaque {
    if (builtin.os.tag == .linux or builtin.os.tag == .macos or builtin.os.tag == .freestanding) {
        const thread_handle_ptr = std.heap.page_allocator.alloc(pthread.pthread_t, 1) catch {
            std.log.err("Failed to allocate memory for pthread_t", .{});
            return null;
        };
        const rc = pthread.pthread_create(thread_handle_ptr, null, entry_fn, arg);
        if (rc != 0) {
            std.heap.page_allocator.free(thread_handle_ptr);
            std.log.err("Failed to create pthread: {d}", .{rc});
            return null;
        }
        return thread_handle_ptr;
    } else if (builtin.os.tag == .windows) {
        // TODO: Implement Windows threading using CreateThread
        @compileError("Windows threading not yet implemented.");
    } else {
        @compileError("Threading only supported on Unix-like systems for now.");
    }
}

pub export fn zig_join_thread(thread_handle: ?*anyopaque) i32 {
    if (builtin.os.tag == .linux or builtin.os.tag == .macos or builtin.os.tag == .freestanding) {
        if (thread_handle == null) {
            std.log.err("Attempted to join a null thread handle", .{});
            return -1;
        }
        const thread_handle_ptr: *pthread.pthread_t = @ptrCast(thread_handle);
        const rc = pthread.pthread_join(thread_handle_ptr.*, null);
        std.heap.page_allocator.free(thread_handle_ptr);
        if (rc != 0) {
            std.log.err("Failed to join pthread: {d}", .{rc});
            return -1;
        }
        return 0;
    } else if (builtin.os.tag == .windows) {
        // TODO: Implement Windows threading using WaitForSingleObject
        @compileError("Windows threading not yet implemented.");
    } else {
        @compileError("Threading only supported on Unix-like systems for now.");
    }
}

pub export fn runtime_matmul(m: usize, n: usize, p: usize, a: [*]const f32, b: [*]const f32, c_out: [*]f32) void {
    kernel.matrix_mult(m, n, p, a, b, c_out);
}
