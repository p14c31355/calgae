const std = @import("std");

// Lightweight async quantizer worker using Zig's built-in async
// For CPU-parallel model quantization loops
// Can be extended to call Mojo for HPC parts

const QuantWorker = struct {
    allocator: std.mem.Allocator,
    id: usize,

    pub fn init(alloc: std.mem.Allocator, id: usize) QuantWorker {
        return .{ .allocator = alloc, .id = id };
    }

    pub fn run_quantization(self: *QuantWorker, model_path: [*:0]const u8, quant_bits: u8) !void {
        const path = std.mem.span(model_path);
        std.log.info("Worker {d}: Starting quantization of {s} to {d} bits", .{ self.id, path, quant_bits });

        // For simplicity, assume quant_bits = 8 (int8), flat f32 model
        if (quant_bits != 8) {
            std.log.err("Only 8-bit quantization supported for now", .{});
            return error.UnsupportedQuantization;
        }

        // Read the entire model as f32
        const file = std.fs.cwd().openFile(path, .{}) catch |err| {
            std.log.err("Cannot open {s}: {s}", .{ path, @errorName(err) });
            return err;
        };
        defer file.close();
        const stat = file.stat() catch |err| {
            std.log.err("Cannot stat {s}: {s}", .{ path, @errorName(err) });
            return err;
        };
        const num_bytes: usize = @intCast(stat.size);

        // Read bytes and reinterpret as f32 slice (assume little-endian alignment)
        const byte_buffer = try self.allocator.alloc(u8, num_bytes);
        defer self.allocator.free(byte_buffer);

        const bytes_read = file.readAll(byte_buffer) catch |err| {
            std.log.err("Read failed: {}", .{err});
            return err;
        };
        if (bytes_read != num_bytes) {
            std.log.err("Incomplete read: expected {d}, got {d}", .{ num_bytes, bytes_read });
            return error.IncompleteRead;
        }

        const num_floats = num_bytes / @sizeOf(f32);
        if (num_bytes % @sizeOf(f32) != 0) {
            std.log.err("File size not multiple of f32 size", .{});
            return error.InvalidFileSize;
        }
        var floats = try self.allocator.alloc(f32, num_floats);
        defer self.allocator.free(floats);

        var j: usize = 0;
        while (j < num_floats) : (j += 1) {
            const bytes_start = j * 4;
            const bytes = byte_buffer[bytes_start..@min(bytes_start + 4, byte_buffer.len)];
            if (bytes.len < 4) {
                std.log.err("Incomplete f32 at position {}", .{j});
                return error.IncompleteFloatRead;
            }
            const int_val = @as(u32, bytes[0]) | (@as(u32, bytes[1]) << 8) | (@as(u32, bytes[2]) << 16) | (@as(u32, bytes[3]) << 24);
            floats[j] = @as(f32, @bitCast(int_val));
        }

        // Parallel quantize and write to quantized_model.bin
        const quantized_path = "models/quantized_model.bin";
        var out_file = try std.fs.cwd().createFile(quantized_path, .{});
        defer out_file.close();

        var file_mutex = std.Thread.Mutex{};
        try self.parallel_quant_loop(floats, &out_file, &file_mutex);
    }

    fn parallel_quant_loop(self: *QuantWorker, floats: []f32, out_file: *std.fs.File, file_mutex: *std.Thread.Mutex) !void {
        const work_items = try std.Thread.getCpuCount();
        if (work_items == 0) return error.NoCoresAvailable;

        const chunk_size = floats.len / work_items;
        var threads = try self.allocator.alloc(std.Thread, work_items);
        defer self.allocator.free(threads);

        for (0..work_items) |i| {
            const start = i * chunk_size;
            const end = if (i == work_items - 1) floats.len else (i + 1) * chunk_size;
            const chunk = floats[start..end];
            const idx = i;

            threads[i] = try std.Thread.spawn(.{}, struct {
                pub fn task(ch: []f32, id: usize, output_file: *std.fs.File, mutex: *std.Thread.Mutex) !void {
                    // Helper function to encapsulate the synchronized write
                    const write_buf = struct {
                        fn write(buffer: []const u8, file_writer: *std.fs.File, mtx: *std.Thread.Mutex) !void {
                            mtx.lock();
                            defer mtx.unlock();
                            try file_writer.writeAll(buffer);
                        }
                    }.write;

                    var local_buf: [1024]u8 = undefined;
                    var buf_len: usize = 0;

                    for (ch) |val| {
                        const scaled = @as(i32, @intFromFloat(@round(val * 127.0))); // Simple linear quant to i8
                        const clamped = std.math.clamp(scaled, -128, 127);
                        local_buf[buf_len] = @as(u8, @as(i8, @intCast(clamped)));
                        buf_len += 1;

                        if (buf_len == 1024) {
                            try write_buf(local_buf[0..buf_len], output_file, mutex);
                            buf_len = 0;
                        }
                    }
                    // Final flush
                    if (buf_len > 0) {
                        try write_buf(local_buf[0..buf_len], output_file, mutex);
                    }
                    std.log.info("Task {d}: Quantized {d} values", .{ id, ch.len });
                }
            }.task, .{ chunk, idx, out_file, file_mutex });
        }

        // Join all
        for (threads) |thread| {
            try thread.join();
        }
    }
};

pub export fn zig_quantize_model(model_path_ptr: [*:0]const u8, bits: u8) isize {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var worker = QuantWorker.init(allocator, 0);
    worker.run_quantization(model_path_ptr, bits) catch |err| {
        std.log.err("Quantization failed: {}", .{err});
        return -1; // Failure
    };
    std.log.info("Quantization successfully completed", .{});
    return 0;
}
