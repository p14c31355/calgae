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

    pub fn run_quantization(self: *QuantWorker, model_path: [*:0]const u8, quant_bits: u8, output_path: [*:0]const u8) !void {
        const path = std.mem.span(model_path);
        const out_path = std.mem.span(output_path);
        std.log.info("Worker {d}: Starting quantization of {s} to {d} bits, output: {s}", .{ self.id, path, quant_bits, out_path });

        // Handle 4-bit and 8-bit quantization
        if (quant_bits != 4 and quant_bits != 8) {
            std.log.err("Only 4-bit and 8-bit quantization supported for now", .{});
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

        const floats: []const f32 = @as([*]const f32, @ptrCast(@alignCast(byte_buffer.ptr)))[0..num_floats];

        // Compute dynamic range
        var max_abs: f32 = 0.0;
        for (floats) |val| {
            const abs_val = if (val >= 0.0) val else -val;
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }
        const max_val: f32 = if (quant_bits == 4) 7.0 else 127.0;
        const scale = if (max_abs > 0.0) max_abs / max_val else 1.0;
        std.log.info("Dynamic range: max_abs={}, scale={}", .{ max_abs, scale });

        // Parallel quantize and write to output
        var out_file = try std.fs.cwd().createFile(out_path, .{});
        defer out_file.close();

        // Write scale as first f32
        const scale_bytes = std.mem.asBytes(&scale);
        _ = try out_file.write(scale_bytes);

        var file_mutex = std.Thread.Mutex{};
        // Pass scale and quant_bits to parallel loop
        try self.parallel_quant_loop(floats, scale, quant_bits, &out_file, &file_mutex);
    }

    fn parallel_quant_loop(self: *QuantWorker, floats: []const f32, scale: f32, quant_bits: u8, out_file: *std.fs.File, file_mutex: *std.Thread.Mutex) !void {
        const work_items = try std.Thread.getCpuCount();
        if (work_items == 0) return error.NoCoresAvailable;

        var threads = try self.allocator.alloc(std.Thread, work_items);
        defer self.allocator.free(threads);

        for (0..work_items) |i| {
            const start = i * (floats.len / work_items);
            const end = if (i == work_items - 1) floats.len else (i + 1) * (floats.len / work_items);

            threads[i] = try std.Thread.spawn(.{}, struct {
                pub fn task(chunk_start: usize, chunk_end: usize, flts: []const f32, scl: f32, q_bits: u8, output_file: *std.fs.File, mtx: *std.Thread.Mutex, worker_id: usize, alloc: std.mem.Allocator) void {
                    if (q_bits == 8) {
                        var quantized_bytes = alloc.alloc(u8, chunk_end - chunk_start) catch return;
                        defer alloc.free(quantized_bytes);

                        for (chunk_start..chunk_end) |k| {
                            const val = flts[k];
                            const dequant = @round(val / scl);
                            const clamped = std.math.clamp(@as(i32, @intFromFloat(dequant)), -128, 127);
                            quantized_bytes[k - chunk_start] = @as(u8, @bitCast(@as(i8, @intCast(clamped))));
                        }
                        mtx.lock();
                        defer mtx.unlock();
                        _ = output_file.writeAll(quantized_bytes) catch return;
                    } else if (q_bits == 4) {
                        const num_floats_in_chunk = chunk_end - chunk_start;
                        const num_bytes_for_chunk = (num_floats_in_chunk + 1) / 2;
                        var quantized_bytes = alloc.alloc(u8, num_bytes_for_chunk) catch return;
                        defer alloc.free(quantized_bytes);

                        var byte_idx: usize = 0;
                        for (chunk_start..chunk_end) |k| {
                            const val = flts[k];
                            const dequant = @round(val / scl);
                            const clamped = std.math.clamp(@as(i32, @intFromFloat(dequant)), -8, 7); // 4-bit signed range
                            const quantized_val_i8 = @as(i8, @intCast(clamped)); // Cast to i8 first
                            const quantized_val_u8 = @as(u8, @bitCast(quantized_val_i8)); // Then bitCast to u8

                            if (k % 2 == 0) { // Even index, lower nibble
                                quantized_bytes[byte_idx] = quantized_val_u8 & 0x0F;
                            } else { // Odd index, upper nibble
                                quantized_bytes[byte_idx] |= (quantized_val_u8 << 4) & 0xF0;
                                byte_idx += 1;
                            }
                        }
                        // Handle odd number of floats in chunk
                        if (num_floats_in_chunk % 2 != 0) {
                            byte_idx += 1;
                        }

                        mtx.lock();
                        defer mtx.unlock();
                        _ = output_file.writeAll(quantized_bytes[0..byte_idx]) catch return;
                    }
                    std.log.info("Worker {d} wrote chunk {d}-{d} (bits: {d})", .{ worker_id, chunk_start, chunk_end, q_bits });
                }
            }.task, .{ start, end, floats, scale, quant_bits, out_file, file_mutex, self.id, self.allocator });
        }

        // Join all threads
        for (threads) |thread| {
            thread.join();
        }
    }
};

pub export fn zig_quantize_model(model_path: [*:0]const u8, bits_int: i32, output_path: [*:0]const u8) i32 {
    const allocator = std.heap.page_allocator;
    if (bits_int != 4 and bits_int != 8) {
        std.log.err("Unsupported bits: {}", .{bits_int});
        return -1;
    }
    const quant_bits: u8 = @intCast(bits_int);

    var worker = QuantWorker.init(allocator, 0);
    worker.run_quantization(model_path, quant_bits, output_path) catch |err| {
        std.log.err("Quantization failed: {s}", .{@errorName(err)});
        return -1;
    };

    return 0;
}

pub export fn zig_quantize_buffer(input_ptr: [*]const f32, num: usize, bits: u8, output_ptr: [*]u8, scale_ptr: *f32) isize {
    if (bits != 4 and bits != 8) {
        std.log.err("Only 4-bit and 8-bit quantization supported", .{});
        return -1;
    }
    if (num == 0) {
        return 0;
    }

    const input = input_ptr[0..num];
    var max_abs: f32 = 0.0;
    for (input) |val| {
        const abs_val = if (val >= 0.0) val else -val;
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }
    const max_val: f32 = if (bits == 4) 7.0 else 127.0;
    const scale = if (max_abs > 0.0) max_abs / max_val else 1.0;
    scale_ptr.* = scale;

    if (bits == 8) {
        for (input, 0..) |val, i| {
            const dequant = @round(val / scale);
            const clamped = std.math.clamp(@as(i32, @intFromFloat(dequant)), -128, 127);
            output_ptr[i] = @as(u8, @bitCast(@as(i8, @intCast(clamped))));
        }
    } else if (bits == 4) {
        var byte_idx: usize = 0;
        for (input, 0..) |val, i| {
            const dequant = @round(val / scale);
            const clamped = std.math.clamp(@as(i32, @intFromFloat(dequant)), -8, 7); // 4-bit signed range
            const quantized_val_i8 = @as(i8, @intCast(clamped)); // Cast to i8 first
            const quantized_val_u8 = @as(u8, @bitCast(quantized_val_i8)); // Then bitCast to u8

            if (i % 2 == 0) { // Even index, lower nibble
                output_ptr[byte_idx] = quantized_val_u8 & 0x0F;
            } else { // Odd index, upper nibble
                output_ptr[byte_idx] |= (quantized_val_u8 << 4) & 0xF0;
                byte_idx += 1;
            }
        }
        // Handle odd number of floats
        if (num % 2 != 0) {
            byte_idx += 1;
        }
    }

    return 0;
}
