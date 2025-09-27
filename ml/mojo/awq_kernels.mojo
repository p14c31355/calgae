# Mojo kernel for AWQ per-channel max abs
# Export as C library for Python interop

@export("per_channel_max_abs_c")
fn per_channel_max_abs_c(
    abs_output: UnsafePointer[Float32],
    batch_size: Int,
    seq_len: Int,
    hidden_size: Int,
    out_max: UnsafePointer[Float32]
) -> Int:
    # Initialize to 0.0
    for i in range(hidden_size):
        out_max.store(i, 0.0)

    for c in range(hidden_size):
        var c_max : Float32 = 0.0
        for b in range(batch_size):
            for s in range(seq_len):
                var idx = (b * seq_len + s) * hidden_size + c
                var val : Float32 = abs_output.load(idx)
                if val > c_max:
                    c_max = val
        out_max.store(c, c_max)

    return 0
