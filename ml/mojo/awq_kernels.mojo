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

# Top-k indices by descending order for salient channels
@export("top_k_indices_c")
fn top_k_indices_c(
    act_max: UnsafePointer[Float32],
    hidden_size: Int,
    top_k: Int,
    indices: UnsafePointer[Int32]
) -> Int:
    # Collects up to top-k candidates by scanning and maintaining a partial list, then bubble sorts for descending order. O(n*k) time.
    # For small top_k relative to hidden_size, this is acceptable. For larger top_k, a min-heap would be more efficient with O(n log k).
    var temp_max: List[Float32] = List[Float32]()
    var temp_idx: List[Int32] = List[Int32]()
    temp_max.reserve(top_k)
    temp_idx.reserve(top_k)

    for i in range(hidden_size):
        var val: Float32 = act_max.load(i)
        var idx: Int32 = i
        var inserted: Bool = False
        for j in range(len(temp_max)):
            if val > temp_max[j]:
                temp_max.insert(j, val)
                temp_idx.insert(j, idx)
                if len(temp_max) > top_k:
                    _ = temp_max.pop()
                    _ = temp_idx.pop()
                inserted = True
                break
        if not inserted and len(temp_max) < top_k:
            temp_max.append(val)
            temp_idx.append(idx)

    # Sort temp by descending (simple bubble for small top_k)
    for i in range(len(temp_max)):
        for j in range(i + 1, len(temp_max)):
            if temp_max[i] < temp_max[j]:
                var swap_max = temp_max[i]
                var swap_idx = temp_idx[i]
                temp_max[i] = temp_max[j]
                temp_idx[i] = temp_idx[j]
                temp_max[j] = swap_max
                temp_idx[j] = swap_idx

    for i in range(top_k):
        indices.store(i, temp_idx[i])

    return 0

# Compute scaling factor: overall_max / salient_max
@export("compute_scale_c")
fn compute_scale_c(
    act_max: UnsafePointer[Float32],
    hidden_size: Int,
    salient_indices: UnsafePointer[Int32],
    num_salient: Int,
    scale_out: UnsafePointer[Float32]
) -> Int:
    var overall_max: Float32 = 0.0
    for i in range(hidden_size):
        var val: Float32 = act_max.load(i)
        if val > overall_max:
            overall_max = val

    var salient_max: Float32 = 0.0
    for i in range(num_salient):
        var idx: Int = Int(salient_indices.load(i))
        var val: Float32 = act_max.load(idx)
        if val > salient_max:
            salient_max = val

    var scale: Float32 = overall_max / (salient_max + 1e-8)
    scale_out.store(0, scale)

    return 0

# Helper to swap elements in an UnsafePointer
fn swap(data: UnsafePointer[Float32], i: Int, j: Int):
    var temp : Float32 = data.load(i)
    data.store(i, data.load(j))
    data.store(j, temp)

# QuickSort implementation for UnsafePointer[Float32] (descending order)
fn quicksort_descending(data: UnsafePointer[Float32], low: Int, high: Int):
    if low < high:
        var pivot_val = data.load(high)
        var i = low - 1
        for j in range(low, high):
            if data.load(j) >= pivot_val: # Descending order
                i += 1
                swap(data, i, j)
        swap(data, i + 1, high)
        var pivot_idx = i + 1

        quicksort_descending(data, low, pivot_idx - 1)
        quicksort_descending(data, pivot_idx + 1, high)

# Compute SmoothQuant per-channel scales
@export("compute_smoothquant_scales_c")
fn compute_smoothquant_scales_c(
    act_max: UnsafePointer[Float32],
    hidden_size: Int,
    sparsity: Float32,
    bits: Int,
    scales_out: UnsafePointer[Float32]
) -> Int:
    # Collect act_max into a temporary buffer for sorting
    var temp_act_max_buffer = UnsafePointer[Float32].alloc(hidden_size)
    for i in range(hidden_size):
        temp_act_max_buffer.store(i, act_max.load(i))

    quicksort_descending(temp_act_max_buffer, 0, hidden_size - 1)

    var num_outliers = Int(sparsity * Float32(hidden_size))
    var beta: Float32 = 0.85
    var qmax: Float32 = (1 << (bits - 1)) - 1  # Assuming signed int

    # Set all scales to 1.0
    for i in range(hidden_size):
        scales_out.store(i, 1.0)

    if num_outliers > 0:
        var outlier_max: Float32 = temp_act_max_buffer.load(0)  # Largest after descending sort
        var scale_factor: Float32 = (outlier_max / beta) / qmax
        for i in range(num_outliers):
            scales_out.store(i, scale_factor)
    
    temp_act_max_buffer.free() # Free the temporary buffer

    return 0
