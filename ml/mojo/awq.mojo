# Mojo Implementation of AWQ and SmoothQuant - Syntax Fixed
from sys.info import *
from algorithm import parallelize
from memory import Pointer, UnsafePointer
from collections import List
alias type = Float32

fn abs_val(x: type) -> type:
    if x >= 0.0:
        return x
    else:
        return -x


# Per-channel max abs (parallelized)
@export
fn per_channel_max_abs_c(
    abs_output: UnsafePointer[type],
    batch_size: Int,
    seq_len: Int,
    hidden_size: Int,
    out_max: UnsafePointer[type]
) -> Int:
    # Zero init
    for i in range(hidden_size):
        out_max[i] = 0.0

    for b in range(batch_size):
        for s in range(seq_len):
            var offset = (b * seq_len + s) * hidden_size
            for c in range(hidden_size):
                var val = abs_output[offset + c]
                if val > out_max[c]:
                    out_max[c] = val
    return 0

# Top-k indices (min-heap like for top_k)
@export
fn top_k_indices_c(
    act_max: UnsafePointer[type],
    hidden_size: Int,
    top_k: Int,
    indices: UnsafePointer[Int32]
) -> Int:
    var effective_top_k: Int
    if top_k > hidden_size:
        effective_top_k = hidden_size
    else:
        effective_top_k = top_k

    var vals = List[type]()
    for i in range(effective_top_k):
        vals.append(0.0)
    var idxs = List[Int32]()
    for i in range(effective_top_k):
        idxs.append(0)

    var count: Int = 0
    for i in range(hidden_size):
        var val : type = act_max[i]
        if count < effective_top_k:
            vals[count] = val
            idxs[count] = i
            count += 1
        else:
            # Replace smallest
            var min_pos: Int = 0
            var min_val : type = vals[0]
            for j in range(1, effective_top_k):
                if vals[j] < min_val:
                    min_val = vals[j]
                    min_pos = j
            if val > min_val:
                vals[min_pos] = val
                idxs[min_pos] = i

    # Sort descending
    for i in range(effective_top_k):
        for j in range(i + 1, effective_top_k):
            if vals[i] < vals[j]:
                var tmp_val : type = vals[i]
                var tmp_idx : Int32 = idxs[i]
                vals[i] = vals[j]
                idxs[i] = idxs[j]
                vals[j] = tmp_val
                idxs[j] = tmp_idx

    for i in range(effective_top_k):
        indices[i] = idxs[i]

    return 0

# Compute scale
@export
fn compute_scale_c(
    act_max: UnsafePointer[type],
    hidden_size: Int,
    salient_indices: UnsafePointer[Int32],
    num_salient: Int,
    scale_out: UnsafePointer[type]
) -> Int:
    var overall_max: type = 0.0
    for i in range(hidden_size):
        overall_max = max(overall_max, act_max[i])

    var salient_max: type = 0.0
    for i in range(num_salient):
        var idx : Int = Int(salient_indices[i])
        if 0 <= idx < hidden_size:
            salient_max = max(salient_max, act_max[idx])

    var scale: type
    if salient_max > 0.0:
        scale = overall_max / (salient_max + Float32(1e-8))
    else:
        scale = 1.0
    scale_out[0] = scale
    return 0

fn swap(mut arr: List[type], a: Int, b: Int):
    var temp : type = arr[a]
    arr[a] = arr[b]
    arr[b] = temp

fn partition_desc(mut arr: List[type], low: Int, high: Int) -> Int:
    var pivot : type = arr[high]
    var i : Int = low - 1
    for j in range(low, high):
        if arr[j] >= pivot:
            i += 1
            swap(arr, i, j)
    swap(arr, i + 1, high)
    return i + 1

fn quicksort_desc(mut arr: List[type], low: Int, high: Int):
    if low < high:
        var pi : Int = partition_desc(arr, low, high)
        quicksort_desc(arr, low, pi - 1)
        quicksort_desc(arr, pi + 1, high)

# Compute SmoothQuant scales
@export
fn compute_smoothquant_scales_c(
    act_max: UnsafePointer[type],
    hidden_size: Int,
    sparsity: type,
    bits: Int,
    scales_out: UnsafePointer[type]
) -> Int:
    var act_list = List[type]()
    for i in range(hidden_size):
        act_list.append(act_max[i])

    quicksort_desc(act_list, 0, hidden_size - 1)

    var num_outliers : Int = Int(sparsity * Float32(hidden_size))
    var beta : type = 0.85
    var qmax : type = Float32((1 << (bits - 1)) - 1)

    for i in range(hidden_size):
        scales_out[i] = 1.0

    if num_outliers > 0:
        var outlier_max : type = act_list[0]
        var scale_factor : type = (outlier_max / beta) / qmax
        for i in range(num_outliers):
            scales_out[i] = scale_factor

    return 0

# Apply AWQ to weight (inout)
@export("apply_awq_quantize_c")
fn apply_awq_quantize(
    weight: UnsafePointer[type],  # inout [out_dim * in_dim]
    out_dim: Int,
    in_dim: Int,
    scale: type,
    salient_indices: UnsafePointer[Int32],
    num_salient: Int,
    bits: Int = 4
) -> Int:
    # Scale salient channels
    var salient_set = List[Int]()

    for i in range(num_salient):
        var ch : Int = Int(salient_indices[i])
        if 0 <= ch < out_dim:
            salient_set.append(ch)

    # Apply scale to salient
    for ch_idx in range(len(salient_set)):
        var s_ch : Int = salient_set[ch_idx]
        var offset : Int = s_ch * in_dim
        for j in range(in_dim):
            var idx : Int = offset + j
            weight[idx] = weight[idx] * scale

    # Quantize non-salient

    var mask = List[Bool]()
    for i in range(out_dim):
        mask.append(False)
    for ch_s in salient_set:
        mask[ch_s] = True

    var qmax_int : Int = 7
    var qmin_int : Int = -8

    for ch in range(out_dim):
        if not mask[ch]:
            var offset : Int = ch * in_dim
            var ch_max: type = 0.0
            for j in range(in_dim):
                var idx : Int = offset + j
                ch_max = max(ch_max, abs_val(weight[idx]))

            var ch_scale : type
            if ch_max > 0.0:
                ch_scale = ch_max / Float32(qmax_int)
            else:
                ch_scale = 1.0
            for j in range(in_dim):
                var idx : Int = offset + j
                var val : type = weight[idx]
                var sign_adjust: type
                if val >= 0.0:
                    sign_adjust = Float32(0.5)
                else:
                    sign_adjust = Float32(-0.5)
                var rounded : type = val / ch_scale + sign_adjust
                var q_val : Int = Int(rounded)
                var clamped : Int = max(min(q_val, qmax_int), qmin_int)
                weight[idx] = Float32(clamped) * ch_scale

    return 0

# Apply SmoothQuant to weight
@export("apply_smoothquant_quantize_c")
fn apply_smoothquant_quantize(
    weight: UnsafePointer[type],
    out_dim: Int,
    in_dim: Int,
    act_scales: UnsafePointer[type],
    bits: Int = 8,
    group_size: Int = 128
) -> Int:
    # Compensate act scales
    for ch in range(out_dim):
        var a_scale : type = act_scales[ch]
        if a_scale < Float32(1e-8):
            a_scale = Float32(1e-8)
        var offset : Int = ch * in_dim
        for j in range(in_dim):
            var idx : Int = offset + j
            weight[idx] = weight[idx] / a_scale

    # Group quant
    var qmax_int : Int = 127
    var qmin_int : Int = -128
    var qmax : type = Float32(qmax_int)
    var qmin : type = Float32(qmin_int)
    for g_start in range(0, out_dim, group_size):
        var g_end : Int = min(g_start + group_size, out_dim)
        var g_max: type = 0.0
        # First pass to find group max
        for ch in range(g_start, g_end):
            var offset : Int = ch * in_dim
            for j in range(in_dim):
                var idx : Int = offset + j
                g_max = max(g_max, abs_val(weight[idx]))

        var g_scale : type
        if g_max > 0.0:
            g_scale = g_max / qmax
        else:
            g_scale = 1.0
        # Second pass to quantize
        for ch in range(g_start, g_end):
            var offset : Int = ch * in_dim
            for j in range(in_dim):
                var idx : Int = offset + j
                var temp_val : type = weight[idx] / g_scale
                var sign_adjust: type
                if temp_val >= 0.0:
                    sign_adjust = 0.5
                else:
                    sign_adjust = -0.5
                var q_val = Float32(Int(temp_val + sign_adjust))
                var clamped : type = max(min(q_val, qmax), qmin)
                weight[idx] = clamped * g_scale

    return 0

# Example usage for FFI layer quantize
@export
fn quantize_layer_awq(
    acts_abs: UnsafePointer[type],
    b: Int,
    s: Int,
    h: Int,
    weight: UnsafePointer[type],
    in_d: Int,
    out_d: Int,
    top_k_p: type
) -> Int:
    # Local buffers
    var act_max = List[type]()
    for i in range(h):
        act_max.append(0.0)
    _ = per_channel_max_abs_c(acts_abs, b, s, h, act_max.unsafe_ptr())

    var num_sal : Int = max(1, Int(top_k_p * Float32(h)))
    var sal_idx = List[Int32]()
    for i in range(num_sal):
        sal_idx.append(0)
    _ = top_k_indices_c(act_max.unsafe_ptr(), h, num_sal, sal_idx.unsafe_ptr())

    var sc = List[type]()
    for i in range(1):
        sc.append(0.0)
    _ = compute_scale_c(act_max.unsafe_ptr(), h, sal_idx.unsafe_ptr(), num_sal, sc.unsafe_ptr())
    var the_scale : type = sc[0]

    _ = apply_awq_quantize(weight, out_d, in_d, the_scale, sal_idx.unsafe_ptr(), num_sal)

    return 0

# Similar for SmoothQuant
@export
fn quantize_layer_smoothquant(
    acts_abs: UnsafePointer[type],
    b: Int,
    s: Int,
    h: Int,
    weight: UnsafePointer[type],
    in_d: Int,
    out_d: Int,
    spar: type,
    bt: Int
) -> Int:
    var act_max = List[type]()
    for i in range(h):
        act_max.append(0.0)
    _ = per_channel_max_abs_c(acts_abs, b, s, h, act_max.unsafe_ptr())

    var scales = List[type]()
    for i in range(h):
        scales.append(0.0)
    _ = compute_smoothquant_scales_c(act_max.unsafe_ptr(), h, spar, bt, scales.unsafe_ptr())

    _ = apply_smoothquant_quantize(weight, out_d, in_d, scales.unsafe_ptr(), bt)

    return 0
