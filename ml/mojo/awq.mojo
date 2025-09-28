# Mojo Implementation of AWQ and SmoothQuant - Syntax Fixed
from sys.info import *
from algorithm import parallelize
from memory.unsafe import malloc, free, memset
from random import rand

alias type = Float32
alias simd_width = simdwidthof[type]()

const eps: Scalar[type] = 1e-8f32

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

    @parameter
    fn compute_channel_max(c: Int):
        var c_max: Scalar[type] = 0.0
        for b in range(batch_size):
            for s in range(seq_len):
                let idx: Int = b * seq_len * hidden_size + s * hidden_size + c
                let val: Scalar[type] = abs_output[idx]
                c_max = max(c_max, val)
        out_max[c] = c_max

    parallelize[compute_channel_max, 256](hidden_size)  # Parallel over channels
    return 0

# Top-k indices (min-heap like for top_k)
@export
fn top_k_indices_c(
    act_max: UnsafePointer[type],
    hidden_size: Int,
    top_k: Int,
    indices: UnsafePointer[Int32]
) -> Int:
    if top_k > hidden_size:
        top_k = hidden_size

    var vals: UnsafePointer[type] = malloc[type](top_k)
    var idxs: UnsafePointer[Int32] = malloc[Int32](top_k)
    for i in range(top_k):
        vals[i] = 0.0
        idxs[i] = 0i32

    var count: Int = 0
    for i in range(hidden_size):
        let val = act_max[i]
        if count < top_k:
            vals[count] = val
            idxs[count] = i
            count += 1
            # Simple find min position for next
        else if val > vals[0]:
            # Replace smallest
            var min_pos: Int = 0
            var min_val: Scalar[type] = vals[0]
            for j in range(1, top_k):
                if vals[j] < min_val:
                    min_val = vals[j]
                    min_pos = j
            if val > min_val:
                vals[min_pos] = val
                idxs[min_pos] = i

    # Sort descending
    for i in range(top_k):
        for j in range(i + 1, top_k):
            if vals[i] < vals[j]:
                let tmp_val = vals[i]
                let tmp_idx = idxs[i]
                vals[i] = vals[j]
                idxs[i] = idxs[j]
                vals[j] = tmp_val
                idxs[j] = tmp_idx

    for i in range(top_k):
        indices[i] = idxs[i]

    free[UnsafePointer[type]](vals, top_k * size_of[type]())
    free[UnsafePointer[Int32]](idxs, top_k * size_of[Int32]())
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
    var overall_max: Scalar[type] = 0.0
    for i in range(hidden_size):
        overall_max = max(overall_max, act_max[i])

    var salient_max: Scalar[type] = 0.0
    for i in range(num_salient):
        let idx = Int(salient_indices[i])
        if 0 <= idx < hidden_size:
            salient_max = max(salient_max, act_max[idx])

    let scale = if salient_max > 0.0 { overall_max / (salient_max + eps) } else { 1.0 }
    scale_out[0] = scale
    return 0

# Compute SmoothQuant scales
@export
fn compute_smoothquant_scales_c(
    act_max: UnsafePointer[type],
    hidden_size: Int,
    sparsity: Scalar[type],
    bits: Int,
    scales_out: UnsafePointer[type]
) -> Int:
    var act_list: UnsafePointer[type] = malloc[type](hidden_size)
    for i in range(hidden_size):
        act_list[i] = act_max[i]

    # Bubble sort descending
    for i in range(hidden_size):
        for j in range(i + 1, hidden_size):
            if act_list[i] < act_list[j]:
                let tmp = act_list[i]
                act_list[i] = act_list[j]
                act_list[j] = tmp

    let num_outliers = Int(sparsity * Scalar[type](hidden_size))
    let beta: Scalar[type] = 0.85
    let qmax: Scalar[type] = if bits == 8 { 127.0 } else { 7.0 }

    for i in range(hidden_size):
        scales_out[i] = 1.0

    if num_outliers > 0:
        let outlier_max = act_list[0]
        let scale_factor = (outlier_max / beta) / qmax
        for i in range(num_outliers):
            scales_out[i] = scale_factor

    free[UnsafePointer[type]](act_list, hidden_size * size_of[type]())
    return 0

# Apply AWQ to weight (inout)
@export
fn apply_awq_quantize(
    weight: UnsafePointer[type],  # inout [out_dim * in_dim]
    out_dim: Int,
    in_dim: Int,
    scale: Scalar[type],
    salient_indices: UnsafePointer[Int32],
    num_salient: Int,
    bits: Int = 4
) -> Int:
    # Scale salient channels
    var salient_set = List[Int]()

    for i in range(num_salient):
        let ch = Int(salient_indices[i])
        if 0 <= ch < out_dim:
            salient_set.append(ch)

    # Apply scale to salient
    for ch in range(len(salient_set)):
        let s_ch = salient_set[ch]
        let offset = s_ch * in_dim
        for j in range(in_dim):
            let idx = offset + j
            weight[idx] = weight[idx] * scale

    # Quantize non-salient
    let qmax = 7.0  # INT4 signed max
    let qmin = -8.0
    for ch in range(out_dim):
        let is_salient = False
        for s in range(len(salient_set)):
            if salient_set[s] == ch:
                is_salient = True
                break
        if not is_salient:
            let offset = ch * in_dim
            var ch_max: Scalar[type] = 0.0
            for j in range(in_dim):
                let idx = offset + j
                ch_max = max(ch_max, abs(weight[idx]))

            let ch_scale = if ch_max > 0.0 { ch_max / qmax } else { 1.0 }
            for j in range(in_dim):
                let idx = offset + j
                let val = weight[idx]
                let q_val = round(val / ch_scale)
                let clamped = max(min(q_val, qmax), qmin)
                weight[idx] = Scalar[type](clamped) * ch_scale

    return 0

# Apply SmoothQuant to weight
@export
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
        let a_scale = max(act_scales[ch], eps)
        let offset = ch * in_dim
        for j in range(in_dim):
            let idx = offset + j
            weight[idx] = weight[idx] / a_scale

    # Group quant
    let qmax = 127.0
    let qmin = -128.0
    for g_start in range(0, out_dim, group_size):
        let g_end = min(g_start + group_size, out_dim)
        var g_max: Scalar[type] = 0.0
        for ch in range(g_start, g_end):
            let offset = ch * in_dim
            for j in range(in_dim):
                let idx = offset + j
                g_max = max(g_max, abs(weight[idx]))

        let g_scale = if g_max > 0.0 { g_max / qmax } else { 1.0 }
        for ch in range(g_start, g_end):
            let offset = ch * in_dim
            for j in range(in_dim):
                let idx = offset + j
                let val = weight[idx]
                let q_val = round(val / g_scale)
                let clamped = max(min(q_val, qmax), qmin)
                weight[idx] = Scalar[type](clamped) * g_scale

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
    top_k_p: Scalar[type]
) -> Int:
    # Local buffers
    var act_max = malloc[type](h)
    _ = per_channel_max_abs_c(acts_abs, b, s, h, act_max)

    let num_sal = max(1, Int(top_k_p * Scalar[type](h)))
    var sal_idx = malloc[Int32](num_sal)
    _ = top_k_indices_c(act_max, h, num_sal, sal_idx)

    var sc = malloc[type](1)
    _ = compute_scale_c(act_max, h, sal_idx, num_sal, sc)
    let the_scale = sc[0]

    _ = apply_awq_quantize(weight, out_d, in_d, the_scale, sal_idx, num_sal)

    free(act_max, h * size_of[type]())
    free(sal_idx, num_sal * size_of[Int32]())
    free(sc, size_of[type]())
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
    spar: Scalar[type],
    bt: Int
) -> Int:
    var act_max = malloc[type](h)
    _ = per_channel_max_abs_c(acts_abs, b, s, h, act_max)

    var scales = malloc[type](h)
    _ = compute_smoothquant_scales_c(act_max, h, spar, bt, scales)

    _ = apply_smoothquant_quantize(weight, out_d, in_d, scales, bt)

    free(act_max, h * size_of[type]())
    free(scales, h * size_of[type]())
    return 0
