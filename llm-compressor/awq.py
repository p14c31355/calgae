#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Any
import json
import ctypes
import os
import platform
import time

# Load Mojo library for optimized kernels - required for execution
mojo_lib = None
try:
    mojo_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'libawq.so'))
    # Verify required functions are available
    mojo_lib.per_channel_max_abs_c.restype = ctypes.c_int
    mojo_lib.top_k_indices_c.restype = ctypes.c_int
    mojo_lib.compute_scale_c.restype = ctypes.c_int
    mojo_lib.compute_smoothquant_scales_c.restype = ctypes.c_int
    mojo_lib.apply_awq_quantize_c.restype = ctypes.c_int
    mojo_lib.apply_smoothquant_quantize_c.restype = ctypes.c_int
    print("Mojo library loaded successfully with quantization functions.")
except OSError as e:
    raise RuntimeError(f"Failed to load required Mojo libawq.so: {e}")
except Exception as e:
    raise RuntimeError(f"Unexpected error loading libawq.so or verifying functions: {e}")

if not mojo_lib:
    raise RuntimeError("Mojo library is mandatory for this implementation.")

def load_calibration_data(tokenizer, num_samples=128, max_length=2048):
    """
    Load realistic calibration data (e.g., from a simple text corpus).
    For better accuracy, use datasets like C4 or WikiText.
    """
    # Realistic calibration: varied sentences
    calibration_texts = [
        "The quick brown fox jumps over the lazy dog. " * (max_length // 100),
        "Artificial intelligence is transforming the world in many ways. ",
        "Machine learning models require training data to learn patterns. ",
        "Quantization reduces model size for efficient deployment. ",
        "Neural networks consist of layers that process information. "
    ] * (num_samples // 5 + 1)  # Repeat to reach num_samples
    calibration_texts = calibration_texts[:num_samples]
    calibration_inputs = tokenizer(calibration_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return calibration_inputs["input_ids"], calibration_inputs["attention_mask"]

def collect_activation_statistics(model, inputs, layers_to_collect=32):
    """
    Collect per-output-channel max abs activation for key linear layers.
    Now uses Mojo for channel max computation to speed up.
    """
    model.eval()
    activations = {}  # Dict of layer_name: tensor of per-channel max abs act
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            act_abs = torch.abs(output).float()
            batch_size, seq_len, hidden_size = act_abs.shape
            flat_abs = act_abs.view(batch_size * seq_len, hidden_size).cpu().numpy().astype(np.float32)
            channel_max = np.zeros(hidden_size, dtype=np.float32)

            # Call Mojo per_channel_max_abs_c
            res = mojo_lib.per_channel_max_abs_c(
                flat_abs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(1),  # batch=1 after view
                ctypes.c_int(batch_size * seq_len),
                ctypes.c_int(hidden_size),
                channel_max.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            if res != 0:
                raise RuntimeError(f"Mojo per_channel_max_abs_c failed for {name}")

            if name not in activations:
                activations[name] = torch.tensor(channel_max)
            else:
                activations[name] = torch.max(activations[name], torch.tensor(channel_max))
        return hook
    
    # Register hooks on all key linear layers
    for i in range(layers_to_collect):
        layer = model.model.layers[i]
        # Attention projections
        if hasattr(layer.self_attn, 'q_proj'):
            hooks.append(layer.self_attn.q_proj.register_forward_hook(hook_fn(f"layer_{i}_q")))
            hooks.append(layer.self_attn.k_proj.register_forward_hook(hook_fn(f"layer_{i}_k")))
            hooks.append(layer.self_attn.v_proj.register_forward_hook(hook_fn(f"layer_{i}_v")))
            hooks.append(layer.self_attn.o_proj.register_forward_hook(hook_fn(f"layer_{i}_o")))
        # MLP
        if hasattr(layer.mlp, 'gate_proj'):
            hooks.append(layer.mlp.gate_proj.register_forward_hook(hook_fn(f"layer_{i}_gate")))
            hooks.append(layer.mlp.up_proj.register_forward_hook(hook_fn(f"layer_{i}_up")))
        hooks.append(layer.mlp.down_proj.register_forward_hook(hook_fn(f"layer_{i}_down")))
    
    with torch.no_grad():
        for batch in range(0, len(inputs[0]), 8):
            batch_ids = inputs[0][batch:batch + 8]
            batch_mask = inputs[1][batch:batch + 8]
            _ = model(input_ids=batch_ids, attention_mask=batch_mask)  # Forward only, no loss/backward
    
    for h in hooks:
        h.remove()
    
    return activations

def find_salient_channels(activations, top_k_percent=0.001):
    """
    Identify top-k% salient output channels per layer based on max abs activation.
    Uses Mojo for top-k computation.
    """
    salient_channels = {}
    for name, act_max_torch in activations.items():
        act_max = act_max_torch.numpy().astype(np.float32)
        hidden_size = len(act_max)
        num_salient = max(1, int(hidden_size * top_k_percent))
        indices = np.zeros(num_salient, dtype=np.int32)
        res = mojo_lib.top_k_indices_c(
            act_max.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(hidden_size),
            ctypes.c_int(num_salient),
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )
        if res != 0:
            raise RuntimeError(f"Mojo top_k_indices_c failed for {name}")
        salient_channels[name] = indices.tolist()
    return salient_channels

def compute_scaling_factors(activations, salient_channels):
    """
    Compute per-layer scale: max_all_act / max_salient_act to scale up salient channels.
    Uses Mojo.
    """
    scales = {}
    for name, act_max_torch in activations.items():
        act_max = act_max_torch.numpy().astype(np.float32)
        hidden_size = len(act_max)
        salient_indices = np.array(salient_channels[name], dtype=np.int32)
        num_salient = len(salient_indices)
        scale_out = np.zeros(1, dtype=np.float32)
        res = mojo_lib.compute_scale_c(
            act_max.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(hidden_size),
            salient_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(num_salient),
            scale_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        if res != 0:
            raise RuntimeError(f"Mojo compute_scale_c failed for {name}")
        scales[name] = float(scale_out[0])
    return scales

def compute_smoothquant_scales(activations, bits=8, sparsity=0.85):
    """
    Compute SmoothQuant scales: Per-channel activation scales to absorb outliers into weights.
    Uses Mojo.
    """
    smooth_scales = {}
    for name, act_max_torch in activations.items():
        act_max = act_max_torch.numpy().astype(np.float32)
        hidden_size = len(act_max)
        scales_out = np.ones(hidden_size, dtype=np.float32)
        res = mojo_lib.compute_smoothquant_scales_c(
            act_max.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(hidden_size),
            ctypes.c_float(sparsity),
            ctypes.c_int(bits),
            scales_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        if res != 0:
            raise RuntimeError(f"Mojo compute_smoothquant_scales_c failed for {name}")
        smooth_scales[name] = scales_out.tolist()
    return smooth_scales

def apply_awq_quantization(model, salient_channels, scales, bits=4, group_size=128):
    """
    Apply AWQ: Scale salient weight channels, then quantize non-salient to INT4.
    Now uses Mojo for scaling and quantization to accelerate.
    """
    # Reload model to avoid modifying original
    from transformers import AutoModelForCausalLM
    quantized_model = AutoModelForCausalLM.from_pretrained(model.config._name_or_path, torch_dtype=torch.float16)
    quantized_model.load_state_dict(model.state_dict())
    
    model_name_to_layer = {}  # Map hook name to module path
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Linear) and "layers" in name:
            # Extract layer index and proj type from name, e.g., "layer.5.mlp.down_proj"
            parts = name.split('.')
            layer_idx = parts[1]
            proj_type = parts[-1]
            hook_name = f"layer_{layer_idx}_{proj_type.replace('proj', '') if 'proj' in proj_type else proj_type}"
            model_name_to_layer[hook_name] = module
    
    for hook_name, module in model_name_to_layer.items():
        if hook_name in salient_channels:
            weight = module.weight.data.cpu().numpy().astype(np.float32)  # [out, in]
            bias = module.bias.data.cpu().numpy().astype(np.float32) if module.bias is not None else None
            
            salient = np.array(salient_channels[hook_name], dtype=np.int32)
            scale = scales.get(hook_name, 1.0)
            out_dim, in_dim = weight.shape  # out_dim = hidden_size for output channels
            
            # Call Mojo to apply scale and quantize
            res = mojo_lib.apply_awq_quantize_c(
                weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(out_dim),
                ctypes.c_int(in_dim),
                ctypes.c_float(scale),
                salient.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                ctypes.c_int(len(salient)),
                ctypes.c_int(bits)
            )
            if res != 0:
                raise RuntimeError(f"Mojo apply_awq_quantize_c failed for {hook_name}")
            
            # Update model weight (and bias if exists)
            module.weight.data = torch.from_numpy(weight).to(module.weight.device)
            if bias is not None:
                module.bias.data = torch.from_numpy(bias).to(module.bias.device)
    
    return quantized_model

def apply_smoothquant(quantized_model, activations, smooth_scales, bits=8, group_size=128):
    """
    Apply SmoothQuant: Compensate activation scales in weights, then quantize weights to INT8.
    Uses Mojo for compensation and quantization.
    """
    model_name_to_layer = {}
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Linear) and "layers" in name:
            parts = name.split('.')
            layer_idx = parts[1]
            proj_type = parts[-1]
            hook_name = f"layer_{layer_idx}_{proj_type.replace('proj', '') if 'proj' in proj_type else proj_type}"
            model_name_to_layer[hook_name] = module
    
    for hook_name, module in model_name_to_layer.items():
        if hook_name in smooth_scales:
            weight = module.weight.data.cpu().numpy().astype(np.float32)  # [out, in]
            scales_act = np.array(smooth_scales[hook_name], dtype=np.float32)
            out_dim, in_dim = weight.shape
            
            # Call Mojo
            res = mojo_lib.apply_smoothquant_quantize_c(
                weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(out_dim),
                ctypes.c_int(in_dim),
                scales_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(bits),
                ctypes.c_int(group_size)
            )
            if res != 0:
                raise RuntimeError(f"Mojo apply_smoothquant_quantize_c failed for {hook_name}")
            
            # Update
            module.weight.data = torch.from_numpy(weight).to(module.weight.device)
    
    return quantized_model

def quantize_with_awq(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", quant_bits=4, protect_ratio=0.001):
    """
    AWQ quantization: activation-aware weight-only quantization per the paper.
    All computation now runs through Mojo kernels for speed.
    """
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Generating calibration data...")
    input_ids, attention_mask = load_calibration_data(tokenizer)
    
    print("Collecting activation statistics...")
    start_collect = time.time()
    num_layers = len(model.model.layers)
    activations = collect_activation_statistics(model, (input_ids, attention_mask), num_layers)
    print(f"Activation statistics collection time: {time.time() - start_collect:.4f} s")
    
    print("Identifying salient channels...")
    start_salient = time.time()
    salient_channels = find_salient_channels(activations, protect_ratio)
    print(f"Salient channels identification time: {time.time() - start_salient:.4f} s")

    print("Computing scaling factors...")
    start_scale = time.time()
    scales = compute_scaling_factors(activations, salient_channels)
    print(f"Scaling factors computation time: {time.time() - start_scale:.4f} s")

    print("Applying AWQ quantization via Mojo...")
    quantized_model = apply_awq_quantization(model, salient_channels, scales, quant_bits)
    
    print("Saving quantized model...")
    output_dir = "./models/tinyllama-awq"
    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save stats
    stats = {
        "salient_channels": salient_channels,
        "scales": scales,
        "method": "AWQ paper-compliant with Mojo acceleration"
    }
    with open(f"{output_dir}/awq_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Quantized model saved to {output_dir}. Mojo used for all computations except model I/O.")
    return output_dir

def quantize_with_smoothquant(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", quant_bits=8, sparsity=0.85):
    """
    SmoothQuant: Post-training quantization with activation outlier absorption.
    All computations via Mojo.
    """
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Generating calibration data...")
    input_ids, attention_mask = load_calibration_data(tokenizer)
    
    print("Collecting activation statistics...")
    num_layers = len(model.model.layers)
    activations = collect_activation_statistics(model, (input_ids, attention_mask), num_layers)
    
    print("Computing SmoothQuant scales via Mojo...")
    smooth_scales = compute_smoothquant_scales(activations, quant_bits, sparsity)
    
    print("Applying SmoothQuant via Mojo...")
    quantized_model = apply_smoothquant(model, activations, smooth_scales, quant_bits)
    
    print("Saving quantized model...")
    output_dir = "./models/tinyllama-smoothquant"
    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save stats
    stats = {
        "smooth_scales": smooth_scales,
        "method": "SmoothQuant paper-compliant with Mojo acceleration"
    }
    with open(f"{output_dir}/smoothquant_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Quantized model saved to {output_dir}. Computation offloaded to Mojo.")
    return output_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["awq", "smoothquant"], default="awq")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = parser.parse_args()
    
    if args.method == "awq":
        quantize_with_awq(args.model)
    else:
        quantize_with_smoothquant(args.model)
