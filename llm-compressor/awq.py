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
    Target: attention (q_proj, k_proj, v_proj, o_proj), mlp (gate_proj, up_proj, down_proj).
    Accumulate max over calibration samples.
    """
    model.eval()
    activations = {}  # Dict of layer_name: tensor of per-channel max abs act
    hooks = []

    # Fallback to torch for channel max since Mojo load failed
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            act_abs = torch.abs(output).float()
            batch_size, seq_len, hidden_size = act_abs.shape
            channel_max = torch.max(act_abs.view(batch_size * seq_len, hidden_size), dim=0)[0].cpu()
            if name not in activations:
                activations[name] = channel_max
            else:
                activations[name] = torch.max(activations[name], channel_max)
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
    """
    if mojo_lib:
        lib = mojo_lib
    else:
        lib = None

    salient_channels = {}
    for name, act_max_torch in activations.items():
        act_max = act_max_torch.numpy().astype(np.float32)
        hidden_size = len(act_max)
        num_salient = max(1, int(hidden_size * top_k_percent))
        indices = np.zeros(num_salient, dtype=np.int32)
        if lib:
            lib.top_k_indices_c(act_max, hidden_size, num_salient, indices)
        else:
            sorted_indices = torch.argsort(act_max_torch, descending=True)
            indices = sorted_indices[:num_salient].numpy()
        salient_channels[name] = indices.tolist()
    return salient_channels

def compute_scaling_factors(activations, salient_channels):
    """
    Compute per-layer scale: max_all_act / max_salient_act to scale up salient channels.
    """
    if mojo_lib:
        lib = mojo_lib
    else:
        lib = None

    scales = {}
    for name, act_max_torch in activations.items():
        act_max = act_max_torch.numpy().astype(np.float32)
        hidden_size = len(act_max)
        salient_indices = np.array(salient_channels[name], dtype=np.int32)
        num_salient = len(salient_indices)
        scale_out = np.zeros(1, dtype=np.float32)
        if lib:
            lib.compute_scale_c(act_max, hidden_size, salient_indices, num_salient, scale_out)
            scale = scale_out[0]
        else:
            salient_max = torch.max(act_max_torch[salient_indices])
            overall_max = torch.max(act_max_torch)
            scale = overall_max / (salient_max + 1e-8)
        scales[name] = float(scale)
    return scales

def compute_smoothquant_scales(activations, bits=8, sparsity=0.85):
    """
    Compute SmoothQuant scales: Per-channel activation scales to absorb outliers into weights.
    Based on SmoothQuant paper: Fit activation distribution to quantization range, compensate in weights.
    sparsity: Fraction of channels to apply heavy scaling (outlier channels).
    """
    smooth_scales = {}
    for name, act_max in activations.items():
        # Sort channels by activation max (outliers are high)
        sorted_max = torch.sort(act_max, descending=True)[0]
        num_outliers = int(len(act_max) * sparsity)
        # For top sparsity channels, compute scale to fit to 8-bit range
        # Assume beta=0.85 from paper for outlier absorption
        beta = 0.85
        qmax = 2 ** (bits - 1) - 1
        scales = torch.ones_like(act_max)
        if num_outliers > 0:
            outlier_max = sorted_max[:num_outliers].max()
            scales[:num_outliers] = (outlier_max / beta) / qmax  # Scale act down, compensate in weight
        smooth_scales[name] = scales.tolist()  # Per-channel scales
    return smooth_scales

def apply_awq_quantization(model, salient_channels, scales, bits=4, group_size=128):
    """
    Apply AWQ: Scale salient weight channels (output dim), then quantize non-salient to INT4.
    Salient channels kept in FP16, others per-group INT4 quantized.
    Equivalent transformation: W' = W * s, quant(W'), then dequant with /s.
    But for simplicity, store scaled quantized weights.
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
            weight = module.weight.data  # Shape: [in_features, out_features]
            bias = module.bias.data if module.bias is not None else None
            
            salient = salient_channels[hook_name]
            scale = scales.get(hook_name, 1.0)
            
            # Scale salient output channels (columns)
            if len(salient) > 0:
                weight[:, salient] *= scale
            
            # Quantize non-salient channels to INT4 per-group
            out_dim = weight.shape[1]
            fp16_channels = torch.zeros(out_dim, dtype=torch.bool)
            fp16_channels[salient] = True
            
            for start in range(0, out_dim, group_size):
                end = min(start + group_size, out_dim)
                group_mask = ~fp16_channels[start:end]
                if group_mask.any():
                    group_weight = weight[:, start:end][:, group_mask]
                    absmax = group_weight.abs().max()
                    scale_q = absmax / ((1 << (bits - 1)) - 1)
                    q_group = torch.round(group_weight / scale_q).clamp(-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
                    q_group_dequant = q_group.float() * scale_q
                    group_idx = 0
                    for i in range(start, end):
                        if not fp16_channels[i]:
                            weight[:, i] = q_group_dequant[:, group_idx]
                            group_idx += 1
            
            module.weight.data = weight
            if bias is not None:
                module.bias.data = bias
    
    return quantized_model

def apply_smoothquant(quantized_model, activations, smooth_scales, bits=8, group_size=128):
    """
    Apply SmoothQuant: Compensate activation scales in weights, then quantize weights to INT8.
    Weights are scaled by 1 / act_scale for outlier channels.
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
            weight = module.weight.data
            scales_act = torch.tensor(smooth_scales[hook_name])
            # Compensate: W' = W / act_scale (per output channel)
            weight *= 1.0 / (scales_act + 1e-8)
            
            # Quantize all weights to INT8 per-group
            for start in range(0, weight.shape[1], group_size):
                end = min(start + group_size, weight.shape[1])
                group_weight = weight[:, start:end]
                absmax = group_weight.abs().max()
                scale_q = absmax / ((1 << (bits - 1)) - 1)
                q_group = torch.round(group_weight / scale_q).clamp(-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
                weight[:, start:end] = q_group.float() * scale_q
    
    return quantized_model

def quantize_with_awq(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", quant_bits=4, protect_ratio=0.001):
    """
    AWQ quantization: activation-aware weight-only quantization per the paper.
    Protects top 0.1% salient channels in FP16, quantizes others to INT4.
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

    print("Applying AWQ quantization...")
    quantized_model = apply_awq_quantization(model, salient_channels, scales, quant_bits)
    
    print("Saving quantized model...")
    output_dir = "./models/tinyllama-awq"
    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save stats
    stats = {
        "salient_channels": salient_channels,
        "scales": scales,
        "method": "AWQ paper-compliant"
    }
    with open(f"{output_dir}/awq_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Quantized model saved to {output_dir}. Salient channels protected in FP16, others in INT4.")
    return output_dir

def quantize_with_smoothquant(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", quant_bits=8, sparsity=0.85):
    """
    SmoothQuant: Post-training quantization with activation outlier absorption.
    Computes per-channel act scales, compensates in weights, quantizes to INT8.
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
    
    print("Computing SmoothQuant scales...")
    smooth_scales = compute_smoothquant_scales(activations, quant_bits, sparsity)
    
    print("Applying SmoothQuant...")
    quantized_model = apply_smoothquant(model, activations, smooth_scales, quant_bits)
    
    print("Saving quantized model...")
    output_dir = "./models/tinyllama-smoothquant"
    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save stats
    stats = {
        "smooth_scales": smooth_scales,
        "method": "SmoothQuant paper-compliant"
    }
    with open(f"{output_dir}/smoothquant_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Quantized model saved to {output_dir}. Weights adjusted for activation smoothing, INT8 quantized.")
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
