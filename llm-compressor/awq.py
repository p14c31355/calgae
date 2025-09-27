#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Any
import json

def load_calibration_data(tokenizer, num_samples=128, max_length=2048):
    """
    Load or generate calibration data (dummy texts or from dataset).
    """
    # Simple dummy calibration: random prompts
    calibration_texts = [
        "Once upon a time, " * (max_length // 20) for _ in range(num_samples)
    ]
    calibration_inputs = tokenizer(calibration_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return calibration_inputs["input_ids"], calibration_inputs["attention_mask"]

def collect_activation_statistics(model, inputs, layers_to_collect=32):
    """
    Collect per-channel activation statistics (max abs) during forward pass.
    Focus on attention layers or linear layers.
    """
    model.eval()
    activations = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            # Assume output is activation tensor, collect channel-wise max
            if isinstance(output, tuple):
                output = output[0]  # For some layers
            act_max = torch.amax(torch.abs(output), dim=(0, 1))  # Per-channel max
            activations[name] = act_max
        return hook
    
    # Register hooks on linear layers in transformer blocks
    for i in range(layers_to_collect):
        layer = model.model.layers[i].mlp  # Example: MLP layers
        hook_mlp = layer.down_proj.register_forward_hook(hook_fn(f"layer_{i}_down"))
        hooks.append(hook_mlp)
        hook_up = layer.up_proj.register_forward_hook(hook_fn(f"layer_{i}_up"))
        hooks.append(hook_up)
    
    with torch.no_grad():
        for batch in range(0, len(inputs[0]), 8):  # Batch size 8
            batch_ids = inputs[0][batch:batch+8]
            batch_mask = inputs[1][batch:batch+8]
            output = model(input_ids=batch_ids, attention_mask=batch_mask)
            # Dummy loss to run forward
            loss = output.logits.mean()
            loss.backward()  # Optional for gradients if needed, but AWQ is forward-only
    
    for h in hooks:
        h.remove()
    
    return activations

def find_salient_channels(activations, top_k_percent=0.01):
    """
    Identify salient channels based on activation max (protection score).
    """
    salient_channels = {}
    for name, act_max in activations.items():
        # Sort channels by activation magnitude
        sorted_indices = torch.argsort(act_max, descending=True)
        num_salient = int(len(act_max) * top_k_percent)
        salient_channels[name] = sorted_indices[:num_salient]
    return salient_channels

def compute_scaling_factors(activations, salient_channels):
    """
    Compute scale factors to protect salient channels.
    Scale = max_act_salient / max_act_all to reduce error.
    """
    scales = {}
    for name, act_max in activations.items():
        salient_max = torch.max(act_max[salient_channels[name]])
        overall_max = torch.max(act_max)
        scale = overall_max / (salient_max + 1e-8)  # Avoid div0
        scales[name] = scale.item()
    return scales

def apply_awq_quantization(model, salient_channels, scales, bits=4, group_size=128):
    """
    Apply AWQ: Scale salient channels, quantize weights to INT bits.
    Protect salient by keeping FP16, others quantized.
    """
    model.eval()
    quantized_model = AutoModelForCausalLM.from_pretrained(model.config._name_or_path)  # Reload for mod
    quantized_model.load_state_dict(model.state_dict(), strict=False)
    
    quantize = torch.quantization.get_default_qconfig('fbgemm')  # Or custom
    quantized_model.qconfig = quantize
    torch.quantization.prepare(quantized_model, inplace=True)
    
    # Custom: For each linear layer
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Linear) and "layer" in name:
            weight = module.weight.data
            # Apply scaling to salient channels (columns in weight)
            if name in salient_channels:
                salient = salient_channels[name]
                scale = scales.get(name, 1.0)
                weight[:, salient] *= scale  # Scale salient channels
            
            # Quantize (manual for weight-only)
            if hasattr(weight, 'quantize_'):
                weight.quantize_(bits=bits, group_size=group_size)
            else:
                # Manual round-to-nearest INT quantization
                scale_factor = 2 ** (bits - 1) - 1
                qweight = torch.round(weight / (weight.abs().max() / scale_factor))
                qweight.clamp_(-scale_factor, scale_factor)
                weight.data = qweight * (weight.abs().max() / scale_factor)
    
    torch.quantization.convert(quantized_model, inplace=True)
    return quantized_model

def quantize_with_awq(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", quant_bits=4, protect_ratio=0.01):
    """
    Custom AWQ quantization implementation based on the paper.
    """
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Generating calibration data...")
    input_ids, attention_mask = load_calibration_data(tokenizer)
    
    print("Collecting activation statistics...")
    activations = collect_activation_statistics(model, (input_ids, attention_mask))
    
    print("Identifying salient channels...")
    salient_channels = find_salient_channels(activations, protect_ratio)
    
    print("Computing scaling factors...")
    scales = compute_scaling_factors(activations, salient_channels)
    
    print("Applying AWQ quantization...")
    quantized_model = apply_awq_quantization(model, salient_channels, scales, quant_bits)
    
    print("Saving quantized model...")
    output_dir = "./models/tinyllama-awq-custom"
    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save stats for reference
    stats = {
        "salient_channels": {k: v.tolist() for k, v in salient_channels.items()},
        "scales": scales
    }
    with open(f"{output_dir}/awq_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Quantized model saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    quantize_with_awq()
