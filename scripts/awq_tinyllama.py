from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import torch.nn as nn

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

# Function to replace NaN in model weights
def replace_nan_in_model(model):
    """
    Replace NaN values in model weights with 0.0 to prevent assertion errors during quantization.
    This is a safety measure for numerical stability issues in float16 loading.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if torch.isnan(module.weight).any():
                print(f"Warning: NaN detected in {name}.weight, replacing with 0.0")
                module.weight.data = torch.nan_to_num(module.weight.data, nan=0.0)
            if module.bias is not None and torch.isnan(module.bias).any():
                print(f"Warning: NaN detected in {name}.bias, replacing with 0.0")
                module.bias.data = torch.nan_to_num(module.bias.data, nan=0.0)
    print("NaN replacement completed.")

# Select model and load it.
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
replace_nan_in_model(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 128 samples for small model to speed up.
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 512

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

ds = ds.map(preprocess)

# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
recipe = [
    AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
]

# Apply algorithms.
replace_nan_in_model(model)
oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = "quantized-tinyllama-awq"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Quantized model saved to {SAVE_DIR}")
