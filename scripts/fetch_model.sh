#!/bin/bash

set -e  # Exit on any error

MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
MODEL_PATH="models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

echo "Fetching TinyLlama model from HuggingFace..."

mkdir -p models

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading model (this may take a while)..."
    wget -O "$MODEL_PATH" "$MODEL_URL"
    echo "Model downloaded to $MODEL_PATH"
else
    echo "Model already exists at $MODEL_PATH"
fi

echo "Model setup complete!"
