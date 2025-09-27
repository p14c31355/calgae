#!/bin/bash

# AWQ Quantization Setup and Run for CalGae Project

# Check if llm-compressor is cloned
if [ ! -d "llm-compressor" ]; then
  echo "Cloning llm-compressor..."
  git clone https://github.com/vllm-project/llm-compressor.git
fi

# Create and activate venv for llm-compressor
cd llm-compressor
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate

# Install llm-compressor
pip install -e .

# Run AWQ quantization on TinyLlama
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
Q_BITS=4
Q_GROUP_SIZE=128
OUTPUT="tinyllama_awq.pt"

python -m llmcompressor.awq --model $MODEL --q_bits $Q_BITS --q_group_size $Q_GROUP_SIZE --dump_awq $OUTPUT

echo "AWQ quantization completed. Scaling factors saved to $OUTPUT"
