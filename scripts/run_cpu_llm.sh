#!/bin/bash

set -e  # Exit on any error

echo "Running CPU LLM with Calgae..."

# Ensure model is fetched
./scripts/fetch_model.sh

# Run the Rust agent with default prompt
cd core
cargo run -- --model ../models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --prompt "Generate a simple Rust hello world program."

echo "LLM run complete!"
