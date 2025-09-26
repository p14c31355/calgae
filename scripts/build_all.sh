#!/bin/bash
# Build all components of Calgae

set -e

echo "Building all Calgae components..."

# Build llama.cpp (C++)
cd engine && mkdir -p build && cd build && cmake .. -DLLAMA_CURL=OFF && cmake --build . --config Release && cd ../..

# Build Rust core
cd core && cargo build --release && cd ..

# Build Zig runtime
cd runtime && zig build

echo "Building Lean4 proof"
(cd proof && lake build)

echo "Installing Python dependencies for ML"
cd ml/codon && pip install codon && cd ../..

# Note: Mojo build requires Modular toolchain
echo "Mojo kernels ready (no build step needed)"

echo "Build complete!"
