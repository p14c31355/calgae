#!/bin/bash
# Build all components of Calgae (Updated for flattened structure)

set -e

echo "Building all Calgae components..."

# Build llama.cpp (C++)
cd engine && mkdir -p build && cd build && cmake .. -DLLAMA_CURL=OFF && cmake --build . --config Release && cd ../..

# Build Rust core
cargo build --release

# Build Zig runtime
cd runtime && zig build-exe src/runtime.zig && cd ..

echo "Building Lean4 proof"
cd proof && lake build && cd ..

echo "Installing Python dependencies for ML (if not already installed)"
cd ml/codon && pip install codon --user && cd ../..

# Note: Mojo kernels can be run directly with 'mojo ml/mojo/kernels.mojo'
echo "Mojo kernels ready (no build step needed)"

echo "Build complete!"
