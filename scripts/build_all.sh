#!/bin/bash
# Build all components of Calgae (Updated for flattened structure)

set -e

echo "Building all Calgae components..."

# Build llama.cpp (C++)
(cd engine && mkdir -p build && cd build && cmake .. -DLLAMA_CURL=OFF && cmake --build . --config Release)

# Build Rust core
cargo build --release

# Build Zig runtime
cd runtime && zig build-exe src/runtime.zig && cd ..

echo "Building Lean4 proof"
(cd proof && lake build)

echo "Installing Python dependencies for ML (if not already installed)"
if ! python3 -c "import codon" 2> /dev/null; then
    echo "Installing Codon..."
    pip install codon --user
fi

# Note: Mojo kernels can be run directly with 'mojo ml/mojo/kernels.mojo'
echo "Mojo kernels ready (no build step needed)"

echo "Build complete!"
