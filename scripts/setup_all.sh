#!/bin/bash

set -e  # Exit on any error

echo "Setting up Calgae dependencies..."

# Update system packages
echo "Updating system packages..."
sudo apt update

# Install base dependencies
echo "Installing base dependencies..."
sudo apt install -y build-essential cmake git curl wget python3 python3-pip

# Rust setup
echo "Setting up Rust..."
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    source "$HOME/.cargo/env"
fi

# Zig setup
echo "Setting up Zig..."
ZIG_VERSION="0.13.0"  # Update to latest stable
ZIG_URL="https://ziglang.org/download/${ZIG_VERSION}/zig-linux-x86_64-${ZIG_VERSION}.tar.xz"
if ! command -v zig &> /dev/null; then
    wget "$ZIG_URL"
    tar -xvf zig-linux-x86_64-${ZIG_VERSION}.tar.xz
    sudo mv zig-linux-x86_64-${ZIG_VERSION} /usr/local/
    rm zig-linux-x86_64-${ZIG_VERSION}.tar.xz
    export PATH="/usr/local/zig-linux-x86_64-${ZIG_VERSION}:$PATH"
else
    echo "Zig already installed."
fi

# Lean4 setup
echo "Setting up Lean4..."
if ! command -v lake &> /dev/null; then
    curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh
    source ~/.elan/env
else
    source ~/.elan/env
fi

# Mojo setup (Modular)
echo "Setting up Mojo..."
if ! command -v mojo &> /dev/null; then
    curl https://get.modular.com | sh -s -- -y
    source ~/.modular/bin/activate
    modular install mojo
else
    source ~/.modular/bin/activate
fi

# Codon setup
echo "Setting up Codon..."
pip3 install codon

# Engine (llama.cpp) setup
echo "Setting up llama.cpp engine..."
git submodule update --init --recursive
cd engine
if [ ! -f "build/bin/llama-cli" ]; then
    make clean
    make -j $(nproc)
fi
cd ..

echo "Setup complete! Activate environments if needed (e.g., source ~/.elan/env, source ~/.modular/bin/activate)."
