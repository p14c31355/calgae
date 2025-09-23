# calgae
Calc Algae - self made LLM project (experimental)

# env-oneliner
```sh
#（Rust + Zig + Lean4 + Python + CMake）
sudo apt update && sudo apt install -y \
    curl git build-essential cmake python3 python3-venv python3-pip pkg-config \
    llvm-dev libclang-dev clang \
    unzip zip wget

# Rust (with rustup)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Zig (latest prebuilt binary)
ZIG_VERSION=0.13.0
wget https://ziglang.org/download/$ZIG_VERSION/zig-linux-x86_64-$ZIG_VERSION.tar.xz
tar -xf zig-linux-x86_64-$ZIG_VERSION.tar.xz
sudo mv zig-linux-x86_64-$ZIG_VERSION /opt/zig
sudo ln -sf /opt/zig/zig /usr/local/bin/zig

# Lean4
git clone https://github.com/leanprover/lean4.git
cd lean4 && mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install
cd ../..

# Python: quantization / LLM tooling (AWQ, GPTQ, etc.)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch transformers accelerate
pip install git+https://github.com/mit-han-lab/llm-awq.git
pip install auto-gptq

# llama.cpp (for lightweight inference)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make -j$(nproc) && cd ..
```

# env Check commands
```sh
rustc --version     # Rust OK
zig version         # Zig OK
lean --version      # Lean4 OK
python3 -m torch.utils.collect_env  # PyTorch OK
./llama.cpp/main -h # llama.cpp OK
```