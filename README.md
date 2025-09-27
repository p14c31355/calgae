# calgae
Calc Algae - self made LLM project (experimental)

## Project Structure

```
├── core/               # Main orchestrator in Rust
├── runtime/            # Zig layer (FFI / OS)
├── proof/              # Lean4: math proof experimental
├── ml/                 # Mojo / Codon: calculate and machine-learning experimental
│   ├── mojo/
│   │   └── kernels.mojo   # HPC based calc kernel
│   └── codon/
│       └── optimize.py    # LLVM tuning calc by codon
├── engine/             # llama.cpp (submodule or clone)
├── models/             # model (quantization OK)
├── scripts/            # builder and runner
│   ├── build_all.sh
│   └── run_cpu_llm.sh
└── README.md
```

## Prerequisites

### Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source ~/.cargo/env
```
Note: To make the PATH permanent, add `source ~/.cargo/env` to your shell configuration file (e.g., ~/.bashrc or ~/.zshrc) and then source it or open a new terminal.

### Zig
```bash
wget https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz && tar -xf zig-linux-x86_64-0.13.0.tar.xz && mv zig-linux-x86_64-0.13.0 ~/.local/zig && export PATH=$PATH:~/.local/zig
```
Note: To make the PATH permanent, add `export PATH=$PATH:~/.local/zig` to your shell configuration file (e.g., ~/.bashrc or ~/.zshrc) and then source it or open a new terminal.

### Lean4
```bash
curl -O https://raw.githubusercontent.com/leanprover/lean4/master/scripts/install_ubuntu.sh
# ... inspect the script ...
sh install_ubuntu.sh
export PATH=$PATH:$HOME/.elan/bin
```
Note: To make the PATH permanent, add `export PATH=$PATH:$HOME/.elan/bin` to your shell configuration file (e.g., ~/.bashrc or ~/.zshrc) and then source it or open a new terminal.

### Mojo
```bash
# Create Modular account and follow https://developer.modular.com/download
pip install modular && modular install mojo
```

### Codon
```bash
pip install codon
```

### Other dependencies
```bash
sudo apt update && sudo apt install cmake build-essential git wget curl python3-pip python3.12-venv
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/p14c31355/calgae.git
cd calgae
```

2. Setup dependencies and fetch model using xtask:
```bash
cargo run --bin xtask -- setup
cargo run --bin xtask -- fetch-model  # Downloads safetensors model to models/tinyllama
```

## Usage

### Rust Agent
```bash
cargo run --model models/tinyllama --prompt "Generate a Rust function to compute fibonacci sequence"
```

### Zig Runtime
```bash
cd runtime && zig build-exe src/runtime.zig
./runtime zig
```


### Lean4 Verification
```bash
cd proof && lake build
```

### Mojo Kernel
```bash
cd ml/mojo && mojo kernels.mojo
```

### Codon Optimized
```bash
cd ml/codon && python optimize.py
```

## Inference with TinyLlama
```bash
cargo run --model models/tinyllama --prompt "Hello, my name is" --tokens 50
```

## AWQ Quantization
To quantize other models, clone and follow AWQ repo instructions.

## EdgeProfiler Benchmark
```bash
# Download and install EdgeProfiler separately
mkdir -p tools && git clone https://github.com/ruoyuliu/EdgeProfiler.git tools/edgeprofiler && cd tools/edgeprofiler && pip install -r requirements.txt
# Run benchmark
python -m edgeprofiler.benchmark --model_path ../../models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --backend llama.cpp
```
Note: The `tools/` directory is included in `.gitignore` to avoid committing downloaded tools.

## 5-Second Quick Start Demo

After cloning, the following commands complete the setup and execution (model download may take time on first run).

```bash
git clone https://github.com/p14c31355/calgae.git
cd calgae
cargo run --bin xtask -- setup
cargo run --bin xtask -- fetch-model  # Fetch model (first time only)
cargo run --model models/tinyllama --prompt "Generate a Rust function to compute fibonacci sequence"  # Run LLM (default prompt for Rust code generation)
```

This will run the basic LLM inference and code generation in Calgae!
