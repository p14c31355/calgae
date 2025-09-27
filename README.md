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

1. Clone the repository and its submodules (if engine is set as submodule, otherwise clone manually):
```bash
git clone --recurse-submodules https://github.com/p14c31355/calgae.git
cd calgae
```
(Note: If you've already cloned without --recurse-submodules, run `git submodule update --init --recursive` inside the repo.)

2. Build the engine:
```bash
cd engine
mkdir -p build && cd build
cmake .. -DLLAMA_CURL=OFF
cmake --build . --config Release
cd ../..
```

3. Download the model:
```bash
mkdir -p models && wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf -O models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
```

## Usage

### Rust Agent
```bash
cd core && cargo run -- --llama-bin ../engine/build/bin/llama-cli --model ../models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --prompt "Generate a Rust function to compute fibonacci sequence"
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
cd engine/build/bin && ./llama-cli -m ../../models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --prompt "Hello, my name is" -n 50 --log-disable
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
./scripts/setup_all.sh
./scripts/fetch_model.sh  # Fetch model (first time only)
./scripts/run_cpu_llm.sh  # Run LLM (default prompt for Rust code generation)
```

This will run the basic LLM inference and code generation in Calgae!
