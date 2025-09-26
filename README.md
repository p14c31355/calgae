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

### Zig
```bash
wget https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz && tar -xf zig-linux-x86_64-0.13.0.tar.xz && sudo mv zig-linux-x86_64-0.13.0 /opt/zig && export PATH=$PATH:/opt/zig
```

### Lean4
```bash
curl https://raw.githubusercontent.com/leanprover/lean4/master/scripts/install_ubuntu.sh | sh && export PATH=$PATH:$HOME/.elan/bin
```

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
sudo apt update && sudo apt install cmake build-essential git wget curl python3-pip python3.13-venv && source ~/.cargo/env
```

## Setup
```bash
mkdir -p {core/src,runtime/src,proof/src,ml/{mojo,codon},engine,models,scripts} && cd calgae && git clone https://github.com/ggerganov/llama.cpp.git engine && cd engine && mkdir -p build && cd build && cmake .. -DLLAMA_CURL=OFF && cmake --build . --config Release && cd ../../.. && wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf -O models/TinyLlama-1.1B-q4_0.gguf
```

## Usage

### Rust Agent
```bash
cd calgae/core && cargo run --prompt "Generate a Rust function to compute fibonacci sequence"
```

### Zig Runtime
```bash
cd calgae/runtime && zig build-exe src/runtime.zig
./src/runtime zig
```

### Lean4 Verification
```bash
cd calgae/proof && lake exe correctness
```

### Mojo Kernel
```bash
cd calgae/ml/mojo && mojo kernels.mojo
```

### Codon Optimized
```bash
cd calgae/ml/codon && python optimize.py
```

## Inference with TinyLlama
```bash
cd calgae/engine/build/bin && ./llama-cli -m ../../models/TinyLlama-1.1B-q4_0.gguf --prompt "Hello, my name is" -n 50 --log-disable
```

## AWQ Quantization
To quantize other models, clone and follow AWQ repo instructions.

## EdgeProfiler Benchmark
```bash
# Download and install EdgeProfiler separately
git clone https://github.com/ruoyuliu/EdgeProfiler.git ../edgeprofiler && cd ../edgeprofiler && pip install -r requirements.txt
# Run benchmark
python -m edgeprofiler.benchmark --model_path calgae/models/TinyLlama-1.1B-q4_0.gguf --backend llama.cpp
```

### Next Steps
- Multi-language code generation from LLM prompts.
- FFI integration for Rust-Zig-Lean-Mojo-Codon.
- Full formal verification pipeline.
- Optimization with Mojo/Codon kernels.
