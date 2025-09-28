# Calgae 🐚: Lightweight LLM Runtime and Model Stack

[![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust)](https://www.rust-lang.org/)
[![Zig](https://img.shields.io/badge/Zig-00D4B6?style=for-the-badge&logo=zig)](https://ziglang.org/)
[![Mojo](https://img.shields.io/badge/Mojo-FF6B35?style=for-the-badge&logo=modular)](https://www.modular.com/mojo)
[![License: MIT/Apache](https://img.shields.io/badge/License-MIT--or--Apache%202.0-blue.svg)](./docs/LICENSE-MIT)
[![GitHub Repo stars](https://img.shields.io/github/stars/p14c31355/calgae?style=social)](https://github.com/p14c31355/calgae)

Calgae is a high-performance, resource-efficient LLM inference runtime engineered with modern safe systems languages: **Rust**, **Zig**, **Mojo**, **Codon**, and **Lean4**. Designed for portability and safety, it emphasizes quantization, distillation, and formal verification to enable efficient CPU-based inference without compromising accuracy.

Explore the fusion of cutting-edge languages for AI acceleration!

## 🚀 Key Features

- **⚡ Efficient Inference**: Powered by Candle in Rust, supporting quantized models (int4/int8) for blazing-fast performance.
- **🔌 Acceleration Plugins**: Seamless integration of SIMD-optimized kernels in Zig, Codon, and Mojo for matmul and GEMM operations.
- **✅ Formal Verification**: Lean4 proofs ensuring quantization correctness and runtime safety.
- **🔧 Plugin System**: Unified API for easy extension with Mojo, Codon, or Zig accelerators.
- **🛠️ Quantization Tools**: Built-in AWQ scripting for models like TinyLlama, enabling model compression.

## 📦 Installation

Get started in minutes! Calgae requires a few modern toolchains.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/p14c31355/calgae.git
   cd calgae
   ```

2. **Install Dependencies**:
   - **Rust**: 
     ```bash
     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
     ```
   - **Zig**: Download from [ziglang.org](https://ziglang.org/download/) and add to your PATH.
   - **Codon**: Install from [codon.com](https://codon.com).
   - **Mojo**: Get it from [Modular](https://www.modular.com/mojo).
   - **Lean4**: Follow the [quickstart](https://lean-lang.org/lean4/doc/quickstart.html).

3. **Build the Project**:
   ```bash
   cd core
   cargo build --release
   ```

4. **Download a Model**:
   ```bash
   cargo run --bin xtask -- fetch-model
   ```

## 🏃 Usage

Run inference with the built-in agent:

```bash
cargo run -- --model models/tinyllama --prompt "Write a hello world in Rust" --tokens 128 --temperature 0.7 --top-k 50 --top-p 0.9
```

### Command-Line Options
- `--model`: Path to the model directory (e.g., TinyLlama in safetensors).
- `--prompt`: Your input prompt.
- `--tokens`: Maximum output tokens.
- `--temperature`: Controls randomness (0.0-1.0).
- `--top-k`: Top-k sampling parameter.
- `--top-p`: Top-p (nucleus) sampling.
- `--execute`: Compile and execute generated code.
- `--interactive`: Enable interactive chat mode.

## 🏗️ Architecture Overview

Calgae's modular design separates concerns for easy extension:

| Component | Language | Purpose |
|-----------|----------|---------|
| **Core** | Rust | LLM inference engine with Candle; agent orchestration. |
| **Runtime** | Zig | Low-level SIMD kernels (e.g., matrix multiplication). |
| **ML Plugins** | Codon / Mojo | High-performance compute kernels for acceleration. |
| **Proofs** | Lean4 | Formal verification of critical algorithms like quantization. |
| **LLM-Compressor** | Python | AWQ quantization scripts for model optimization. |

Visualize the stack: Core Rust runtime orchestrates plugins, offloading compute to specialized kernels while Lean4 verifies safety.

## 🔄 Quantization

Compress models efficiently with AWQ:

```bash
cargo run --bin xtask -- awq-quantize
```

This creates a virtual environment, processes the model, and outputs to `models/tinyllama-awq`. Supports TinyLlama and similar architectures.

## 🤝 Contributing

We welcome contributions to accelerate AI on safe systems!

- **Add Plugins**: Extend `core/src/agent.rs` for new accelerators.
- **Enhance Proofs**: Build on `proof/src/` for more verifications.
- **Optimize Kernels**: Improve `ml/` (Codon/Mojo) or `runtime/` (Zig).
- **Report Issues**: Open a GitHub issue for bugs or features.

Publish Rust crates to [crates.io](https://crates.io/) and release the full stack via GitHub.

For detailed guidelines, see [CONTRIBUTING.md](docs/architecture.md) (TBD).

## 📄 License

Dual-licensed under [MIT](docs/LICENSE-MIT) or [Apache-2.0](docs/LICENSE-APACHE). See the LICENSE files for details.

---

⭐ **Star this repo if Calgae sparks your interest in safe AI systems!** Questions? Open an issue on GitHub.
