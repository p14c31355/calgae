# Calgae

[![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Zig](https://img.shields.io/badge/Zig-%23DE0000.svg?style=for-the-badge&logo=zig&logoColor=white)](https://ziglang.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Mojo](https://img.shields.io/badge/Mojo-%23000?style=for-the-badge&logo=modular&logoColor=white)](https://www.modular.com/mojo)
[![Lean](https://img.shields.io/badge/Lean-87208C?style=for-the-badge&logo=lean&logoColor=white)](https://lean-lang.org/)

Calgae is an experimental, high-speed, resource-efficient LLM runtime and model stack built with modern safe systems languages: Rust, Zig, Mojo, Codon, and Lean4. It focuses on quantization, distillation, and formal verification to enable safe, portable inference on CPU, with acceleration plugins for optimized performance. This is a self-made project for exploring LLM inference techniques.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Quantization](#quantization)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Efficient Inference**: Leverages Candle in Rust for supporting quantized models (int4/int8) with low resource usage.
- **Acceleration Plugins**: Custom kernels in Zig, Codon, and Mojo for SIMD-optimized matrix operations like matmul and GEMM.
- **Formal Verification**: Lean4 proofs ensuring quantization correctness and reliability.
- **Plugin System**: Unified runtime interface for seamless integration of Mojo, Codon, and Zig accelerations.
- **Quantization Tools**: AWQ (Activation-aware Weight Quantization) scripts for models like TinyLlama.
- **Code Generation Agent**: Integrated agent for generating, compiling, and executing code based on LLM outputs.

## Quick Start

1. Clone the repository:
   ```
   git clone https://github.com/p14c31355/calgae.git
   cd calgae
   ```

2. Install dependencies (see [Installation](#installation) for details).

3. Build the project:
   ```
   cd core
   cargo build --release
   ```

4. Download a sample model:
   ```
   cargo run --bin xtask -- fetch-model
   ```

5. Run inference:
   ```
   cargo run -- --model models/tinyllama --prompt "Hello, world!" --tokens 50
   ```

## Installation

Calgae requires multiple languages and tools. Ensure you have a compatible environment (Linux/macOS recommended).

### Prerequisites

- **Rust**: Install via [rustup](https://rustup.rs/):
  ```
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **Zig**: Download from [ziglang.org/download](https://ziglang.org/download/) and add to your PATH.
- **Codon**: Download and install from [codon.com](https://codon.com).
- **Mojo**: Install the Modular SDK from [modular.com/mojo](https://www.modular.com/mojo).
- **Lean4**: Follow the [quickstart guide](https://lean-lang.org/lean4/doc/quickstart.html).
- **Python**: For quantization tools (Python 3.10+ with pip).

### Build the Project

```
cd core
cargo build --release
```

This builds the Rust core, linking to Zig and Mojo components where applicable.

### Download Models

Use the built-in xtask to fetch a quantized TinyLlama model:
```
cargo run --bin xtask -- fetch-model
```

Models are saved in the `models/` directory.

## Usage

The main entrypoint is `cargo run` from the `core/` directory. It supports both one-shot inference and interactive modes.

### Basic Inference

Generate text from a prompt:
```
cargo run -- --model models/tinyllama --prompt "Write a simple Rust function to add two numbers." --tokens 100 --temperature 0.7 --top-k 50 --top-p 0.9
```

- `--model`: Path to the model directory (e.g., safetensors files).
- `--prompt`: The input prompt for the LLM.
- `--tokens`: Maximum number of tokens to generate (default: 128).
- `--temperature`: Sampling temperature for creativity (0.0-1.0; default: 0.7).
- `--top-k`: Top-K sampling parameter (default: 50).
- `--top-p`: Top-P (nucleus) sampling parameter (default: 0.9).

### Code Generation and Execution

Use the agent to generate and run code:
```
cargo run -- --model models/tinyllama --prompt "Generate a Python script for Fibonacci sequence." --tokens 200 --execute
```

The `--execute` flag compiles and runs the generated code (supports Rust/Python via the agent).

### Interactive Mode

For conversational interaction:
```
cargo run -- --model models/tinyllama --interactive
```

Enter prompts interactively; type `exit` to quit.

### Quantization

See the [Quantization](#quantization) section for model preparation.

## Architecture

Calgae is modular, with each component handling a specific aspect of LLM inference and acceleration.

- **Core (Rust)**: 
  - Handles LLM inference using Candle tensors.
  - `LlmInference` struct loads models and performs forward passes.
  - `Agent` struct orchestrates code generation, parallel inference, and execution (e.g., `generate_code`, `compile_and_execute`).
  - CLI parsing via `clap` in `cli.rs`.

- **Runtime (Zig)**: Provides low-level SIMD kernels for matrix operations (e.g., `zig_kernel.zig` for matmul).

- **ML (Mojo/Codon)**: High-performance kernels for compute-intensive tasks.
  - Mojo: `awq.mojo`, `kernels.mojo` for quantization-aware operations.
  - Codon: Integrated for alternative acceleration.

- **Proof (Lean4)**: Formal verification of key algorithms, such as quantization correctness (`Correctness.lean`, `proof.lean`).

- **LLM-Compressor (Python)**: Scripts for AWQ and other compression techniques.

The runtime unifies these via plugins, allowing dynamic selection of accelerators in `agent.rs` (e.g., `use_codon_kernel`, `infer_async`).

For a visual overview, refer to `docs/architecture.md`.

## Quantization

Calgae includes tools for AWQ quantization to reduce model size while maintaining performance.

1. Run the quantization xtask:
   ```
   cargo run --bin xtask -- awq-quantize
   ```

2. This creates a Python virtual environment, processes the model (e.g., TinyLlama), and outputs to `models/tinyllama-awq`.

Dependencies are managed automatically. Customize via scripts in `llm-compressor/` or `ml/mojo/awq.mojo`.

## Contributing

Contributions are welcome! This is an experimental project, so focus on:

- Adding new acceleration plugins (e.g., extend `Agent::new` and kernels in `ml/` or `runtime/`).
- Improving proofs in `proof/src/` for new features.
- Optimizing inference in `core/src/inference.rs`.
- Enhancing CLI options in `core/src/cli.rs`.
- Documentation updates in `docs/`.

### Development Workflow

1. Fork and clone the repo.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Make changes and test: `cargo test --release`.
4. Commit and push: `git push origin feature/your-feature`.
5. Open a PR to `develop` branch.

For Rust components, consider publishing to crates.io. Use GitHub Releases for binaries.

Report issues or suggest features via GitHub Issues.

## License

Dual-licensed under [MIT](docs/LICENSE-MIT) or [Apache-2.0](docs/LICENSE-APACHE). See the LICENSE files for details.

---

*Built with ❤️ by [p14c31355](https://github.com/p14c31355). Experimental project – use at your own risk!*
