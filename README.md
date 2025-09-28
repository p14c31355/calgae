# Calgae: Lightweight LLM Runtime and Model Stack

Calgae is a high-speed, resource-efficient LLM runtime built with modern safe systems languages: Rust, Zig, Mojo, Codon, and Lean4. It prioritizes quantization, distillation, and formal verification for safe, portable inference on CPU.

## Features

- **Efficient Inference**: Candle-based Rust runtime supporting quantized models (int4/int8).
- **Acceleration Plugins**: Integrated kernels in Zig, Codon, and Mojo for SIMD-optimized matmul and GEMM.
- **Formal Verification**: Lean4 proofs for quantization correctness.
- **Plugin System**: Unified runtime for Mojo/Codon/Zig acceleration.
- **Quantization Tools**: AWQ scripting for TinyLlama and similar models.

## Installation

1. Clone the repo:
   ```
   git clone https://github.com/p14c31355/calgae.git
   cd calgae
   ```

2. Install dependencies:
   - Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
   - Zig: Download from https://ziglang.org/download/ and follow installation instructions to add it to your PATH.
   - Codon: Download from https://codon.com
   - Mojo: Install from https://www.modular.com/mojo
   - Lean4: Follow https://lean-lang.org/lean4/doc/quickstart.html

3. Build:
   ```
   cd core
   cargo build --release
   ```

4. Download a model:
   ```
   cargo run --bin xtask -- fetch-model
   ```

## Usage

Run the agent:
```
cargo run -- --model models/tinyllama --prompt "Write a hello world in Rust" --tokens 128 --temperature 0.7 --top-k 50 --top-p 0.9
```

- `--model`: Path to TinyLlama safetensors model directory.
- `--prompt`: Input prompt.
- `--tokens`: Max generation length.
- `--temperature`: Sampling temperature.
- `--top_k`: Top-k sampling.
- `--top_p`: Top-p sampling.
- `--execute`: Compile and run generated code.
- `--interactive`: Interactive mode.

## Architecture

- **Core (Rust)**: LLM inference with Candle, agent orchestrator.
- **Runtime (Zig)**: SIMD kernels (matmul).
- **ML (Codon/Mojo)**: Accelerated computation kernels.
- **Proof (Lean4)**: Formal verification of quantization.
- **LLM-Compressor**: Python AWQ quantization scripts.

## Quantization

Run AWQ quantization:
```
cargo run --bin xtask -- awq-quantize
```

Creates a virtual env for dependencies and saves to `models/tinyllama-awq`.

## Contributing

- Add new acceleration plugins in `core/src/agent.rs`.
- Extend proofs in `proof/src/`.
- Optimize kernels in `ml/` and `runtime/`.

For publication, consider crates.io for Rust components and GitHub releases for the full stack.

## License

MIT / Apache-2.0 (see LICENSE files).
