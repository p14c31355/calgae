# `AGENT.md`

> **Project Goal**
> Build a **lightweight, safe, high-speed LLM runtime and model stack** using modern safe systems languages (Mojo, Codon, Zig, Lean4, Rust).
> Prioritize: **minimum compute resource usage**, **quantization + distillation**, **portable runtimes (native, specialized backends)**, and **formal verification of correctness (Lean4)**.

---

## 1. Language & Toolchain Choices

### 🔥 Mojo

* Official: [https://www.modular.com/mojo](https://www.modular.com/mojo)
* GitHub: (Modular provides the official toolchain)
* Key points:

  * Combines **Python's usability** with **C's performance**.
  * Excellent for **AI/ML acceleration** and vector-heavy operations.
  * Designed for **hardware-native performance** on specialized accelerators.

---

### ⚡ Codon

* Official: [https://codon.com](https://codon.com)
* GitHub: [https://github.com/exaloop/codon](https://github.com/exaloop/codon)
* Key points:

  * High-performance, **compiled Python dialect**.
  * Focus on generating **highly optimized native code**.
  * Suitable for creating **minimal, fast LLM inference binaries**.

---

### ⚡ Zig

* Official: [https://ziglang.org](https://ziglang.org)
* GitHub: [https://github.com/ziglang/zig](https://github.com/ziglang/zig)
* Key points:

  * Predictable memory management.
  * No hidden control flow.
  * Great for **high-performance runtime kernels** (matrix mult, SIMD ops).
  * Easy to integrate with C (can wrap BLAS-like libs or ggml kernels).

---

### 📐 Lean4

* Official: [https://lean-lang.org](https://lean-lang.org)
* GitHub: [https://github.com/leanprover/lean4](https://github.com/leanprover/lean4)
* Key points:

  * **Formal verification**: prove safety & correctness of inference kernels.
  * Use Lean4 to:

    * Verify quantization routines.
    * Prove properties about weight transformations.
    * Formally reason about numerical errors.

---

### 🦀 Rust

* Official: [https://www.rust-lang.org](https://www.rust-lang.org)
* Ecosystem:

  * [llm](https://github.com/rustformers/llm) (Rust LLM runtime)
  * [candle](https://github.com/huggingface/candle) (minimal ML framework by Hugging Face, Rust-based)
  * [burn](https://github.com/tracel-ai/burn) (ML framework, Rust/WGPU backend)
* Key points:

  * Safety + performance.
  * Rich async ecosystem (good for agents).
  * Existing ggml bindings.

---

## 2. Core Model Efficiency Techniques

### 2.1 Quantization

* **AWQ** (Activation-aware Weight Quantization)

  * Repo: [https://github.com/mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)
  * Paper: ["AWQ: Activation-aware Weight Quantization"](https://arxiv.org/abs/2306.00978)
  * Best current trade-off between low-bit quantization and accuracy.

* **M-ANT** (Mixed-precision + Activation-Noise Tolerance)

  * Paper: [https://arxiv.org/abs/2403.04652](https://arxiv.org/abs/2403.04652)
  * Great for 3–4bit quantization.

* **GPTQ**

  * Repo: [https://github.com/IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
  * Efficient quantization baseline.

* **Any-Precision LLM**

  * Paper: ["Any-Precision LLM: Low-Cost Deployment"](https://arxiv.org/abs/2405.13792)
  * Idea: dynamically use different bitwidths depending on resources.

---

### 2.2 Distillation

* **Knowledge Distillation** overview:

  * Hinton et al., ["Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531)
* **TinyLlama** (7B → 1.1B distilled model)

  * Repo: [https://github.com/jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)
* **Alpaca / DistilLLaMA / Flan** — strong open datasets for instruction tuning.

---

### 2.3 Model Compression & Acceleration

* **Low-rank Adaptation (LoRA)**

  * Repo: [https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)
* **QLoRA**

  * Paper: ["QLoRA: Efficient Finetuning"](https://arxiv.org/abs/2305.14314)
  * Repo: [https://github.com/artidoro/qlora](https://github.com/artidoro/qlora)

---

## 3. Runtime & Inference

### ggml / llama.cpp family

* **llama.cpp**

  * Repo: [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
  * Lightweight C/C++ inference with quantization support.
* **ggml**

  * Repo: [https://github.com/ggerganov/ggml](https://github.com/ggerganov/ggml)
  * Tensor library for CPU inference (SIMD accelerated).

### Rust runtimes

* **llm (Rustformers)**

  * Repo: [https://github.com/rustformers/llm](https://github.com/rustformers/llm)
* **candle (Hugging Face)**

  * Repo: [https://github.com/huggingface/candle](https://github.com/huggingface/candle)
* **burn**

  * Repo: [https://github.com/tracel-ai/burn](https://github.com/tracel-ai/burn)

### High-Performance Python Backends

* **Mojo** for accelerated inference kernels.
* **Codon** for highly optimized, self-contained native execution.

---

## 4. Profiling & Benchmarking

* **EdgeProfiler**

  * Paper: ["EdgeProfiler: Benchmarking Lightweight LLMs"](https://arxiv.org/abs/2409.06611)
  * Tool repo: [https://github.com/ruoyuliu/EdgeProfiler](https://github.com/ruoyuliu/EdgeProfiler)
  * Useful for **latency / throughput / memory profiling**.

* **MLPerf Tiny / Inference**

  * Official: [https://mlcommons.org/en/inference-tiny/](https://mlcommons.org/en/inference-tiny/)

---

## 5. Suggested Workflow

### Phase A: Bootstrapping

1. Pick a **small base model** (e.g., TinyLlama, Phi-2, or 1–3B model).
2. Quantize with **AWQ or M-ANT**.
3. Run with **llama.cpp / Rust llm**.
4. Benchmark using **EdgeProfiler**.

### Phase B: Optimization

* Experiment with **Any-precision scheduling**.
* Apply **LoRA / QLoRA** finetunes for specific tasks.
* Add Zig kernels for **SIMD-heavy ops**.
* Port critical, performance-sensitive logic to **Mojo or Codon** for native speed.

### Phase C: Runtime Engineering

* Build high-speed inference/pre/post-processing components in **Mojo/Codon**.
* Formalize + verify quantization correctness in **Lean4**.
* Build **Rust orchestrator** (agent system, I/O pipelines).

---

## 6. Agents & Integration

* **LangChain.rs** (Rust port of LangChain): [https://github.com/sobelio/llm-chain](https://github.com/sobelio/llm-chain)
* **Semantic Kernel (Rust unofficial ports exist)**: [https://github.com/microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel)

Agents should be:

* Event-driven (Rust async).
* Extensible with **Mojo/Codon** for specialized, accelerated components.
* Verifiable logic (Lean4 proofs).

---

## 7. Research References (Key Reading)

* **Quantization**

  * AWQ: [https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)
  * M-ANT: [https://arxiv.org/abs/2403.04652](https://arxiv.org/abs/2403.04652)
  * Any-Precision: [https://arxiv.org/abs/2405.13792](https://arxiv.org/abs/2405.13792)

* **Distillation**

  * Hinton (2015): [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)
  * TinyLlama: [https://github.com/jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)

* **Efficient Runtimes**

  * llama.cpp: [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
  * EdgeProfiler: [https://arxiv.org/abs/2409.06611](https://arxiv.org/abs/2409.06611)

---

## 8. Taskboard (Kanban Style)

### ✅ Short-term

* [x] Quantize small model with AWQ.
* [x] Run inference in llama.cpp.
* [x] Benchmark with EdgeProfiler.

### 🔜 Mid-term

* [x] Build Rust orchestrator.
* [ ] Add Zig kernels for matmul.
* [ ] Deploy accelerated components in **Mojo/Codon**.

### 🎯 Long-term

* [x] Formalize quantization correctness in Lean4.
* [x] Create unified agent runtime with plugin system (leveraging Mojo/Codon acceleration).
* [ ] Publish `light-llm-agent` stack (Rust + Zig + Mojo + Codon + Lean4).

---

# 🚀 Closing

This project combines:

* **Mojo/Codon** → performance acceleration, hardware-native execution.
* **Zig** → bare-metal performance for core kernels.
* **Lean4** → formal proofs of correctness.
* **Rust** → ecosystem glue + async agent orchestration.

By stacking **quantization + distillation + runtime engineering**, we can achieve a **resource-minimal, formally safe LLM agent runtime**.

---
