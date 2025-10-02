# Reading List: Efficient LLM on CPU (Quantization & Inference)

## Core References

- [Efficient LLM Inference on CPUs](https://arxiv.org/abs/2311.00502)  
  Techniques for W4/INT4 quantization and optimized threading on CPUs. Directly relevant for CPU-only design.

- [Evaluating Quantized Large Language Models](https://arxiv.org/abs/2402.18158)  
  Systematic evaluation of weight, activation, and KV cache quantization across models.

- [Extreme Compression of Large Language Models via Additive Quantization (AQLM)](https://arxiv.org/html/2401.06118v2)  
  Pushing ultra-low-bit quantization (2–3 bits). Useful for understanding accuracy/compression trade-offs.

- [NoMAD-Attention: Efficient LLM Inference on CPU](https://arxiv.org/abs/2403.01273)  
  Attention redesign for CPUs via in-register lookup. Keeps model architecture intact while optimizing runtime.

- [Pushing the Limits of Large Language Model Quantization](https://arxiv.org/abs/2411.17525)  
  Exploration of quantization lower bounds and practical limits.

- [Highly Optimized Kernels and Fine-Grained Codebooks for LLM Inference on CPU](https://arxiv.org/abs/2501.00032)  
  Optimized kernels for 4-bit group-wise quantization, improving prefill/decode throughput by 3x or more. Practical implementation-focused paper to maximize CPU potential.

- [SAIL: SRAM-Accelerated LLM Inference System with Lookup-Table Quantization](https://arxiv.org/abs/2509.25853)  
  Accelerates arbitrary-bit quantization on CPU using SRAM for lookup tables. Hardware-oriented approach evolving in-register lookups from NoMAD-Attention (2403.01273).

---

## Foundational Techniques

- [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs](https://arxiv.org/abs/2211.10438)  
  Absorbs activation outliers into weights for stable 8-bit quantization.

- [What Makes Quantization for Large Language Models Hard](https://arxiv.org/abs/2403.06408)  
  Identifies challenges like long-tail distributions and activation scale variability.

- [ABQ-LLM: Arbitrary-Bit Quantized Inference Acceleration](https://arxiv.org/abs/2408.08554)  
  Framework for arbitrary-bit inference acceleration. Relevant for mixed precision strategies.

- [GuidedQuant: Large Language Model Quantization via Exploiting Gradient Information](https://arxiv.org/abs/2505.07004)  
  Uses gradient information to guide quantization, improving accuracy.

- [A Systems Approach to Advancing Low-Bit LLM Quantization](https://arxiv.org/abs/2412.20185)  
  System-wide approach to low-bit quantization, improving quality via CPU memory utilization. Updates challenge analysis from 2403.06408 with 2025 state-of-the-art limits.

---

## CPU-centric Optimizations

- [Separating Prefill-Decode Compilation for Efficient CPU](https://arxiv.org/abs/2507.18454)  
  Splits prefill and decode phases for specialized compilation and optimization.

- [When CPUs Outperform for On-Device LLM Inference](https://arxiv.org/abs/2505.06461)  
  Demonstrates cases where CPUs can outperform GPUs in constrained environments.

- [On-Device Qwen2.5: Efficient LLM Inference on Embedded / Edge](https://arxiv.org/abs/2504.17376)  
  Design strategies for running LLMs on CPU and edge devices.

- [T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Inference](https://arxiv.org/abs/2407.00088)  
  LUT-based inference for low-bit (2-4bit) LLMs dramatically improves CPU efficiency. Complements on-device comparisons in 2505.06461 with concrete lookup implementations.

- [Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Systems](https://arxiv.org/abs/2504.11750)  
  Analysis and optimization of LLM inference on CPU-GPU hybrid systems. Deepens prefill-decode separation from 2507.18454 with real-system perspectives like PCIe/GH200.

---

## Training & QAT (Quantization-Aware Training)

- [Direct Quantized Training of Language Models with Stochastic Rounding](https://arxiv.org/html/2412.04787v1)  
  Direct training with quantized weights using stochastic rounding.

- [Efficient Quantization-Aware Training for Large Language Models](https://arxiv.org/abs/2407.11062)  
  Block-wise training and joint learning of quantization parameters.

- [LRQ: Optimizing Post-Training Quantization for Large Models](https://arxiv.org/abs/2407.11534)  
  Layer reconstruction techniques to preserve quality after quantization.
