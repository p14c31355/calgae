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

---

## CPU-centric Optimizations

- [Separating Prefill-Decode Compilation for Efficient CPU](https://arxiv.org/abs/2507.18454)  
  Splits prefill and decode phases for specialized compilation and optimization.

- [When CPUs Outperform for On-Device LLM Inference](https://arxiv.org/abs/2505.06461)  
  Demonstrates cases where CPUs can outperform GPUs in constrained environments.

- [On-Device Qwen2.5: Efficient LLM Inference on Embedded / Edge](https://arxiv.org/abs/2504.17376)  
  Design strategies for running LLMs on CPU and edge devices.

---

## Training & QAT (Quantization-Aware Training)

- [Direct Quantized Training of Language Models with Stochastic Rounding](https://arxiv.org/html/2412.04787v1)  
  Direct training with quantized weights using stochastic rounding.

- [Efficient Quantization-Aware Training for Large Language Models](https://arxiv.org/abs/2407.11062)  
  Block-wise training and joint learning of quantization parameters.

- [LRQ: Optimizing Post-Training Quantization for Large Models](https://arxiv.org/abs/2407.11534)  
  Layer reconstruction techniques to preserve quality after quantization.
