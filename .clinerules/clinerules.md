# .clinerules
# Project: Lightweight, Safe, High-Speed LLM Runtime Quality Gate Rules

[General]
# All core inference kernels (Rust, Zig, Mojo) must pass near-100% test coverage.
RULE_ID: KERNEL_COVERAGE
LANGUAGES: Rust, Zig, Mojo
SEVERITY: ERROR
CHECK: Coverage > 99.0%
MESSAGE: Core kernels are performance/safety critical and must be fully covered.

[Rust]
# Enforce a minimum level of lints for memory safety and clean code.
RULE_ID: RUST_SAFETY_LINT
CHECK: rustc --cap-lints warn -D warnings
FLAGS: -D unsafe_code, -D unused_results, -D non_snake_case, -D clippy::pedantic
MESSAGE: Must pass all safety-related lints. Unsafe blocks require thorough review and justification comments.

[Zig]
# Ensure Zig code is memory-safe and avoids hidden allocations/control flow.
RULE_ID: ZIG_ALLOCATION_CHECK
CHECK: Explicit allocators (e.g., Arena, FixedBuffer) must be used in all performance-critical contexts.
TAGS: performance, kernel
MESSAGE: Default or hidden allocations are prohibited in runtime kernels.

[Mojo]
# Check for proper use of Mojo's vectorization capabilities for maximum speed.
RULE_ID: MOJO_VECTOR_USE
CHECK: Vectorized operations must utilize SIMD types (`Vector[DType, N]`) and fast C APIs where applicable.
TAGS: performance, SIMD
MESSAGE: Must optimize for vectorization to maximize hardware utilization. Non-vectorized core loops will fail.

[Lean4]
# Formal verification: ensure proofs are complete and trusted code is verified.
RULE_ID: LEAN_PROOF_COMPLETENESS
CHECK: All definitions tagged `@[verified]` must have associated formal proofs (e.g., .olean file generated).
SEVERITY: ERROR
MESSAGE: Formal verification of quantization and core numerical routines is mandatory before merge.

[Quantization_Benchmarking]
# Enforce acceptable quality/performance trade-off using EdgeProfiler results.
RULE_ID: QUANT_ACCURACY_CHECK
TOOL: EdgeProfiler
CHECK: perplexity_increase < 1.05 * baseline_perplexity AND latency_decrease > 4.5x
MESSAGE: Quantization must yield a significant speedup (target 4.5x+) while maintaining accuracy (perplexity increase < 5%).