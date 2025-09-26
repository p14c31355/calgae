agent-root/
├── core/               # Rust (safety-and-high-speed-core)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── inference.rs   # llama.cpp binding (ffi or cxx)
│
├── runtime/            # Zig layer (FFI / OS )
│   └── src/
│       └── runtime.zig
│
├── proof/              # Lean4: math proof experimental
│   ├── lakefile.lean
│   └── src/
│       └── Correctness.lean
│
├── ml/                 # Mojo / Codon: calculate and machine-learning experimental
│   ├── mojo/
│   │   └── kernels.mojo   # HPC based calc kernel
│   └── codon/
│       └── optimize.py    # LLVM tuning calc by codon
│
├── engine/             # llama.cpp (submodule or clone)
│   └── ...
│
├── models/             # model (quantitization OK)
│   ├── TinyLlama-1.1B-q4_0.gguf
│   ├── Phi-2-q4_K_M.gguf
│   └── ...
│
├── scripts/            # builder and runner
│   ├── build_all.sh
│   └── run_cpu_llm.sh
│
└── README.md
