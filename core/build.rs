use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::fs;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=runtime/src/runtime.zig");
    println!("cargo:rerun-if-changed=ml/codon/kernel.codon");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    println!("cargo:rustc-link-search=native={}", out_dir.display());

    // Link Mojo runtime for quantization kernels
    println!("cargo:rustc-link-lib=mojo_runtime");
    println!("cargo:rustc-link-search=native=/usr/local/lib");  // Adjust to actual Mojo lib path
}
