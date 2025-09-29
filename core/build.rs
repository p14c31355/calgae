use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::fs;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=runtime/src/runtime.zig");
    println!("cargo:rerun-if-changed=ml/codon/kernel.codon");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR environment variable not set by Cargo"));
    
    println!("cargo:rustc-link-search=native={}", out_dir.display());

    // Link Mojo runtime for quantization kernels
    println!("cargo:rustc-link-lib=mojo_runtime");
    println!("cargo:rustc-link-search=native={}", env::var("MOJO_LIB_PATH").unwrap_or_else(|_| "/usr/local/lib".to_string()));  // Set MOJO_LIB_PATH to your Mojo lib directory
}
