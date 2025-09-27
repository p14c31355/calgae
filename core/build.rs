use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::fs;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=runtime/src/runtime.zig");
    println!("cargo:rerun-if-changed=ml/codon/kernel.codon");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // Compile Zig to shared library
    let status = Command::new("zig")
        .args(&["build-lib", "-O", "ReleaseSmall", "runtime/src/runtime.zig", "-dynamic", "--name", "zig_kernel"])
        .current_dir("runtime")
        .status();

    if let Ok(status) = status {
        if !status.success() {
            println!("cargo:warning=Zig compilation failed with status: {}", status);
        }
    } else {
        println!("cargo:warning=Fail to run zig build-lib, skipping Zig kernel");
    }

    // Assume libzig_kernel.so is generated in runtime/
    let zig_lib_source = PathBuf::from("runtime").join("libzig_kernel.so");
    if zig_lib_source.exists() {
        if let Err(e) = fs::copy(&zig_lib_source, out_dir.join("libzig_kernel.so")) {
            println!("cargo:warning=Failed to copy zig library: {}", e);
        }
    }

    // Compile Codon to shared library for acceleration
    let codon_status = Command::new("codon")
        .args(&["compile", "--embed-rt", "--opt-level=3", "ml/codon/kernel.codon", "-o", "libcodon_kernel.so", "-lshared"])
        .current_dir(".")
        .status();

    if let Ok(status) = codon_status {
        if status.success() {
            let codon_lib_source = PathBuf::from("libcodon_kernel.so");
            if codon_lib_source.exists() {
                if let Err(e) = fs::copy(&codon_lib_source, out_dir.join("libcodon_kernel.so")) {
                    println!("cargo:warning=Failed to copy codon library: {}", e);
                }
                println!("cargo:rustc-link-search=native={}", out_dir.display());
                println!("cargo:rustc-link-lib=codon_kernel");
            }
        } else {
            println!("cargo:warning=Codon compilation failed with status: {}", status);
        }
    } else {
        println!("cargo:warning=Failed to run codon compile, skipping Codon kernel");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=zig_kernel");
}
