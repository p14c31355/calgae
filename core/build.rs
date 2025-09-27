use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Link Zig library
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let zig_lib = out_dir.join("libzig_kernel.so");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=zig_kernel");

    // Compile Zig during build using cc
    println!("cargo:rerun-if-changed=runtime/src/runtime.zig");
    cc::Build::new()
        .file("runtime/src/runtime.zig")
        .compile("zig_kernel");
}
