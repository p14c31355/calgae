use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::fs;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=runtime/src/runtime.zig");

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
        fs::copy(&zig_lib_source, out_dir.join(&zig_lib_source)).ok();
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=zig_kernel");
}
