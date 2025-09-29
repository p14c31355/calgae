use std::env;
use std::path::PathBuf;

fn main() {
    // Build Mojo library
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mojo_out = out_dir.join("awq");
    let status = std::process::Command::new("mojo")
        .args(&[
            "build",
            "../ml/mojo/awq.mojo",
            "-o", &mojo_out.to_string_lossy(),
            "--emit", "shared-lib",
            "-O3"
        ])
        .status()
        .expect("Failed to build Mojo lib");

    if !status.success() {
        panic!("Mojo build failed");
    }

    // Link the generated lib (assuming it produces libawq.a or similar)
    let lib_filename = if cfg!(target_os = "windows") {
        "awq.dll"
    } else if cfg!(target_os = "macos") {
        "libawq.dylib"
    } else {
        "libawq.so"
    };
    let lib_path = mojo_out.join(lib_filename);
    if lib_path.exists() {
        println!("cargo:rustc-link-search=native={}", mojo_out.display());
        println!("cargo:rustc-link-lib=dylib=awq");
    } else {
        panic!("Mojo lib not found at expected path: {}", lib_path.display());
    }

    // Ensure cc dependency is used if needed
    println!("cargo:rerun-if-changed=../ml/mojo/awq.mojo");
}
