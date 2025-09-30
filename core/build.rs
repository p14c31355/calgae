use std::env;
use std::path::PathBuf;

fn main() {
    // Ensure target dir for Zig build
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Re-enable the Zig build process
    let status_zig = std::process::Command::new("zig")
        .args(&["build", "-Dtarget=x86_64-linux-gnu", "-Doptimize=ReleaseSafe"])
        .current_dir("../runtime/zig") // Corrected path to zig build directory
        .status()
        .expect("Failed to build Zig libs");
    if !status_zig.success() {
        panic!("Zig build failed");
    }

    // Link Zig libs (assuming zig-out/lib has lib*.a)
    println!("cargo:rustc-link-search=native=../runtime/zig/zig-out/lib");
    println!("cargo:rustc-link-lib=static=runtime");
    println!("cargo:rustc-link-lib=static=quantizer");

    // Build Mojo AWQ library
    let mojo_out = out_dir.join("awq");
    let status_mojo = std::process::Command::new("mojo")
        .args(&[
            "build",
            "../ml/mojo/awq.mojo",
            "-o", &mojo_out.to_string_lossy(),
            "--emit", "shared-lib",
            "-O3"
        ])
        .status()
        .expect("Failed to build Mojo lib");

    if !status_mojo.success() {
        panic!("Mojo build failed");
    }

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

    // Rerun if changed
    println!("cargo:rerun-if-changed=../core/src/");
    println!("cargo:rerun-if-changed=../ml/mojo/");
}
