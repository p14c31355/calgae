use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let zig_root = manifest_dir.join("../runtime/zig");

    // Ensure target dir for Zig build
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Re-enable the Zig build process
    let target = std::env::var("TARGET").unwrap();
    let zig_target = target.replace("-unknown", ""); // Remove -unknown for Zig compatibility

    let status_zig = std::process::Command::new("zig")
        .args(&["build", &format!("-Dtarget={}", zig_target), "-Doptimize=ReleaseSafe"])
        .current_dir(&zig_root)
        .status()
        .expect("Failed to build Zig libs");
    if !status_zig.success() {
        panic!("Zig build failed");
    }

    // Link Zig libs (assuming zig-out/lib has lib*.a)
    println!("cargo:rustc-link-search=native={}", zig_root.join("zig-out/lib").display());
    println!("cargo:rustc-link-lib=static=runtime");
    println!("cargo:rustc-link-lib=static=quantizer");

    // Build Mojo AWQ library
    let mojo_awq_out = out_dir.join("awq");
    let mojo_awq_path = manifest_dir.join("../ml/mojo/awq.mojo");
    let status_mojo_awq = std::process::Command::new("mojo")
        .args(&[
            "build",
            &mojo_awq_path.to_string_lossy(),
            "-o", &mojo_awq_out.to_string_lossy(),
            "--emit", "shared-lib",
            "-O3"
        ])
        .status()
        .expect("Failed to build Mojo AWQ lib");

    if !status_mojo_awq.success() {
        panic!("Mojo AWQ build failed");
    }

    let lib_filename = if cfg!(target_os = "windows") {
        "awq.dll"
    } else if cfg!(target_os = "macos") {
        "libawq.dylib"
    } else {
        "libawq.so"
    };
    let lib_awq_path = mojo_awq_out.join(lib_filename);
    if lib_awq_path.exists() {
        println!("cargo:rustc-link-search=native={}", mojo_awq_out.display());
        println!("cargo:rustc-link-lib=dylib=awq");
    } else {
        panic!("Mojo AWQ lib not found at expected path: {}", lib_awq_path.display());
    }

    // Rerun if changed
    println!("cargo:rerun-if-changed=../core/src/");
    println!("cargo:rerun-if-changed=../ml/mojo/");
    println!("cargo:rerun-if-changed=../runtime/zig/src/");
}
