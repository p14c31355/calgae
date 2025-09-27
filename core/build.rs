use std::env;
use std::path::PathBuf;

fn main() {
    // Build llama.cpp using cmake
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_dir = out_dir.join("llama_build");
    let dst = cmake::Config::new("../engine")
        .define("LLAMA_BUILD_SHARED", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("GGML_NATIVE", "ON")
        .define("GGML_BLAS", "OFF")  // Keep minimal, no external BLAS
        .profile("Release")
        .out_dir(&build_dir)
        .build();

    // Link the static libraries
    println!("cargo:rustc-link-search={}/lib64", dst.display());  // llama.cpp uses lib64 for some builds
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");

    // System libs for Linux
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=m");  // math lib
        println!("cargo:rustc-link-lib=pthread");
    }

    // Generate bindings with bindgen
    let bindings = bindgen::Builder::default()
        .header("../engine/llama.h")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_.*")
        .clang_arg("-I").clang_arg("../engine/")
        .clang_arg("-I").clang_arg(dst.join("include").to_str().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // Re-run build script if headers change
    println!("cargo:rerun-if-changed=../engine/");
    println!("cargo:rerun-if-changed=../engine/llama.h");
}
