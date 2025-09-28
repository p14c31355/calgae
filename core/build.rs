fn main() {
    // Link Mojo runtime for quantization kernels
    println!("cargo:rustc-link-lib=mojo_runtime");
    println!("cargo:rustc-link-search=native=/usr/local/lib");  // Adjust to actual Mojo lib path
    // Link other deps if needed
    println!("cargo:rerun-if-changed=build.rs");
}
