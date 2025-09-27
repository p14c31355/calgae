// Build script for Calgae: Model fetching and preparation moved to xtask for explicit control.
// No implicit downloads during compilation.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
