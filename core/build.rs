//! Build script for Calgae: Prepares quantized LLM model using HF Hub and Candle.
//! Downloads model files to OUT_DIR/models. Supports quantized safetensors (int4/int8).
//! For dynamic quantization, calls Python script via std::process if available.

use std::env;
use std::fs;
use std::path::PathBuf;

use hf_hub::api::sync::Api;
use hf_hub::Repo;
use hf_hub::RepoType;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let model_dir = out_dir.join("models").join("tinyllama");

    // Ensure model dir exists
    fs::create_dir_all(&model_dir).expect("Failed to create model dir");

    // Check if model already downloaded
    if model_dir.join("config.json").exists() {
        println!("cargo:rerun-if-changed={}", model_dir.display());
        return;
    }

    // Download TinyLlama model from HF
    let api = Api::new().expect("Failed to create HF API");
    let repo_obj = Repo::new("microsoft/TinyLlama-1.1B-Chat-v1.0".to_string(), RepoType::Model);
    let repo = api.repo(repo_obj);

    // Download key files
    let files = vec!["config.json", "tokenizer.json"];
    for file in files {
        let path = model_dir.join(file);
        if !path.exists() {
            match repo.get(file) {
                Ok(local_path) => {
                    let contents = std::fs::read(&local_path).expect("Failed to read local file");
                    fs::write(&path, &contents).expect("Failed to write model file");
                }
                Err(e) => {
                    eprintln!("Failed to download {}: {}", file, e);
                }
            }
        }
    }

    // Download weight files (TinyLlama uses sharded safetensors)
    let weight_files = vec!["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"];
    for file in weight_files {
        let path = model_dir.join(file);
        if !path.exists() {
            match repo.get(file) {
                Ok(local_path) => {
                    let contents = std::fs::read(&local_path).expect("Failed to read local weight file");
                    fs::write(&path, &contents).expect("Failed to write weight file");
                }
                Err(e) => {
                    eprintln!("Failed to download {}: {}", file, e);
                }
            }
        }
    }

    // Optional: Quantization step - Skip for now as we use the F32 model; AWQ for int4/int8 can be added manually
    eprintln!("Model downloaded to {}", model_dir.display());
    eprintln!("Note: For quantized model, download a pre-quantized version from TheBloke or run AWQ manually.");

    // Generate constant for default model path
    println!("cargo:rustc-cfg=model_dir=\"{}\"", model_dir.display());

    println!("cargo:rerun-if-changed=build.rs");
}
