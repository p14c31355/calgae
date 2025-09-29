use calgae::agent::run_agent;
use calgae::cli::Args;
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use anyhow::{anyhow, Result};

const ALGAE_ART: &str = "\n                                 ,---.\n                                /   ,     ,-~^\"#\n                                \\   \\      /  \\\n                                 '---^--^    ^---^ \nCalgae: Lightweight LLM Runtime\n\"Write code for me\" and press Enter for assistance.\n";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("{}", ALGAE_ART);

    let mut args = Args::parse();

    if args.prompt.is_empty() {
        args.interactive = true;
    }

    if !args.model.exists() {
        eprintln!(
            "Warning: Model file not found at {:?}. Run `cargo run --bin xtask -- fetch-model` to download it.",
            args.model
        );
    }

    if args.quantize {
        let model_path = &args.model;
        let weights: Vec<PathBuf> = fs::read_dir(model_path)?
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().map_or(false, |s| s == "safetensors"))
            .map(|e| e.path())
            .collect();
        for weight_path in weights {
            let output = std::process::Command::new("mojo")
                .args(&["build", "ml/mojo/awq.mojo", "-o", "awq_bin"])
                .output()
                .map_err(|e| anyhow!("Mojo build failed: {}", e))?;
            if !output.status.success() {
                eprintln!("Mojo build failed: {}", String::from_utf8_lossy(&output.stderr));
                continue;
            }
            let mut cmd = std::process::Command::new("./awq_bin");
            cmd.arg(args.quantize_mode.clone());
            if let Some(path_str) = weight_path.to_str() {
                cmd.arg(path_str);
            } else {
                eprintln!("Warning: skipping non-UTF8 path: {}", weight_path.display());
                continue;
            }
            if args.quantize_mode == "awq" {
                cmd.arg(args.top_k_p.to_string());
            } else if args.quantize_mode == "smoothquant" {
                cmd.arg(args.sparsity.to_string());
            }
            let output = cmd.output()
                .map_err(|e| anyhow!("Mojo quantization failed: {}", e))?;
            if output.status.success() {
                println!("Quantization completed for {}", weight_path.display());
            } else {
                eprintln!("Quantization failed: {}", String::from_utf8_lossy(&output.stderr));
            }
        }
        println!("Quantization finished. Now run the agent.");
        return Ok(());
    }

    run_agent(args.model.clone(), args.prompt.clone(), args.tokens, args.temperature, args.top_k, args.top_p, args.execute, args.interactive).await?;
    Ok(())
}
