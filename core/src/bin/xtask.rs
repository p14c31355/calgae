use anyhow::Result;
use clap::{Parser, Subcommand};
use std::fs;
use std::path::Path;
use tokio::join;
use tokio::process::Command as AsyncCommand;

#[derive(Parser)]
#[command(name = "xtask", about = "Task runner for Calgae")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Prepare models and artifacts (optimization, kernel build)
    Prepare,

    /// Setup all dependencies (Rust, Zig, Lean, Mojo, Codon, etc.)
    Setup,

    /// Build all components (Zig runtime, Lean proofs, etc.)
    BuildAll,

    /// Fetch safetensors model from HuggingFace
    FetchModel,

    /// Fetch quantized GGUF model for llm inference
    FetchGGUF,

    /// Run AWQ quantization on TinyLlama using llm-compressor
    AwqQuantize,
}

async fn is_command_available(cmd: &str) -> Result<bool> {
    let output = AsyncCommand::new("which").arg(cmd).output().await?;
    Ok(output.status.success())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Prepare => {
            println!("Preparing artifacts (Codon, Mojo)...");

            let codon_handle = tokio::spawn(async {
                println!("Running Codon optimization...");
                let status = AsyncCommand::new("codon")
                    .args(["run", "ml/codon/optimize.py", "-o", "models"])
                    .status()
                    .await;
                if let Ok(s) = status {
                    if s.success() {
                        println!("Codon optimization completed.");
                    } else {
                        eprintln!("Warning: Codon optimization failed.");
                    }
                } else {
                    eprintln!("Failed to run Codon optimization.");
                }
                status
            });

            let mojo_handle = tokio::spawn(async {
                println!("Building Mojo kernels...");
                let status = AsyncCommand::new("mojo")
                    .args(["build", "ml/mojo/kernels.mojo", "-o", "models/kernels"])
                    .status()
                    .await;
                if let Ok(s) = status {
                    if s.success() {
                        println!("Mojo build completed.");
                    } else {
                        eprintln!("Warning: Mojo build failed.");
                    }
                } else {
                    eprintln!("Failed to run Mojo build.");
                }
                status
            });

            let (codon, mojo) = join!(codon_handle, mojo_handle);
            let codon_status = codon??;
            let mojo_status = mojo??;

            if codon_status.success() && mojo_status.success() {
                println!("All preparations completed.");
            } else {
                anyhow::bail!("Some preparation tasks failed.");
            }

            Ok(())
        }

        Commands::Setup => {
            println!("Setting up Calgae dependencies...");

            // Check base system dependencies
            let base_deps = [
                "build-essential",
                "cmake",
                "git",
                "curl",
                "wget",
                "python3",
                "python3-pip",
            ];
            for &dep in &base_deps {
                if dep == "build-essential" {
                    // Check if gcc is available as proxy for build-essential
                    if !is_command_available("gcc").await? {
                        eprintln!(
                            "Warning: GCC (build-essential) not found. Please install: sudo apt install build-essential"
                        );
                    }
                } else {
                    let cmd = match dep {
                        "cmake" => "cmake",
                        "git" => "git",
                        "curl" => "curl",
                        "wget" => "wget",
                        "python3" => "python3",
                        "python3-pip" => "pip3",
                        _ => continue,
                    };
                    if !is_command_available(cmd).await? {
                        eprintln!(
                            "Warning: {} not found. Please install: sudo apt install {}",
                            cmd, dep
                        );
                    }
                }
            }

            // Rust setup
            println!("Checking Rust...");
            if !is_command_available("cargo").await? {
                println!(
                    "Rust not found. Install via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
                );
                println!("Then add to PATH: source $HOME/.cargo/env");
            } else {
                println!("Rust is available.");
            }

            // Zig setup
            println!("Checking Zig...");
            if !is_command_available("zig").await? {
                let zig_version = "0.13.0";
                let zig_url = format!(
                    "https://ziglang.org/download/{}/zig-linux-x86_64-{}.tar.xz",
                    zig_version, zig_version
                );
                println!("Zig not found. Download and install manually:");
                println!("  wget {}", zig_url);
                println!("  tar -xvf zig-linux-x86_64-{}.tar.xz", zig_version);
                println!("  sudo mv zig-linux-x86_64-{} /usr/local/", zig_version);
                println!("  Add /usr/local/zig-linux-x86_64-{} to PATH.", zig_version);
            } else {
                println!("Zig is available.");
            }

            // Lean4 setup
            println!("Checking Lean4...");
            if !is_command_available("lake").await? {
                println!("Lean4 not found. Install via Elan:");
                println!(
                    "  curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh"
                );
                println!("Then add to PATH: source ~/.elan/env");
            } else {
                println!("Lean4 is available.");
            }

            // Mojo setup
            println!("Checking Mojo...");
            if !is_command_available("mojo").await? {
                println!("Mojo not found. Install via Modular:");
                println!("  curl https://get.modular.com | sh -s -- -y");
                println!("  source ~/.modular/bin/activate");
                println!("  modular install mojo");
                println!("Then run 'source ~/.modular/bin/activate' to use.");
            } else {
                println!("Mojo is available.");
            }

            // Codon setup
            println!("Setting up Codon...");
            let codon_install = AsyncCommand::new("pip3")
                .args(["install", "codon"])
                .status()
                .await?;
            if !codon_install.success() {
                eprintln!("Warning: Codon installation failed. Run 'pip3 install codon' manually.");
            }

            println!("Setup complete! Source env files as needed.");
            Ok(())
        }

        Commands::BuildAll => {
            println!("Building all Calgae components...");

            let zig_build = AsyncCommand::new("bash")
                .args(["-c", "cd runtime && zig build-exe src/runtime.zig"])
                .status()
                .await?;
            if !zig_build.success() {
                eprintln!("Warning: Zig build failed.");
            }

            let lean_build = AsyncCommand::new("bash")
                .args(["-c", "cd proof && lake build"])
                .status()
                .await?;
            if !lean_build.success() {
                eprintln!("Warning: Lean build failed.");
            }

            println!("Build complete! Rust and engine via cargo.");
            Ok(())
        }

        Commands::FetchModel => {
            println!("Fetching TinyLlama model in safetensors format...");

            let model_dir = Path::new("models/tinyllama");
            fs::create_dir_all(model_dir)?;

            let files = vec![
                (
                    "config.json",
                    "https://huggingface.co/microsoft/TinyLlama-1.1B-Chat-v1.0/resolve/main/config.json",
                ),
                (
                    "tokenizer.json",
                    "https://huggingface.co/microsoft/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json",
                ),
                (
                    "model.safetensors",
                    "https://huggingface.co/microsoft/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors",
                ),
            ];

            let mut all_success = true;
            for (filename, url) in files {
                let file_path = model_dir.join(filename);
                if !file_path.exists() {
                    println!("Downloading {}...", filename);
                    let wget = AsyncCommand::new("wget")
                        .arg("-O")
                        .arg(&file_path)
                        .arg(url)
                        .status()
                        .await?;
                    if !wget.success() {
                        eprintln!("Failed to download {} from {}.", filename, url);
                        all_success = false;
                    } else {
                        println!("Downloaded {} to {}.", filename, file_path.display());
                    }
                } else {
                    println!("{} already exists at {}.", filename, file_path.display());
                }
            }

            if all_success {
                println!(
                    "TinyLlama model downloaded successfully to {}.",
                    model_dir.display()
                );
                Ok(())
            } else {
                anyhow::bail!("One or more files failed to download.");
            }
        }

        Commands::AwqQuantize => {
            println!("Running AWQ quantization...");

            // Install autoawq if not present
            let install = AsyncCommand::new("pip3")
                .args(["install", "autoawq", "accelerate", "transformers"])
                .status()
                .await?;
            if !install.success() {
                eprintln!("Warning: Failed to install autoawq dependencies. Install manually: pip3 install autoawq accelerate transformers");
            }

            let awq = AsyncCommand::new("python3")
                .args(["llm-compressor/awq.py"])
                .current_dir(".")
                .status()
                .await?;
            if awq.success() {
                println!("AWQ quantization completed. Quantized model saved to ./models/tinyllama-awq");
                // Copy to models dir if needed
                let copy = AsyncCommand::new("cp")
                    .args(["-r", "./models/tinyllama-awq", "./models/"])
                    .status()
                    .await?;
                if copy.success() {
                    println!("Copied quantized model to models/");
                }
            } else {
                anyhow::bail!("AWQ quantization failed. Check python3 and dependencies.");
            }

            Ok(())
        }

        Commands::FetchGGUF => {
            println!("Fetching GGUF model...");
            // TODO: Implement GGUF fetch similar to FetchModel
            println!("Placeholder: Fetch GGUF from HuggingFace.");
            Ok(())
        }
    }
}
