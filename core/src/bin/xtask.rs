use anyhow::Result;
use clap::{Parser, Subcommand};
use std::env;
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
    Setup {
        /// Skip system package updates
        #[arg(long, default_value_t = false)]
        no_apt: bool,
    },

    /// Build all components (Zig runtime, Lean proofs, etc.)
    BuildAll,

    /// Fetch models from HuggingFace
    FetchModel,

    /// Run AWQ quantization on TinyLlama using llm-compressor
    AwqQuantize,
}

async fn is_command_available(cmd: &str) -> Result<bool> {
    let output = AsyncCommand::new("which")
        .arg(cmd)
        .output()
        .await?;
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

            if codon_status.is_ok() && codon_status.as_ref().unwrap().success() && mojo_status.is_ok() && mojo_status.as_ref().unwrap().success() {
                println!("All preparations completed.");
            } else {
                anyhow::bail!("Some preparation tasks failed.");
            }

            Ok(())
        },

        Commands::Setup { no_apt } => {
            println!("Setting up Calgae dependencies...");

            if !no_apt {
                println!("Updating system packages...");
                let apt_update = AsyncCommand::new("sudo")
                    .args(["apt", "update"])
                    .status()
                    .await?;
                if !apt_update.success() {
                    eprintln!("Warning: apt update failed. Continuing...");
                }

                println!("Installing base dependencies...");
                let apt_install = AsyncCommand::new("sudo")
                    .args(["apt", "install", "-y", "build-essential", "cmake", "git", "curl", "wget", "python3", "python3-pip"])
                    .status()
                    .await?;
                if !apt_install.success() {
                    eprintln!("Warning: Base dependencies install failed. Continuing...");
                }
            }

            // Rust setup
            println!("Setting up Rust...");
            if !is_command_available("cargo").await? {
                println!("Installing Rust...");
                let rustup = AsyncCommand::new("bash")
                    .args(["-c", "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"])
                    .status()
                    .await?;
                if rustup.success() {
                    println!("Rust installed. Run 'source $HOME/.cargo/env' in your shell.");
                } else {
                    eprintln!("Warning: Rust installation failed.");
                }
            }

            // Zig setup
            println!("Setting up Zig...");
            if !is_command_available("zig").await? {
                let zig_version = "0.13.0";
                let zig_url = format!("https://ziglang.org/download/{}/zig-linux-x86_64-{}.tar.xz", zig_version, zig_version);
                println!("Installing Zig...");
                let wget_zig = AsyncCommand::new("wget")
                    .arg(&zig_url)
                    .status()
                    .await?;
                if wget_zig.success() {
                    let tar_zig = AsyncCommand::new("tar")
                        .args(["-xvf", format!("zig-linux-x86_64-{}.tar.xz", zig_version)])
                        .status()
                    .await?;
                    if tar_zig.success() {
                        let mv_zig = AsyncCommand::new("sudo")
                            .args(["mv", format!("zig-linux-x86_64-{}", zig_version), "/usr/local/"])
                            .status()
                            .await?;
                        if mv_zig.success() {
                            println!("Zig installed to /usr/local. Add /usr/local/zig-linux-x86_64-{} to PATH.", zig_version);
                        }
                    }
                    let _ = AsyncCommand::new("rm")
                        .arg(format!("zig-linux-x86_64-{}.tar.xz", zig_version))
                        .status()
                        .await?;
                }
            }

            // Lean4 setup
            println!("Setting up Lean4...");
            if !is_command_available("lake").await? {
                println!("Installing Lean4...");
                let elan = AsyncCommand::new("bash")
                    .args(["-c", "curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh"])
                    .status()
                    .await?;
                if elan.success() {
                    println!("Lean4 installed. Run 'source ~/.elan/env'.");
                }
            }

            // Mojo setup
            println!("Setting up Mojo...");
            if !is_command_available("mojo").await? {
                println!("Installing Mojo...");
                let modular = AsyncCommand::new("bash")
                    .args(["-c", "curl https://get.modular.com | sh -s -- -y"])
                    .status()
                    .await?;
                if modular.success() {
                    let activate_install = AsyncCommand::new("bash")
                        .args(["-c", "source ~/.modular/bin/activate && modular install mojo"])
                        .status()
                        .await?;
                    if activate_install.success() {
                        println!("Mojo installed. Run 'source ~/.modular/bin/activate'.");
                    }
                }
            }

            // Codon setup
            println!("Setting up Codon...");
            let codon_install = AsyncCommand::new("pip3")
                .args(["install", "codon"])
                .status()
                .await?;
            if !codon_install.success() {
                eprintln!("Warning: Codon installation failed.");
            }

            // Engine setup
            println!("Setting up llama.cpp engine...");
            let submodule = AsyncCommand::new("git")
                .args(["submodule", "update", "--init", "--recursive"])
                .status()
                .await?;
            if submodule.success() {
                let engine_build = AsyncCommand::new("bash")
                    .args(["-c", "cd engine && make clean && make -j$(nproc)"])
                    .status()
                    .await?;
                if !engine_build.success() {
                    eprintln!("Warning: engine build failed.");
                }
            }

            println!("Setup complete! Source env files as needed.");
            Ok(())
        },

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
        },

        Commands::FetchModel => {
            println!("Fetching TinyLlama model...");

            let model_url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
            let model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";

            fs::create_dir_all("models")?;

            if !Path::new(model_path).exists() {
                println!("Downloading model...");
                let wget = AsyncCommand::new("wget")
                    .args(["-O", model_path, model_url])
                    .status()
                    .await?;
                if !wget.success() {
                    anyhow::bail!("Download failed.");
                }
                println!("Model downloaded to {}.", model_path);
            } else {
                println!("Model already exists.");
            }

            Ok(())
        },

        Commands::AwqQuantize => {
            println!("Running AWQ quantization...");

            let compressor_dir = "llm-compressor";
            if !Path::new(compressor_dir).exists() {
                let clone = AsyncCommand::new("git")
                    .args(["clone", "https://github.com/vllm-project/llm-compressor.git"])
                    .status()
                    .await?;
                if !clone.success() {
                    anyhow::bail!("Clone failed.");
                }
            }

            let setup = AsyncCommand::new("bash")
                .args(["-c", format!("cd {} && [ -d venv ] || python3 -m venv venv && source venv/bin/activate && pip install -e .", compressor_dir)])
                .status()
                .await?;
            if !setup.success() {
                eprintln!("Warning: Setup failed.");
            }

            let awq = AsyncCommand::new("bash")
                .args(["-c", format!("cd {} && source venv/bin/activate && python -m llmcompressor.awq --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --q_bits 4 --q_group_size 128 --dump_awq tinyllama_awq.pt", compressor_dir)])
                .status()
                .await?;
            if awq.success() {
                println!("AWQ completed. Output in {}.", compressor_dir);
            } else {
                anyhow::bail!("AWQ failed.");
            }

            Ok(())
        },
    }
}
