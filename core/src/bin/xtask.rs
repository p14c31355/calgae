use anyhow::Result;
use clap::{Parser, Subcommand};
use std::process::ExitStatus;
use tokio::process::Command as AsyncCommand;
use tokio::join;

#[derive(Parser)]
#[command(name = "xtask", about = "Task runner for Calgae")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Prepare models and artifacts (quantumization, optimization)
    Prepare,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Prepare => {
            println!("Preparing artifacts...");

            // Spawn async commands for parallel execution
            let awq_handle = tokio::spawn(async {
                println!("Running AWQ quantization...");
                let status = AsyncCommand::new("python")
                    .arg("scripts/awq_tinyllama.py")
                    .status()
                    .await;
                match status {
                    Ok(s) if s.success() => println!("AWQ quantization completed."),
                    Ok(_) => eprintln!("Warning: AWQ quantization failed."),
                    Err(e) => eprintln!("Failed to run AWQ: {}", e),
                }
                s
            });

            let codon_handle = tokio::spawn(async {
                println!("Running Codon optimization...");
                let status = AsyncCommand::new("codon")
                    .args(["run", "ml/codon/optimize.py", "-o", "models"])
                    .status()
                    .await;
                match status {
                    Ok(s) if s.success() => println!("Codon optimization completed."),
                    Ok(_) => eprintln!("Warning: Codon optimization failed."),
                    Err(e) => eprintln!("Failed to run Codon: {}", e),
                }
                s
            });

            let mojo_handle = tokio::spawn(async {
                println!("Building Mojo kernels...");
                let status = AsyncCommand::new("mojo")
                    .args(["build", "ml/mojo/kernels.mojo", "-o", "models/kernels"])
                    .status()
                    .await;
                match status {
                    Ok(s) if s.success() => println!("Mojo build completed."),
                    Ok(_) => eprintln!("Warning: Mojo build failed."),
                    Err(e) => eprintln!("Failed to run Mojo: {}", e),
                }
                s
            });

            // Wait for all to complete
            let (awq, codon, mojo) = join!(awq_handle, codon_handle, mojo_handle);
            let awq_status = awq??;
            let codon_status = codon??;
            let mojo_status = mojo??;

            if awq_status.success() && codon_status.success() && mojo_status.success() {
                println!("All preparations completed.");
            } else {
                anyhow::bail!("Some preparation tasks failed.");
            }

            Ok(())
        }
    }
}
