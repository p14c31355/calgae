use crate::agent::run_agent;
use crate::cli::Args;
use clap::Parser;

const ALGAE_ART: &str = "\n                                 ,---.\n                                /   ,     ,-~^\"#\n                                \\   \\      /  \\\n                                 '---^--^    ^---^ \nCalgae: Lightweight LLM Runtime\n\"Write code for me\" and press Enter for assistance.\n";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("{}", ALGAE_ART);

    let args = Args::parse();

    if !args.model.exists() {
        eprintln!("Warning: Model file not found at {:?}. Run `cargo run --bin xtask prepare` to generate artifacts first.", args.model);
    }

    run_agent(args.model.clone(), args.prompt).await?;
    Ok(())
}
