use calgae_core::{run_agent, cli::Args};
use clap::Parser;

const ALGAE_ART: &str = "\n                                 ,---.\n                                /   ,     ,-~^\"#\n                                \\   \\      /  \\\n                                 '---^--^    ^---^ \nCalgae: Lightweight LLM Runtime\n\"Write code for me\" and press Enter for assistance.\n";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("{}", ALGAE_ART);

    let args = Args::parse();
    run_agent(&args)
}
