use calgae_core::{run_agent, cli::Args};
use clap::Parser;
use std::io::{self, Write};
use std::result;

const ALGAE_ART: &str = r#"
                                 ,---.
                                /   ,     ,-~^"#
                                \   \      /  \
                                 `---^--^    ^---^ 
Calgae: Lightweight LLM Runtime
"Write code for me" and press Enter for assistance.
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("{}", ALGAE_ART);

    let args = Args::parse();
    run_agent(&args)
}
