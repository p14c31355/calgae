use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "calgae-agent", about = "CLI Coding Agent using TinyLlama LLM")]
pub struct Args {
    #[arg(short, long, default_value = "Write a simple Rust function to add two numbers")]
    pub prompt: String,

    #[arg(short, long, default_value = "50")]
    pub tokens: usize,

    #[arg(long, default_value = "../../engine/build/bin/llama-cli")]
    pub llama_bin: String,

    #[arg(long, default_value = "../../models/TinyLlama-1.1B-q4_0.gguf")]
    pub model: String,
}

pub fn run_cli() {
    let args = Args::parse();
    println!("Prompt: {}", args.prompt);
    println!("Generating {} tokens...", args.tokens);
    // Call LlmInference::infer(&args)
    // For now, stub
    println!("Agent response stub: Code generation using LLM...");
}
