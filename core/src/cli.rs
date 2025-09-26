use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "calgae-agent", about = "CLI Coding Agent using TinyLlama LLM")]
pub struct Args {
    #[arg(short, long, default_value = "Write a simple Rust function to add two numbers")]
    pub prompt: String,

    #[arg(short, long, default_value = "50")]
    pub tokens: usize,

    #[arg(long, help = "Path to the llama-cli binary")]
    pub llama_bin: std::path::PathBuf,

    #[arg(long, help = "Path to the GGUF model file")]
    pub model: std::path::PathBuf,
}
