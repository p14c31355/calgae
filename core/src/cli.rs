use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "calgae-agent",
    about = "CLI Coding Agent using Candle LLM on CPU"
)]
pub struct Args {
    #[arg(
        short,
        long,
        default_value = "Write a simple Rust function to add two numbers"
    )]
    pub prompt: String,

    #[arg(short, long, default_value = "512")]
    pub tokens: usize,

    #[arg(short = 't', long, default_value = "0.7")]
    pub temperature: f32,

    #[arg(short, long, default_value = "50")]
    pub top_k: usize,

    #[arg(short, long, default_value = "0.95")]
    pub top_p: f32,

    #[arg(
        short,
        long,
        help = "Path to the model directory (HF format)",
        value_name = "PATH"
    )]
    pub model: std::path::PathBuf,
    // #[arg(short, long, default_value_t = false, help = "Execute generated code and collect results")]
    // pub execute: bool,

    // #[arg(short = 'p', long = "parallel", default_value_t = 1, help = "Number of parallel prompts (for multi-task)")]
    // pub parallel: usize, // TODO: Implement multi-tasking
}
