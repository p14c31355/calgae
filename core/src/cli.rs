use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "calgae",
    about = "CLI Coding Agent using Candle LLM on CPU"
)]
pub struct Args {
    #[arg(
        short,
        long,
        default_value = ""
    )]
    pub prompt: String,

    #[arg(short, long, default_value = "128")]
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
        default_value = "models/tinyllama/TinyLlama-1.1B-Chat-v1.0",
        help = "Path to the model directory (HF format)",
        value_name = "PATH"
    )]
    pub model: std::path::PathBuf,
    #[arg(short, long, default_value_t = false, help = "Execute generated code and collect results")]
    pub execute: bool,

    #[arg(
        short = 'i',
        long,
        default_value_t = false,
        help = "Interactive mode for multi-turn coding assistance"
    )]
    pub interactive: bool,

    #[arg(short, long, default_value_t = false, help = "Quantize model weights using Mojo")]
    pub quantize: bool,

    #[arg(long, default_value = "awq", help = "Quantization mode: awq or smoothquant")]
    pub quantize_mode: String,

    #[arg(long, default_value = "0.1", help = "Top-k percent for AWQ")]
    pub top_k_p: f32,

    #[arg(short, long, default_value = "0.2", help = "Sparsity for SmoothQuant")]
    pub sparsity: f32,

    // #[arg(short = 'p', long = "parallel", default_value_t = 1, help = "Number of parallel prompts (for multi-task)")]
    // pub parallel: usize, // TODO: Implement multi-turning
}
