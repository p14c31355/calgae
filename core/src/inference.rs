//! Candle-based LLM inference for CPU, supporting quantized models (int4/int8 via compatible safetensors or gguf).
//! Replaces llama.cpp with Candle for pure Rust CPU inference.

use anyhow::{Context, Result};
use candle_core::{Device, DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config, Model as Llama};
use hf_hub::Api;
use std::path::{Path, PathBuf};
use tokenizers::tokenizer::Tokenizer;

use serde_json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model load error: {0}")]
    ModelLoadError(#[from] anyhow::Error),
    #[error("No output generated")]
    NoOutputError,
    #[error("Quantization not supported for this model format")]
    QuantizationError,
}

pub struct LlmInference {
    model: Llama,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
    pad_token_id: Option<u32>,
}

impl LlmInference {
    /// Creates a new inference instance from a model path (HF format directory).
    /// Supports quantized models if weights are in int4/int8 safetensors.
    pub fn new(model_path: PathBuf) -> Result<Self> {
        let device = Device::Cpu;

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .with_context(|| format!("Failed to load tokenizer from {:?}", tokenizer_path))?;

        // Load config
        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config from {:?}", config_path))?;
        let config: Config = serde_json::from_str(&config_str)?;

        // Detect quantization (simple check; advanced inspection TODO)
        let is_quantized = model_path
            .read_dir()
            .context("Failed to read model directory")?
            .any(|entry| {
                if let Ok(entry) = entry {
                    if let Some(file_name) = entry.file_name().to_str() {
                        file_name.contains("int8") || file_name.contains("quantized")
                    } else {
                        false
                    }
                } else {
                    false
                }
            });
        let dtype = if is_quantized { DType::I8 } else { DType::F32 };

        let vb_paths: Vec<_> = model_path
            .read_dir()
            .context("Failed to read model dir")?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
            .map(|e| e.path())
            .collect();
        let vb = if vb_paths.is_empty() {
            return Err(InferenceError::ModelLoadError(anyhow::anyhow!("No safetensors files found")));
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&vb_paths, dtype, &device)? }
        };

        let model = Llama::load(&vb, &config)?;

        let eos_token_id = tokenizer.token_to_id("<|endoftext|>")? as u32;
        let pad_token_id = tokenizer
            .token_to_id("<pad>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .map(|id| id as u32);

        Ok(Self {
            model,
            tokenizer,
            device,
            eos_token_id,
            pad_token_id,
        })
    }

    /// Performs inference on the prompt, generating up to `max_tokens`.
    /// Uses greedy sampling on CPU.
    pub fn infer(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?
            .get_ids()
            .to_vec();

        let mut output = String::new();
        let mut generated = 0;

        while generated < max_tokens {
            let context_len = tokens.len();

            // Prepare input tensor (batch=1, seq=len)
            let input = Tensor::new(tokens.as_slice(), &self.device)?
                .unsqueeze(0)?
                .to_dtype(self.model.dtype())?;  // Match model dtype

            // Forward pass (simplified no cache; inefficient for long seq, TODO: add KV cache)
            let (logits, _) = self.model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;

            // Get logits for last position
            let pos = input.dim(1)? - 1;
            let logits_next = logits.i((.., pos))?.squeeze(1)?;

            // Greedy: argmax
            let next_token_tensor = logits_next.argmax(0)?;
            let next_token_i = u32::try_from(next_token_tensor.to_scalar::<u32>()?)?;

            // Check EOS
            if next_token_i == self.eos_token_id {
                break;
            }

            // Decode single token
            output.push_str(&self.tokenizer.decode(&[next_token_i as usize])?.text);

            tokens.push(next_token_i as usize);
            generated += 1;
        }

        if output.is_empty() {
            Err(InferenceError::NoOutputError.into())
        } else {
            Ok(output)
        }
    }
}

impl Drop for LlmInference {
    fn drop(&mut self) {
        // Candle tensors are dropped automatically
    }
}
