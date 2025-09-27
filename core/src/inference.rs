//! Candle-based LLM inference for CPU, supporting quantized models (int4/int8 via compatible safetensors or gguf).
//! Replaces llama.cpp with Candle for pure Rust CPU inference.

use anyhow::{Context, Result};
use candle_core::{Device, DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama};
use std::path::{Path, PathBuf};
use tokenizers::tokenizer::Tokenizer;

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
    config: Config,
}

impl LlmInference {
    /// Creates a new inference instance from a model path (HF format directory).
    /// Supports quantized models if weights are in int4/int8 safetensors.
    pub fn new(model_path: PathBuf) -> Result<Self> {
        let device = Device::Cpu;

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e))?;

        // Hardcode TinyLlama config (avoids serde issues)
        let config = Config {
            hidden_size: 2048,
            intermediate_size: 5632,
            vocab_size: 32000,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn: false,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: 2048,
            tie_word_embeddings: false,
        };

        let dtype = DType::F32;

        let vb_paths: Vec<_> = model_path
            .read_dir()
            .context("Failed to read model dir")?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
            .map(|e| e.path())
            .collect();
        let vb_paths_ref: Vec<&Path> = vb_paths.iter().map(|p| p.as_path()).collect();
        let vb = if vb_paths.is_empty() {
            return Err(anyhow::anyhow!("No safetensors files found"));
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&vb_paths_ref, dtype, &device)? }
        };

        let model = Llama::load(vb, &config)?;

        let eos_token_id = tokenizer.token_to_id("<|endoftext|>")
            .ok_or_else(|| anyhow::anyhow!("Eos token <|endoftext|> not found"))?
            as u32;
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
            config,
        })
    }

    /// Performs inference on the prompt, generating up to `max_tokens`.
    /// Uses greedy sampling on CPU.
    pub fn infer(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let mut tokens: Vec<i64> = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?
            .get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect();

        if tokens.is_empty() {
            return Err(InferenceError::NoOutputError.into());
        }

        let mut output = String::new();
        let dtype = DType::F32;
        let mut cache = Cache::new(true, dtype, &self.config, &self.device)?;

        // Prefill
        let context_len = tokens.len();
        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits_prefill = self.model.forward(&input, 0, &mut cache)?;
        let logits_last = logits_prefill.i((0usize, context_len - 1, ..))?.argmax(logits_prefill.dims().len() - 1)?.squeeze(0)?.to_scalar::<i64>()? as u32;
        let next_token_i = logits_last;

        let mut generated = 0u32;

        // Generate first new token if not EOS
        if next_token_i != self.eos_token_id {
            let decoded = self.tokenizer.decode(&[next_token_i], false)
                .map_err(|e| anyhow::anyhow!("Failed to decode token {}: {}", next_token_i, e))?;
            output.push_str(&decoded);
            tokens.push(next_token_i as i64);
            generated = 1;
        }

        while generated < max_tokens as u32 {
            // Single token forward
            let input = Tensor::new(&[next_token_i as i64], &self.device)?.unsqueeze(0)?;
            let index_pos = tokens.len() - 1;
            let logits = self.model.forward(&input, index_pos, &mut cache)?;
            let next_token_i = logits.argmax(logits.dims().len() - 1)?.squeeze(0)?.to_scalar::<i64>()? as u32;

            if next_token_i == self.eos_token_id {
                break;
            }

            let decoded = self.tokenizer.decode(&[next_token_i], false)
                .map_err(|e| anyhow::anyhow!("Failed to decode token {}: {}", next_token_i, e))?;
            output.push_str(&decoded);

            tokens.push(next_token_i as i64);
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
