#![warn(missing_docs)]

//! Lightweight LLM inference using Candle for safetensors weights.
//! Supports Llama-based models like TinyLlama.

use anyhow::{anyhow, Result};

use candle_core::{Device, DType, Tensor};

use candle_nn::VarBuilder;

use candle_transformers::models::llama::{Config as LlamaConfig, Model as Llama};

use candle_transformers::models::hf::HfLlamaConfig;

use tokenizers::Tokenizer;

use std::path::{Path, PathBuf};

use std::sync::Arc;

use log::info;

use std::fs;

use serde_json::from_str;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model load error: {0}")]
    ModelLoadError(#[from] anyhow::Error),
    #[error("Tokenizer load error: {0}")]
    TokenizerLoadError(#[from] tokenizers::Error),
    #[error("No output generated")]
    NoOutputError,
    #[error("Unsupported model format: {0}")]
    UnsupportedFormat(String),
}

pub struct LlmInference {
    model: Arc<Llama>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    eos_token_id: u32,
}

impl LlmInference {
    /// Creates a new inference instance from a model directory containing safetensors files and tokenizer.json.
    /// Supports Llama-based models like TinyLlama.
    pub fn new(model_path: PathBuf, dtype: Option<DType>) -> Result<Self> {
        let dtype = dtype.unwrap_or(DType::F32);
        let device = Device::Cpu;

        if !device.is_cuda() {
            info!("Warning: CUDA is not available, this example runs on CPU");
        }

        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(InferenceError::TokenizerLoadError)?;

        let config_path = model_path.join("config.json");
        let config_str = fs::read_to_string(&config_path).map_err(|e| anyhow!("Failed to read config.json: {}", e))?;
        let hf_config: HfLlamaConfig = from_str(&config_str).map_err(|e| anyhow!("Failed to parse config.json: {}", e))?;

        let num_kv_heads = hf_config.num_key_value_heads.unwrap_or(hf_config.num_attention_heads);
        let config = LlamaConfig {
            hidden_size: hf_config.hidden_size,
            intermediate_size: hf_config.intermediate_size,
            vocab_size: hf_config.vocab_size,
            num_hidden_layers: hf_config.num_hidden_layers,
            num_attention_heads: hf_config.num_attention_heads,
            num_key_value_heads: num_kv_heads,
            rms_norm_eps: hf_config.rms_norm_eps as f64,
            rope_theta: hf_config.rope_theta,
            max_position_embeddings: hf_config.max_position_embeddings,
            tie_word_embeddings: !hf_config.tie_word_embeddings,
        };

        let weights: Vec<PathBuf> = fs::read_dir(&model_path)?
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().map_or(false, |s| s == "safetensors"))
            .map(|e| e.path())
            .collect();
        if weights.is_empty() {
            return Err(anyhow::anyhow!("No safetensors files found"));
        }
        let weights_ref: Vec<&Path> = weights.iter().map(|p| p.as_path()).collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_ref, dtype, &device)? };

        let model = Llama::load(&vb, &config)?;

        let eos_token_id = tokenizer.token_to_id("</s>")
            .ok_or_else(|| anyhow!("EOS token '</s>' not found in tokenizer vocab"))? as u32;

        info!("Model loaded successfully, vocab size: {}", tokenizer.vocab_len());

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            device,
            eos_token_id,
        })
    }

    /// Performs inference on the prompt, generating up to `max_tokens`.
    /// Uses greedy sampling for simplicity.
    pub fn infer(
        &self,
        prompt: &str,
        max_tokens: usize,
        _temperature: f32,
        _top_k: usize,
        _top_p: f32,
    ) -> Result<String> {
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow!("Tokenizer encode error: {}", e))?;
        let prompt_len = encoding.get_ids().len();
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

        for _ in 0..max_tokens {
            let seq_len = tokens.len();
            let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let position_ids = Tensor::arange(0i64, seq_len as i64, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, &position_ids)?;
            let full_logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let our_logits = full_logits.i((seq_len as i64 - 1) as i64)?.squeeze(0)?;
            // Greedy sampling
            let next_token = our_logits.argmax(DType::U32, &self.device)?.to_scalar::<u32>()?;

            tokens.push(next_token);

            if next_token == self.eos_token_id {
                break;
            }
        }

        if tokens.len() <= prompt_len {
            return Err(InferenceError::NoOutputError.into());
        }

        let generated_tokens = &tokens[prompt_len..];
        let response = self.tokenizer.decode(generated_tokens, true)
            .map_err(|e| anyhow!("Tokenizer decode error: {}", e))?;

        let response = response.trim().to_string();
        if response.is_empty() {
            return Err(InferenceError::NoOutputError.into());
        }

        Ok(response)
    }
}

impl Drop for LlmInference {
    fn drop(&mut self) {
        // Resources automatically cleaned
    }
}
