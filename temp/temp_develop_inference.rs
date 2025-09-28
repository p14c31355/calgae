#![warn(missing_docs)]

//! Lightweight LLM inference using Candle for safetensors weights.
//! Supports Llama-based models like TinyLlama.

use anyhow::{anyhow, Result};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config as LlamaConfig, Llama, Cache, LlamaEosToks};
use tokenizers::Tokenizer;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use log::info;
use std::fs;
use serde_json::Value;
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
    config: Arc<LlamaConfig>,
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
        let tokenizer = Arc::new(Tokenizer::from_file(&tokenizer_path).map_err(InferenceError::TokenizerLoadError)?);

        let config_path = model_path.join("config.json");
        let config_str = fs::read_to_string(&config_path).map_err(|e| anyhow!("Failed to read config.json: {}", e))?;
        let config_value: Value = serde_json::from_str(&config_str).map_err(|e| anyhow!("Failed to parse config.json: {}", e))?;

        let num_attention_heads = config_value["num_attention_heads"].as_u64().unwrap_or(32) as usize;
        let num_key_value_heads = config_value["num_key_value_heads"].as_u64().unwrap_or(num_attention_heads as u64) as usize;

        let config = LlamaConfig {
            hidden_size: config_value["hidden_size"].as_u64().unwrap_or(4096) as usize,
            intermediate_size: config_value["intermediate_size"].as_u64().unwrap_or(11008) as usize,
            vocab_size: config_value["vocab_size"].as_u64().unwrap_or(32000) as usize,
            num_hidden_layers: config_value["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
            num_attention_heads,
            num_key_value_heads,
            rms_norm_eps: config_value["rms_norm_eps"].as_f64().unwrap_or(1e-5),
            rope_theta: config_value["rope_theta"].as_f64().unwrap_or(500000.0) as f32,
            max_position_embeddings: config_value["max_position_embeddings"].as_u64().unwrap_or(2048) as usize,
            tie_word_embeddings: config_value["tie_word_embeddings"].as_bool().unwrap_or(false),
            bos_token_id: config_value["bos_token_id"].as_u64().map(|v| v as u32),
            eos_token_id: config_value["eos_token_id"].as_u64().map(|v| LlamaEosToks::Single(v as u32)),
            rope_scaling: None,
            use_flash_attn: false,
        };
        let config = Arc::new(config);

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

        let model = Arc::new(Llama::load(vb, &config)?);

        let eos_token_id = tokenizer.token_to_id("</s>")
            .ok_or_else(|| anyhow!("EOS token '</s>' not found in tokenizer vocab"))? as u32;

        let vocab = tokenizer.get_vocab(false);
        info!("Model loaded successfully, vocab size: {}", vocab.len());

        Ok(Self {
            model,
            tokenizer,
            config,
            device,
            eos_token_id,
        })
    }

    /// Performs efficient inference with KV cache: prefill prompt, then autoregressive decode.
    /// Supports greedy sampling. Based on CPU-efficient LLM inference techniques.
    pub fn infer(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<String> {
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow!("Tokenizer encode error: {}", e))?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = tokens.len();

        if prompt_len == 0 {
            return Err(InferenceError::NoOutputError.into());
        }

        let mut cache = Cache::new(true, DType::F32, &self.config, &self.device)?;
        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0, &mut cache)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

        let mut generated_tokens = vec![];
        let last_pos = Tensor::new(&[(prompt_len - 1) as i64], &self.device)?;
        let next_token_logits = logits.index_select(&last_pos, 0usize)?;
        let next_token = next_token_logits.argmax(1)?.to_scalar::<u32>()?;
        generated_tokens.push(next_token);
        tokens.push(next_token);

        if next_token == self.eos_token_id {
            let generated_str = self.tokenizer.decode(&generated_tokens, false).map_err(|e| anyhow!("Decode error: {}", e))?;
            return Ok(generated_str.trim().to_string());
        }

        let mut position = prompt_len;

        for _ in 1..max_tokens {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

            let logits = self.model.forward(&input, position, &mut cache)?;
            let next_token_logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            let next_token = next_token_logits.argmax(0)?.to_scalar::<u32>()?;
            generated_tokens.push(next_token);
            tokens.push(next_token);
            position += 1;

            if next_token == self.eos_token_id {
                break;
            }
        }

        let response = self.tokenizer.decode(&generated_tokens, false)
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
