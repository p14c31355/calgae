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

use rand::Rng;

use std::fs;

use serde_json::from_str;

use thiserror::Error;
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

use rand::Rng;

use std::fs;

use serde_json::from_str;

use thiserror::Error;

#[derive(Error, Debug)]
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
            tie_word_embeddings: hf_config.tie_word_embeddings,
            use_flash_attn: false,
            bos_token_id: Some(hf_config.bos_token_id as u32),
            eos_token_id: Some(hf_config.eos_token_id as u32),
            rope_scaling: hf_config.rope_scaling.map(|s| candle_transformers::models::llama::RopeScaling { factor: s.factor, r#type: s.r#type }),
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
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = prompt_tokens.len();

        if prompt_len == 0 {
            return Err(InferenceError::NoOutputError.into());
        }

        // Prefill: Compute KV cache for prompt
        let prompt_input = Tensor::new(&prompt_tokens, &self.device)?.unsqueeze(0)?;
        let prompt_pos_ids = Tensor::arange(0i64, prompt_len as i64, &self.device)?.unsqueeze(0)?;
        let (prompt_logits, kv_cache) = self.model.forward_with_cache(&prompt_input, &prompt_pos_ids)?;

        let mut generated_tokens = vec![];
        let mut current_cache = kv_cache;
        let mut current_pos = prompt_len as i64;

        for _ in 0..max_tokens {
            // Decode: Single token input (last token)
            let last_token = Tensor::new(&[prompt_tokens.last().unwrap()], &self.device)?.unsqueeze(0)?;
            let pos_id = Tensor::new(&[current_pos], &self.device)?.unsqueeze(0)?;
            let (logits, new_cache) = self.model.forward_with_cache(&last_token, &pos_id, Some(&current_cache))?;
            current_cache = new_cache;

            let logits_f32 = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logits_f32.argmax(DType::U32, &self.device)?.to_scalar::<u32>()?;
            generated_tokens.push(next_token);
            current_pos += 1;

            if next_token == self.eos_token_id {
                break;
            }
        }

        let response = self.tokenizer.decode(&generated_tokens, true)
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
            tie_word_embeddings: hf_config.tie_word_embeddings,
            use_flash_attn: false,
            bos_token_id: Some(hf_config.bos_token_id as u32),
            eos_token_id: Some(hf_config.eos_token_id as u32),
            rope_scaling: hf_config.rope_scaling.map(|s| candle_transformers::models::llama::RopeScaling { factor: s.factor, r#type: s.r#type }),
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
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = prompt_tokens.len();

        if prompt_len == 0 {
            return Err(InferenceError::NoOutputError.into());
        }

        // Prefill: Compute KV cache for prompt
        let prompt_input = Tensor::new(&prompt_tokens, &self.device)?.unsqueeze(0)?;
        let prompt_pos_ids = Tensor::arange(0i64, prompt_len as i64, &self.device)?.unsqueeze(0)?;
        let (prompt_logits, kv_cache) = self.model.forward_with_cache(&prompt_input, &prompt_pos_ids)?;

        let mut generated_tokens = vec![];
        let mut current_cache = kv_cache;
        let mut current_pos = prompt_len as i64;

        for _ in 0..max_tokens {
            // Decode: Single token input (last token)
            let last_token = Tensor::new(&[prompt_tokens.last().unwrap()], &self.device)?.unsqueeze(0)?;
            let pos_id = Tensor::new(&[current_pos], &self.device)?.unsqueeze(0)?;
            let (logits, new_cache) = self.model.forward_with_cache(&last_token, &pos_id, Some(&current_cache))?;
            current_cache = new_cache;

            let logits_f32 = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logits_f32.argmax(DType::U32, &self.device)?.to_scalar::<u32>()?;
            generated_tokens.push(next_token);
            current_pos += 1;

            if next_token == self.eos_token_id {
                break;
            }
        }

        let response = self.tokenizer.decode(&generated_tokens, true)
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
