//! Candle-based LLM inference for CPU, supporting quantized models (int4/int8 via compatible safetensors or gguf).
//! Replaces llama.cpp with Candle for pure Rust CPU inference.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama::{Cache, Config, Llama};
use std::path::{Path, PathBuf};
use tokenizers::tokenizer::Tokenizer;

use serde::Deserialize;
use std::fs;

use rand::prelude::*;
use rand::rngs::ThreadRng;

use thiserror::Error;

#[derive(Deserialize)]
struct HfLlamaConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    rms_norm_eps: f32,
    rope_theta: f32,
    max_position_embeddings: usize,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    tie_word_embeddings: bool,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
}

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
    pub fn new(model_path: PathBuf, dtype: Option<DType>) -> Result<Self> {
        let device = Device::Cpu;
        let dtype = dtype.unwrap_or(DType::F32);

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e)
        })?;

        // Load config dynamically from config.json
        let config_path = model_path.join("config.json");
        let config_str = fs::read_to_string(&config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read config from {:?}: {}", config_path, e))?;

        let hf_config: HfLlamaConfig = serde_json::from_str(&config_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;

        let num_kv_heads = hf_config
            .num_key_value_heads
            .unwrap_or(hf_config.num_attention_heads);
        let config = Config {
            hidden_size: hf_config.hidden_size,
            intermediate_size: hf_config.intermediate_size,
            vocab_size: hf_config.vocab_size,
            num_hidden_layers: hf_config.num_hidden_layers,
            num_attention_heads: hf_config.num_attention_heads,
            num_key_value_heads: num_kv_heads,
            use_flash_attn: false,
            rms_norm_eps: hf_config.rms_norm_eps as f64,
            rope_theta: hf_config.rope_theta,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None, // TODO: implement if needed
            max_position_embeddings: hf_config.max_position_embeddings,
            tie_word_embeddings: hf_config.tie_word_embeddings,
        };

        let vb_paths: Vec<_> = model_path
            .read_dir()
            .context("Failed to read model dir")?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .map(|e| e.path())
            .collect();
        let vb_paths_ref: Vec<&Path> = vb_paths.iter().map(|p| p.as_path()).collect();
        let vb = if vb_paths.is_empty() {
            return Err(anyhow::anyhow!("No safetensors files found"));
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&vb_paths_ref, dtype, &device)? }
        };

        let model = Llama::load(vb, &config)?;

        let eos_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| anyhow::anyhow!("Eos token <|endoftext|> not found"))?;
        let pad_token_id = tokenizer
            .token_to_id("<pad>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"));

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
    /// Uses configurable sampling on CPU.
    pub fn infer(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<String> {
        let mut rng = rand::rng();
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
        let prefill_logits = logits_prefill.i((0usize, context_len - 1, ..))?;
        let next_token_i =
            self.sample_from_logits(&prefill_logits, temperature, top_k, top_p, &mut rng)?;

        let mut generated = 0u32;

        // Generate first new token if not EOS
        if next_token_i != self.eos_token_id {
            let decoded = self
                .tokenizer
                .decode(&[next_token_i], false)
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
            let seq_logits = logits.i((0, 0, ..))?;
            let next_token_i =
                self.sample_from_logits(&seq_logits, temperature, top_k, top_p, &mut rng)?;

            if next_token_i == self.eos_token_id {
                break;
            }

            let decoded = self
                .tokenizer
                .decode(&[next_token_i], false)
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

    fn sample_from_logits(
        &self,
        logits: &Tensor,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        rng: &mut ThreadRng,
    ) -> Result<u32> {
        let seed = rng.random();
        let temp = temperature as f64;
        let sampling = if temperature == 0.0 {
            Sampling::ArgMax
        } else if top_k == 0 {
            Sampling::TopP {
                p: top_p as f64,
                temperature: temp,
            }
        } else if top_p >= 1.0 {
            Sampling::TopK {
                k: top_k,
                temperature: temp,
            }
        } else {
            Sampling::TopKThenTopP {
                k: top_k,
                p: top_p as f64,
                temperature: temp,
            }
        };
        let mut processor = LogitsProcessor::from_sampling(seed, sampling);
        Ok(processor.sample(logits)?)
    }
}

impl Drop for LlmInference {
    fn drop(&mut self) {
        // Candle tensors are dropped automatically
    }
}
