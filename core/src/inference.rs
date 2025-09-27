#![warn(missing_docs)]

//! Lightweight LLM inference using Candle for quantized CPU inference.
//! Supports phi-2 model from HuggingFace with safetensors weights.

use anyhow::{anyhow, Result};
use std::fs;
use serde_json::from_str;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use log::info;

use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::phi::{Config, Model as PhiModel};
use tokenizers::Tokenizer;
use thiserror::Error;
use rand::prelude::*;
use rand::rngs::StdRng;

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
    model: Arc<Mutex<PhiModel>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
}

impl LlmInference {
    /// Creates a new inference instance from a model directory containing safetensors files and tokenizer.json.
    /// Supports phi-2 model.
    pub fn new(model_path: PathBuf, dtype: Option<DType>) -> Result<Self> {
        let dtype = dtype.unwrap_or(DType::BF16);
        let device = Device::Cpu;

        if !device.is_cuda() {
            info!("Warning: CUDA is not available, this example runs on CPU");
        }

        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(InferenceError::TokenizerLoadError)?;

        let config_path = model_path.join("config.json");
        let config_str = fs::read_to_string(&config_path).map_err(|e| anyhow!("Failed to read config.json: {}", e))?;
        let config: Config = from_str(&config_str).map_err(|e| anyhow!("Failed to parse config.json: {}", e))?;

        let weights = fs::read_dir(&model_path)?
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().map_or(false, |s| s == "safetensors"))
            .map(|e| e.path())
            .collect::<Vec<_>>();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device) }?;

        let model = PhiModel::new(&config, vb)?;

        info!("Model loaded successfully, vocab size: {}", tokenizer.get_vocab(true).len());

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }

    /// Performs inference on the prompt, generating up to `max_tokens`.
    /// Uses top-k/top-p sampling with temperature.
    pub fn infer(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<String> {
        let encoding = self.tokenizer.encode(prompt, true).map_err(|e| anyhow!("Tokenizer encode error: {}", e))?;
        let prompt_len = encoding.get_ids().len();
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(42);
        let sampling = Sampling::TopKThenTopP {
            k: top_k,
            p: top_p as f64,
            temperature: temperature as f64,
        };
        let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

        for _ in 0..max_tokens {
            let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.lock().unwrap().forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

            let next_token = logits_processor.sample(&logits)?;

            tokens.push(next_token as u32);

            let end_token = self.tokenizer.token_to_id("<|endoftext|>").unwrap_or(50256);
            if next_token == end_token {
                break;
            }
        }

        if tokens.len() <= prompt_len {
            return Err(InferenceError::NoOutputError.into());
        }

        let generated_tokens = &tokens[prompt_len..];
        let response = self.tokenizer.decode(generated_tokens, true)
            .map_err(|e| anyhow!("Tokenizer decode error: {}", e))?;

        if response.trim().is_empty() {
            return Err(InferenceError::NoOutputError.into());
        }

        Ok(response.trim().to_string())
    }
}

impl Drop for LlmInference {
    fn drop(&mut self) {
        // Resources automatically cleaned
    }
}
