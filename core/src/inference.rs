#![warn(missing_docs)]

//! Lightweight LLM inference using Candle for quantized CPU inference.
//! Supports phi-2 model from HuggingFace with safetensors weights.

use anyhow::{Result};
use std::path::PathBuf;
use std::sync::Arc;
use log::info;

use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, SamplingMode, self};
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
    model: Arc<PhiModel>,
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

        let config = Config::phi_2();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path.join("model.safetensors")], dtype, &device)
            .map_err(anyhow::Error::msg)? };

        let model = PhiModel::new(&config, vb)?;

        info!("Model loaded successfully");

        Ok(Self {
            model: Arc::new(model),
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
        let mut tokens = self.tokenizer.encode(prompt, true)?.get_ids().to_vec();

        let mut logits_processor = LogitsProcessor::new(42u64, temperature.max(1e-8).into());

        let mut generated = 0;
        let seed = 42;
        let mut rng = StdRng::seed_from_u64(seed as u64);

        let vocab_size = self.tokenizer.get_vocab(true).len();

        while generated < max_tokens {
            let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, 0i32)?;  // use 0 for causal mask offset
            let logits = logits.squeeze(0)?.contiguous()?.to_dtype(DType::F32)?;

            let logits_ref = logits.as_ref();
            let next_token = logits_processor.sample_token(
                logits_ref,
                0,
                top_k,
                top_p,
                None,
                &mut rng,
                vocab_size,
            )?;

            tokens.push(next_token);

            let eos_token_id = self.tokenizer.token_to_id("<|endoftext|>") .unwrap_or(0);
            if next_token == eos_token_id {
                break;
            }

            generated += 1;
        }

        let response = self.tokenizer.decode(&tokens, true)? .trim().to_string();

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
