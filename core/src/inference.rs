#![warn(missing_docs)]

//! LLM inference using rustformers/llm crate for lightweight, quantized CPU inference.
//! Supports GGUF models with built-in quantization (Q4_0, Q8_0, etc.).
//! Achieves high speed and low memory via no_std-safe design and mmap loading.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use num_cpus;
use thiserror::Error;

use llm::KnownModel;

/// Custom errors for inference
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model load error: {0}")]
    ModelLoadError(#[from] anyhow::Error),
    #[error("No output generated")]
    NoOutputError,
    #[error("Unsupported model format: {0}")]
    UnsupportedFormat(String),
}

/// Struct for lightweight LLM inference using llm crate
pub struct LlmInference {
    model: llm::Model,
    quantized: bool,  // Flag for quantization status
}

impl LlmInference {
    /// Creates a new inference instance from a GGUF model file path.
    /// Supports quantized models via GGUF format (e.g., tinyllama-q4.gguf).
    pub fn new(model_path: PathBuf, _dtype: Option<()>) -> Result<Self> {
        let path_str = model_path.to_string_lossy().to_string();

        // Check if it's GGUF file
        if model_path.extension().map_or(false, |ext| ext == "gguf") {
            let model = llm::load_from_file(KnownModel::TinyLlama, &path_str)
                .context("Failed to load GGUF model")?;

            let quantized = model_path.file_name()
                .and_then(|name| name.to_str())
                .map_or(false, |name| name.contains('q') || name.contains("quant"));

            Ok(Self {
                model,
                quantized,
            })
        } else {
            return Err(InferenceError::UnsupportedFormat("Only GGUF format supported. Use xtask fetch-gguf".to_string()).into());
        }
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
        let session = self.model.start_session(Default::default());

        let mut req = llm::InferenceRequest::new();
        req.prompt = prompt.to_string();
        req.params = llm::InferenceParameters {
            n_predict: max_tokens as i32,
            n_threads: Some(num_cpus::get() as u32),
            temperature: temperature as f32,
            top_k: top_k as usize,
            top_p: top_p as f32,
            repeat_penalty: 1.1,
            ..Default::default()
        };

        let mut output = String::new();
        let mut stream = session.infer(req);

        while let Some(token) = stream.next() {
            match token {
                Ok(token) => output.push_str(&token),
                Err(_) => break,
            }
        }

        if output.trim().is_empty() {
            return Err(InferenceError::NoOutputError.into());
        }

        Ok(output.trim().to_string())
    }
}

impl Drop for LlmInference {
    fn drop(&mut self) {
        // Resources automatically cleaned
    }
}
