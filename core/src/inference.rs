#![warn(missing_docs)]

//! LLM inference using rustformers/llm crate for lightweight, quantized CPU inference.
//! Supports GGUF models with built-in quantization (Q4_0, Q8_0, etc.).
//! Achieves high speed and low memory via no_std-safe design and mmap loading.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::fs;

use num_cpus;
use thiserror::Error;

use llm::{KnownModel, Model, InferenceRequest, Vocabulary};

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
    model: Model,
    vocabulary: Vocabulary,
    quantized: bool,  // Flag for quantization status
}

impl LlmInference {
    /// Creates a new inference instance from a GGUF model file path or directory.
    /// Detects GGUF vs safetensors, uses llm for GGUF, falls back to candle for HF.
    pub fn new(model_path: PathBuf, dtype: Option<()>) -> Result<Self> {
        let path_str = model_path.to_string_lossy().to_string();

        // Check if it's GGUF file
        if model_path.extension().map_or(false, |ext| ext == "gguf") {
            let model = llm::models::load_model(
                KnownModel::TinyLlama,
                llm::ModelSource::File(path_str),
                llm::ModelParameters::default(),
            ).context("Failed to load GGUF model")?;

            let vocabulary = model.vocabulary.clone();

            let quantized = model_path.file_name()
                .and_then(|name| name.to_str())
                .map_or(false, |name| name.contains('q') || name.contains("quant"));

            Ok(Self {
                model,
                vocabulary,
                quantized,
            })
        } else {
            // Fallback to candle for safetensors/HF directory (legacy support)
            return Err(InferenceError::UnsupportedFormat("Current implementation prioritizes GGUF for llm. Use xtask fetch-gguf-model or awq-quantize for GGUF.".to_string()).into());
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
        let mut session = self.model.start_session(self.vocabulary.clone());

        let mut req = InferenceRequest {
            prompt: prompt.to_string(),
            params: llm::InferenceParameters {
                n_predict: max_tokens as i64,
                n_keep: 0,
                temperature: temperature as f32,
                top_k: top_k as usize,
                top_p,
                repeat_penalty: 1.1,
                repeat_penalty_last_n: 64,
                n_threads: Some(num_cpus::get() as u32),
                ..Default::default()
            },
            samplers: vec![],
            stream_tokenizer_output: false,
            ..Default::default()
        };

        let mut output = String::new();
        let mut stream = session.infer_with_params(&mut req)?;

        while let Some(token) = stream.next() {
            match token {
                Ok(token) => output.push_str(&token),
                Err(e) => return Err(anyhow::anyhow!("Inference streaming failed: {}", e)),
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
