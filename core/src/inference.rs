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
use log::{info, error};
use std::fs;
use std::ffi::CString;
use std::os::raw::c_char;
use serde_json::Value;
use thiserror::Error;

use rand::prelude::*;

use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;
use foreign_types::ForeignTypeRef;

use candle_core::safetensors::MmapedSafetensors;

extern "C" {
    fn zig_quantize_buffer(input: *const f32, num: i32, bits: u8, output: *mut u8, scale: *mut f32) -> i32;
}

extern "C" {
    fn zig_quantize_model(model_path: *const c_char, bits: i32, output_path: *const c_char) -> i32;
}

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
    dtype: DType,
    eos_token_id: u32,
}

impl LlmInference {
    /// Creates a new inference instance from a model directory containing safetensors files and tokenizer.json.
    /// Supports Llama-based models like TinyLlama.
    /// Optionally quantizes the model using Zig quantizer before loading.
    pub fn new(model_path: PathBuf, dtype: Option<DType>, quantize_bits: Option<u8>) -> Result<Self> {
        let dtype = dtype.unwrap_or(DType::F16);
        let device = Device::Cpu;
        let original_dir = model_path.clone();

        // Optional quantization using Zig
        if let Some(bits) = quantize_bits {
            let safetensors_files: Vec<PathBuf> = fs::read_dir(model_path)?
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().map_or(false, |s| s == "safetensors"))
                .map(|e| e.path())
                .collect();
            if safetensors_files.is_empty() {
                return Err(anyhow!("No safetensors files found in model directory"));
            }
            let quantized_dir = model_path.join(format!("quantized_{}", bits));
            fs::create_dir_all(&quantized_dir)?;
            let mut quantize_success = true;
            for file in safetensors_files {
                let file_name = file.file_name().ok_or(anyhow!("Invalid file name"))?.to_str().ok_or(anyhow!("Invalid UTF-8 file name"))?;
                let quantized_file = quantized_dir.join(format!("{}_{}.bin", file_name.strip_suffix(".safetensors").unwrap_or(file_name), bits));
                let file_path_str = file.to_str().ok_or(anyhow!("Invalid file path"))?;
                let output_path_str = quantized_file.to_str().ok_or(anyhow!("Invalid output path"))?;
                let c_path = CString::new(file_path_str)?;
                let c_output = CString::new(output_path_str)?;
                let res = unsafe { zig_quantize_model(c_path.as_ptr(), bits as i32, c_output.as_ptr()) };
                if res < 0 {
                    quantize_success = false;
                    error!("Zig quantization failed for {}", file.display());
                } else {
                    info!("File {} quantized to {} bits at {}", file.display(), bits, output_path_str);
                }
            }
            if !quantize_success {
                return Err(anyhow!("Zig quantization failed for one or more files"));
            }
            model_path = quantized_dir; // Use quantized directory
        }

        if !device.is_cuda() {
            info!("Warning: CUDA is not available, this example runs on CPU");
        }

        let tokenizer_path = original_dir.join("tokenizer.json");
        let tokenizer = Arc::new(Tokenizer::from_file(&tokenizer_path).map_err(InferenceError::TokenizerLoadError)?);

        let config_path = original_dir.join("config.json");
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

        let weights: Vec<PathBuf> = if quantize_bits.is_some() {
            fs::read_dir(&model_path)?
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().map_or(false, |s| s == "bin"))
                .map(|e| e.path())
                .collect()
        } else {
            fs::read_dir(&original_dir)?
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().map_or(false, |s| s == "safetensors"))
                .map(|e| e.path())
                .collect()
        };
        if weights.is_empty() {
            return Err(anyhow::anyhow!(if quantize_bits.is_some() { "No quantized bin files found" } else { "No safetensors files found" }));
        }


        let weights_ref: Vec<&Path> = weights.iter().map(|p| p.as_path()).collect();

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_ref, dtype, &device) }?;

        // TODO: Implement custom loader for quantized .bin files (i8 tensors)
        // For now, use safetensors loader; will fail for quantized
        let model = Arc::new(Llama::load(&mut *vb, &config)?);

        let eos_token_id = tokenizer.token_to_id("</s>")
            .ok_or_else(|| anyhow!("EOS token '</s>' not found in tokenizer vocab"))? as u32;

        let vocab = tokenizer.get_vocab(false);
        info!("Model loaded successfully, vocab size: {}", vocab.len());

        Ok(Self {
            model,
            tokenizer,
            config,
            device,
            dtype,
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
        let start = std::time::Instant::now();
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow!("Tokenizer encode error: {}", e))?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = tokens.len();

        if prompt_len == 0 {
            return Err(InferenceError::NoOutputError.into());
        }

        let mut cache = Cache::new(true, self.dtype, &self.config, &self.device)?; // Use model's dtype
        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0, &mut cache)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

        let mut generated_tokens = vec![];
        let vocab_size = self.config.vocab_size;
        let last_pos = Tensor::new(&[(prompt_len - 1) as i64], &self.device)?;
        let next_token_logits = logits.index_select(&last_pos, 0usize)?.squeeze(0)?.to_dtype(DType::F32)?;
        let next_token = Self::sample_token(&next_token_logits, temperature, top_k, top_p, vocab_size, &self.device)?;
        generated_tokens.push(next_token);
        tokens.push(next_token);

        if next_token == self.eos_token_id {
            let generated_str = self.tokenizer.decode(&generated_tokens, false).map_err(|e| anyhow!("Decode error: {}", e))?;
            println!("Inference time: {:?} for EOS early stop", start.elapsed());
            return Ok(generated_str.trim().to_string());
        }

        let mut position = prompt_len;

        for _ in 1..max_tokens {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

            let logits = self.model.forward(&input, position, &mut cache)?;
            let next_token_logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            let next_token = Self::sample_token(&next_token_logits, temperature, top_k, top_p, vocab_size, &self.device)?;
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
        println!("Inference time: {:?}", start.elapsed());

        Ok(response)
    }

    fn sample_token(
        logits: &Tensor,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        vocab_size: usize,
        device: &Device,
    ) -> Result<u32> {
        let mut logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        if logits_vec.len() != vocab_size {
            return Err(anyhow!("Logits size mismatch"));
        }
        if temperature <= 0.0 {
            let max_idx = logits_vec.iter().enumerate().fold((0, f32::NEG_INFINITY), |acc, (i, &val)| {
                if val > acc.1 { (i, val) } else { acc }
            }).0;
            return Ok(max_idx as u32);
        }

        // Scale logits
        for logit in &mut logits_vec {
            *logit /= temperature;
        }

        let mut filtered_logits = logits_vec.clone();

        // Top-k filtering
        if top_k > 0 && top_k < vocab_size {
            let mut heap: BinaryHeap<Reverse<(f32, usize)>> = BinaryHeap::new();
            for (i, &v) in logits_vec.iter().enumerate() {
                heap.push(Reverse((v, i)));
                if heap.len() > top_k {
                    heap.pop();
                }
            }
            let mut top_k_indices: Vec<usize> = heap.into_iter().map(|Reverse((_, i))| i).collect();
            top_k_indices.sort(); // Optional, for deterministic order
            let top_k_set: HashSet<usize> = top_k_indices.into_iter().collect();
            for i in 0..vocab_size {
                if !top_k_set.contains(&i) {
                    filtered_logits[i] = f32::NEG_INFINITY;
                }
            }
        }

        // Top-p filtering
        if top_p < 1.0 {
            let max_l = *filtered_logits.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(&0.0);
            let exp_logits: Vec<f32> = filtered_logits.iter().map(|l| (*l - max_l).exp()).collect();
            let sum_exp = exp_logits.iter().sum::<f32>();
            let probs: Vec<f32> = exp_logits.iter().map(|e| *e / sum_exp).collect();
            let mut indexed_probs: Vec<(f32, usize)> = probs.iter().enumerate().map(|(i, &v)| (v, i)).collect();
            indexed_probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)); // descending
            let mut cumsum = 0.0;
            let mut cutoff = 0;
            for (prob, _) in indexed_probs.iter() {
                cumsum += *prob;
                cutoff += 1;
                if cumsum > top_p {
                    break;
                }
            }
            let nucleus_indices: Vec<usize> = indexed_probs[..cutoff].iter().map(|(_, i)| *i).collect();
            let nucleus_set: HashSet<usize> = nucleus_indices.into_iter().collect();
            for i in 0..vocab_size {
                if !nucleus_set.contains(&i) {
                    filtered_logits[i] = f32::NEG_INFINITY;
                }
            }
        }

        // Softmax and sample
        let max_l = *filtered_logits.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(&0.0);
        let exp_logits: Vec<f32> = filtered_logits.iter().map(|l| (*l - max_l).exp()).collect();
        let sum_exp = exp_logits.iter().sum::<f32>();
        let probs: Vec<f32> = exp_logits.iter().map(|e| *e / sum_exp).collect();

        // Sample
        let r = thread_rng().gen::<f32>();
        let mut cum = 0.0;
        for (i, prob) in probs.iter().enumerate() {
            cum += prob;
            if r < cum {
                return Ok(i as u32);
            }
        }
        Ok((probs.len().saturating_sub(1)) as u32) // fallback
    }
}

impl Drop for LlmInference {
    fn drop(&mut self) {
        // Resources automatically cleaned
    }
}
