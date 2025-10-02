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
use std::io::Read;
use std::fs::File;

use rand::prelude::*;

use foreign_types::ForeignTypeRef;
use candle_core::safetensors::{SafeTensors, MmapedSafetensors};
use crate::zig_ffi::{zig_quantize_buffer, zig_quantize_model}; // zig_ffi からインポート
use candle_transformers::generation::LogitsProcessor;

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
            let safetensors_files: Vec<PathBuf> = fs::read_dir(&model_path)?
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().map_or(false, |s| s == "safetensors"))
                .map(|e| e.path())
                .collect();
            if safetensors_files.is_empty() {
                return Err(anyhow!("No safetensors files found in model directory"));
            }

            let quantized_dir = model_path.join(format!("quantized_{}", bits));
            fs::create_dir_all(&quantized_dir)?;

            for file in safetensors_files {
                info!("Quantizing file: {}", file.display());
                let mmaped = unsafe { MmapedSafetensors::new(&file)? };
                let tensors = mmaped.tensors();

                let file_name = file.file_name().ok_or(anyhow!("Invalid file name"))?.to_str().ok_or(anyhow!("Invalid UTF-8 file name"))?;
                let quantized_output_path = quantized_dir.join(format!("{}_{}.bin", file_name.strip_suffix(".safetensors").unwrap_or(file_name), bits));
                let mut output_file = File::create(&quantized_output_path)?;

                for (tensor_name, tensor_info) in tensors.iter() {
                    let tensor_data = mmaped.load_tensor(tensor_info, &device)?;
                    let tensor_data_f32 = tensor_data.to_dtype(DType::F32)?.to_vec1::<f32>()?;

                    let num_elements = tensor_data_f32.len();
                    let output_buffer_size = (num_elements * bits as usize + 7) / 8; // Calculate buffer size for quantized data
                    let mut output_buffer = vec![0u8; output_buffer_size];
                    let mut scale: f32 = 0.0;

                    let res = unsafe {
                        zig_quantize_buffer(
                            tensor_data_f32.as_ptr(),
                            num_elements,
                            bits,
                            output_buffer.as_mut_ptr() as *mut c_void,
                            &mut scale as *mut f32,
                        )
                    };

                    if res < 0 {
                        error!("Zig quantization failed for tensor {} in file {}", tensor_name, file.display());
                        return Err(anyhow!("Zig quantization failed for tensor {}", tensor_name));
                    } else {
                        // Write tensor name length, name, scale, data length, and quantized data to the output file
                        let data_len = res as usize;
                        output_file.write_all(&(tensor_name.len() as u32).to_le_bytes())?;
                        output_file.write_all(tensor_name.as_bytes())?;
                        output_file.write_all(&scale.to_le_bytes())?;
                        output_file.write_all(&data_len.to_le_bytes())?;
                        output_file.write_all(&output_buffer[..data_len])?;
                        info!("Tensor {} quantized to {} bits, scale: {}", tensor_name, bits, scale);
                    }
                }
                info!("File {} quantized to {} bits at {}", file.display(), bits, quantized_output_path.display());
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


        let mut vb = if let Some(bits) = quantize_bits {
            // Custom loader for quantized .bin files
            let mut tensors = std::collections::HashMap::new();
            for file_path in weights {
                let mut file = File::open(&file_path)?;
                while let Ok(name_len_bytes) = file.read_exact(&mut [0u8; 4]) {
                    let name_len = u32::from_le_bytes(name_len_bytes) as usize;
                    let mut name_bytes = vec![0u8; name_len];
                    file.read_exact(&mut name_bytes)?;
                    let tensor_name = String::from_utf8(name_bytes)?;

                    let mut scale_bytes = [0u8; 4];
                    file.read_exact(&mut scale_bytes)?;
                    let scale = f32::from_le_bytes(scale_bytes);

                    // Read quantized data (this part needs to be adjusted based on how zig_quantize_buffer saves data)
                    // For now, assuming the rest of the file is quantized data for this tensor
                    let mut quantized_data_buffer = Vec::new();
                    let mut len_bytes = [0u8; 8];
                    file.read_exact(&mut len_bytes)?;
                    let data_len = u64::from_le_bytes(len_bytes) as usize;
                    quantized_data_buffer.resize(data_len, 0);
                    file.read_exact(&mut quantized_data_buffer)?;

                    let dequantized_tensor = Self::load_quantized_tensor_from_buffer(&quantized_data_buffer, bits, scale, &device)?;
                    tensors.insert(tensor_name, dequantized_tensor);
                }
            }
            VarBuilder::from_tensors(tensors, dtype, &device)
        } else {
            let weights_ref: Vec<&Path> = weights.iter().map(|p| p.as_path()).collect();
            unsafe { VarBuilder::from_mmaped_safetensors(&weights_ref, dtype, &device) }?
        };

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
        let mut logits_processor = LogitsProcessor::new(0, Some(temperature), Some(top_p)); // seed は適当な値 (0) を設定
        if top_k > 0 {
            logits_processor.top_k = Some(top_k);
        }

        let last_pos = Tensor::new(&[(prompt_len - 1) as i64], &self.device)?;
        let next_token_logits = logits.index_select(&last_pos, 0usize)?.squeeze(0)?.to_dtype(DType::F32)?;
        let next_token = logits_processor.sample(&next_token_logits)?;
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

            let next_token = logits_processor.sample(&next_token_logits)?;
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

    fn load_quantized_tensor_from_buffer(buffer: &[u8], bits: u8, scale: f32, device: &Device) -> Result<Tensor> {
        let num_elements = if bits == 4 {
            buffer.len() * 2
        } else {
            buffer.len()
        };

        let mut dequantized_data = Vec::with_capacity(num_elements);

        if bits == 8 {
            for &byte in buffer.iter() {
                let val = (byte as i8) as f32 * scale;
                dequantized_data.push(val);
            }
        } else if bits == 4 {
            for &byte in buffer.iter() {
                let lower_nibble = (byte & 0x0F);
                let upper_nibble = ((byte >> 4) & 0x0F);

                // Convert 4-bit unsigned to 4-bit signed, then to f32
                let val1 = if lower_nibble > 7 { lower_nibble as i8 - 16 } else { lower_nibble as i8 } as f32 * scale;
                let val2 = if upper_nibble > 7 { upper_nibble as i8 - 16 } else { upper_nibble as i8 } as f32 * scale;

                dequantized_data.push(val1);
                dequantized_data.push(val2);
            }
        } else {
            return Err(anyhow!("Unsupported quantization bits: {}", bits));
        }

        Tensor::new(&dequantized_data, device)
    }
}

impl Drop for LlmInference {
    fn drop(&mut self) {
        // Resources automatically cleaned
    }
}
