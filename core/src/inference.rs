include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Unsafe blocks are used for FFI calls
#[derive(Debug)]
pub enum InferenceError {
    ModelLoadError(String),
    ContextInitError(String),
    TokenizationError(String),
    EvaluationError(String),
    NoOutputError,
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::ModelLoadError(err) => write!(f, "Failed to load model: {}", err),
            InferenceError::ContextInitError(err) => write!(f, "Failed to initialize context: {}", err),
            InferenceError::TokenizationError(err) => write!(f, "Tokenization failed: {}", err),
            InferenceError::EvaluationError(err) => write!(f, "Evaluation failed: {}", err),
            InferenceError::NoOutputError => write!(f, "No output generated"),
        }
    }
}

impl std::error::Error for InferenceError {}

pub struct LlmInference {
    ctx: *mut llama_context,
    model: *mut llama_model,
    params: llama_context_params,
}

impl LlmInference {
    pub fn new(model_path: std::path::PathBuf) -> Result<Self, InferenceError> {
        unsafe {
            let mut model_params = llama_model_default_params();
            model_params.n_gpu_layers = 0i32;

            // Load model
            let model = llama_load_model_from_file(
                model_path.to_str().ok_or(InferenceError::ModelLoadError("Invalid path".to_string()))?.as_ptr() as *const std::os::raw::c_char,
                model_params,
            );
            if model.is_null() {
                return Err(InferenceError::ModelLoadError("llama_load_model_from_file failed".to_string()));
            }

            let mut context_params = llama_context_default_params();
            context_params.n_ctx = 2048;
            context_params.n_seed =

            // Initialize context
            let ctx = llama_new_context_with_model(model, context_params);
            if ctx.is_null() {
                llama_free_model(model);
                return Err(InferenceError::ContextInitError("llama_new_context_with_model failed".to_string()));
            }

            Ok(LlmInference { ctx, model, params: context_params })
        }
    }

    pub fn infer(&self, prompt: &str, tokens: usize) -> Result<String, InferenceError> {
        unsafe {
            let vocab = llama_model_get_vocab(self.model);

            // Tokenize prompt
            let mut tokens_arr: [llama_token; 2048] = std::mem::zeroed();
            let n_prompt = llama_tokenize(
                vocab as *const llama_vocab,
                prompt.as_bytes().as_ptr() as *const std::os::raw::c_char,
                prompt.len() as i32,
                tokens_arr.as_mut_ptr(),
                tokens_arr.len() as i32,
                true,
                true,
            );
            if n_prompt < 0 {
                return Err(InferenceError::TokenizationError("llama_tokenize failed".to_string()));
            }

            let n_vocab_i32 = llama_vocab_n_tokens( vocab );
            let n_vocab = n_vocab_i32 as usize;

            // Evaluate prompt tokens
            let mut batch = llama_batch_get_one(tokens_arr.as_ptr(), n_prompt as i32);
            if llama_decode(self.ctx, batch) != 0 {
                return Err(InferenceError::EvaluationError("llama_decode for prompt failed".to_string()));
            }

            let mut output_tokens: Vec<llama_token> = Vec::new();
            let mut n_past = n_prompt;
            for _ in 0..tokens {
                let logits_ptr = llama_get_logits(self.ctx);
                let last_logits = std::slice::from_raw_parts( logits_ptr, n_vocab );

                // Greedy: take max logit token
                let mut max_logit = f32::MIN;
                let mut next_token = 0i32;
                for (i, &logit) in last_logits.iter().enumerate() {
                    if logit > max_logit {
                        max_logit = logit;
                        next_token = i as i32;
                    }
                }

                // Check EOS
                let eos_token = llama_vocab_eos( vocab );
                if next_token == eos_token {
                    break;
                }

                output_tokens.push(next_token );

                // Eval next token
                let mut batch = llama_batch_get_one(&next_token as *const llama_token, 1i32);
                if llama_decode(self.ctx, batch) != 0 {
                    return Err(InferenceError::EvaluationError("llama_decode for generation failed".to_string()));
                }
                n_past += 1;
            }

            // Decode output tokens to string
            let mut output = String::new();
            for &tok in &output_tokens {
                let mut buf: [std::os::raw::c_char; 256] = [0; 256];
                let n = llama_token_to_piece(
                    vocab,
                    tok,
                    buf.as_mut_ptr(),
                    buf.len() as i32,
                    0i32,
                    false,
                );
                if n < 0 {
                    continue;
                }
                let piece = std::str::from_utf8(std::slice::from_raw_parts(
                    buf.as_ptr() as *const u8,
                    n as usize,
                ))
                .unwrap_or("")
                .to_string();
                output.push_str(&piece);
            }

            if output.is_empty() {
                Err(InferenceError::NoOutputError)
            } else {
                Ok(output)
            }
        }
    }
}

impl Drop for LlmInference {
    fn drop(&mut self) {
        unsafe {
            llama_free(self.ctx);
            llama_free_model(self.model);
        }
    }
}
