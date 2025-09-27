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
            let mut params = llama_default_params();
            params.n_ctx = 2048;
            params.seed = 1234u32;
            params.n_threads = (std::thread::available_parallelism()
                .map_or(4u32, |n| n.get() as u32)) as i32;

            // Load model
            let model = llama_load_model_from_file(
                model_path.to_str().ok_or(InferenceError::ModelLoadError("Invalid path".to_string()))?.as_ptr() as *const std::os::raw::c_char,
                params,
            );
            if model.is_null() {
                return Err(InferenceError::ModelLoadError("llama_load_model_from_file failed".to_string()));
            }

            // Initialize context
            let ctx = llama_new_context_with_model(model, params);
            if ctx.is_null() {
                llama_free_model(model);
                return Err(InferenceError::ContextInitError("llama_new_context_with_model failed".to_string()));
            }

            Ok(LlmInference { ctx, model, params })
        }
    }

    pub fn infer(&self, prompt: &str, tokens: usize) -> Result<String, InferenceError> {
        unsafe {
            // Tokenize prompt
            let mut tokens_arr: [llama_token; 2048] = std::mem::zeroed();
            let n_prompt = llama_tokenize(
                self.ctx,
                prompt.as_ptr() as *const std::os::raw::c_char,
                prompt.len() as i32,
                &mut tokens_arr,
                tokens_arr.len() as i32,
                true,
            );
            if n_prompt < 0 {
                return Err(InferenceError::TokenizationError("llama_tokenize failed".to_string()));
            }

            // Simple sampling parameters (greedy for minimal)
            let mut logits = std::vec::from_raw_parts(
                llama_get_logits(self.ctx),
                (llama_n_vocab(self.model) as usize) * (llama_n_ctx(self.ctx) as usize),
                (llama_n_vocab(self.model) as usize) * (llama_n_ctx(self.ctx) as usize),
            );

            // Evaluate prompt tokens
            if llama_eval(self.ctx, tokens_arr.as_ptr(), n_prompt, 0) != 0 {
                return Err(InferenceError::EvaluationError("llama_eval for prompt failed".to_string()));
            }

            let mut output_tokens: Vec<llama_token> = Vec::new();
            for _ in 0..tokens {
                // Get logits for last token
                let n_vocab = llama_n_vocab(self.model);
                let last_logits = std::slice::from_raw_parts(
                    logits.as_ptr().add((llama_n_ctx(self.ctx) - 1) * n_vocab),
                    n_vocab as usize,
                );

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
                if next_token == llama_token_eos(self.model) {
                    break;
                }

                output_tokens.push(next_token as llama_token);

                // Eval next token
                if llama_eval(self.ctx, &next_token as *const llama_token, 1, n_prompt + output_tokens.len() as i32 - 1) != 0 {
                    return Err(InferenceError::EvaluationError("llama_eval for generation failed".to_string()));
                }
            }

            // Decode output tokens to string
            let mut output = String::new();
            for &tok in &output_tokens {
                let mut buf: [std::os::raw::c_char; 256] = [0; 256];
                let n = llama_token_to_piece(
                    self.ctx,
                    tok,
                    buf.as_mut_ptr(),
                    buf.len() as i32,
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
