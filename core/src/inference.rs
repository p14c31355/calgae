use std::process::Command;

#[derive(Debug)]
pub enum InferenceError {
    CommandFailed(String),
    Utf8Error(std::string::FromUtf8Error),
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::CommandFailed(err) => write!(f, "LLM inference command failed: {}", err),
            InferenceError::Utf8Error(err) => write!(f, "Invalid UTF-8 in output: {}", err),
        }
    }
}

impl std::error::Error for InferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            InferenceError::CommandFailed(_) => None,
            InferenceError::Utf8Error(err) => Some(err),
        }
    }
}

pub struct LlmInference {
    llama_bin: std::path::PathBuf,
    model: std::path::PathBuf,
}

impl LlmInference {
    pub fn new(llama_bin: std::path::PathBuf, model: std::path::PathBuf) -> Self {
        LlmInference { llama_bin, model }
    }

    pub fn infer(&self, prompt: &str, tokens: usize) -> Result<String, InferenceError> {
        let output = Command::new(self.llama_bin.as_os_str())
            .arg("-m")
            .arg(self.model.as_os_str())
            .arg("--prompt")
            .arg(prompt)
            .arg("-n")
            .arg(tokens.to_string())
            .arg("--log-disable")
            .output()
            .map_err(|e| InferenceError::CommandFailed(format!("Failed to run command: {}", e)))?;

        if output.status.success() {
            String::from_utf8(output.stdout).map_err(InferenceError::Utf8Error)
                .map(|s| s.trim().to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(InferenceError::CommandFailed(format!("LLM inference failed: {}", stderr)))
        }
    }
}
