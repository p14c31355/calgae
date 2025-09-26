use std::process::{Command, Output};

pub struct LlmInference {
    llama_bin: String,
    model: String,
}

impl LlmInference {
    pub fn new(llama_bin: String, model: String) -> Self {
        LlmInference { llama_bin, model }
    }

    pub fn infer(&self, prompt: &str, tokens: usize) -> Result<String, Box<dyn std::error::Error>> {
        let output: Output = Command::new(&self.llama_bin)
            .arg("-m")
            .arg(&self.model)
            .arg("--prompt")
            .arg(prompt)
            .arg("-n")
            .arg(tokens.to_string())
            .arg("--log-disable")
            .output()?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("LLM inference failed: {}", String::from_utf8_lossy(&output.stderr))
            )))
        }
    }
}
