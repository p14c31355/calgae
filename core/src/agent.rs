use super::cli::Args;
use super::inference::{LlmInference, InferenceError};
use regex::Regex;

pub struct Agent {
    inference: LlmInference,
}

#[derive(Debug)]
pub enum AgentError {
    Inference(InferenceError),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::Inference(err) => write!(f, "{}", err),
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AgentError::Inference(err) => Some(err),
        }
    }
}

impl From<InferenceError> for AgentError {
    fn from(err: InferenceError) -> Self {
        AgentError::Inference(err)
    }
}

impl Agent {
    pub fn new(llama_bin: &std::path::Path, model: &std::path::Path) -> Self {
        let inference = LlmInference::new(llama_bin.to_path_buf(), model.to_path_buf());
        Agent { inference }
    }

    pub fn generate_code(&self, args: &Args) -> Result<String, AgentError> {
        // Enhance prompt for code generation
        let enhanced_prompt = format!(
            "You are a helpful coding assistant. Generate Rust code for: {}",
            args.prompt
        );

        let response = self.inference.infer(&enhanced_prompt, args.tokens)?;

        // Robust extraction of Rust code block using regex
        let re = Regex::new(r#"```rust\s*(.*?)\s*```"#).unwrap();
        let code = re.captures(&response)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_else(|| {
                // Fallback: try extracting any code block
                if let Some(caps) = Regex::new(r#"```(?:\w+)?\s*(.*?)\s*```"#).unwrap().captures(&response) {
                    caps.get(1).map(|m| m.as_str().trim().to_string()).unwrap_or_default()
                } else {
                    "".to_string()
                }
            });

        if code.is_empty() {
            Ok(response)
        } else {
            Ok(format!("Generated code:\n{}", code))
        }
    }
}

pub fn run_agent(args: &Args) -> Result<(), AgentError> {
    let agent = Agent::new(args.llama_bin.clone(), args.model.clone());
    let result = agent.generate_code(args)?;
    println!("{}", result);
    Ok(())
}
