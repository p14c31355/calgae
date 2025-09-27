use super::cli::Args;
use super::inference::{LlmInference, InferenceError};
use once_cell::sync::Lazy;
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
    pub fn new(model: &std::path::Path) -> Result<Self, AgentError> {
        let inference = LlmInference::new(model.to_path_buf()).map_err(AgentError::Inference)?;
        Ok(Agent { inference })
    }

    pub fn generate_code(&self, args: &Args) -> Result<String, AgentError> {
        // Enhance prompt for code generation
        let enhanced_prompt = format!(
            "You are a helpful coding assistant. Generate Rust code for: {}",
            args.prompt
        );

        let response = self.inference.infer(&enhanced_prompt, args.tokens)?;

        // Robust extraction of code block using lazy-compiled regexes
        static RUST_CODE_BLOCK_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"(?s)```rust\s*(.*?)```"#).unwrap());
        static ANY_CODE_BLOCK_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"(?s)```(?:\w+)?\s*(.*?)```"#).unwrap());

        let code = RUST_CODE_BLOCK_RE.captures(&response)
            .and_then(|caps| caps.get(1))
            .or_else(|| {
                ANY_CODE_BLOCK_RE.captures(&response)
                    .and_then(|caps| caps.get(1))
            })
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_default();

        if code.is_empty() {
            Ok(response)
        } else {
            Ok(format!("Generated code:\n{}", code))
        }
    }
}

pub fn run_agent(args: &Args) -> Result<(), AgentError> {
    let agent = Agent::new(&args.model)?;
    let result = agent.generate_code(args)?;
    println!("{}", result);
    Ok(())
}
