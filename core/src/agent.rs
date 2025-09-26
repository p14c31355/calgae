use super::cli::Args;
use super::inference::LlmInference;

pub struct Agent {
    inference: LlmInference,
}

impl Agent {
    pub fn new(llama_bin: &std::path::Path, model: &std::path::Path) -> Self {
        let inference = LlmInference::new(llama_bin.to_path_buf(), model.to_path_buf());
        Agent { inference }
    }

    pub fn generate_code(&self, args: &Args) -> Result<String, Box<dyn std::error::Error>> {
        // Enhance prompt for code generation
        let enhanced_prompt = format!(
            "You are a helpful coding assistant. Generate Rust code for: {}",
            args.prompt
        );

        let response = self.inference.infer(&enhanced_prompt, args.tokens)?;

        // Safe extraction of code block if present, fallback to full response
        let code = response
            .split("```rust")
            .nth(1)
            .and_then(|s| s.split("```").next())
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

        if code.is_empty() {
            Ok(response)
        } else {
            Ok(format!("Generated code:\n{}", code))
        }
    }
}

pub fn run_agent(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new(args.llama_bin.clone(), args.model.clone());
    let result = agent.generate_code(args)?;
    println!("{}", result);
    Ok(())
}
