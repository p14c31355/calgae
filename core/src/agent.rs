use super::cli::Args;
use super::inference::LlmInference;

pub struct Agent {
    inference: LlmInference,
}

impl Agent {
    pub fn new(llama_bin: String, model: String) -> Self {
        let inference = LlmInference::new(llama_bin, model);
        Agent { inference }
    }

    pub fn generate_code(&self, args: &Args) -> Result<String, Box<dyn std::error::Error>> {
        // Enhance prompt for code generation
        let enhanced_prompt = format!("You are a helpful coding assistant. Generate Rust code for: {}", args.prompt);
        
        let response = self.inference.infer(&enhanced_prompt, args.tokens)?;
        
        // Safe extraction of code block if present, fallback to full response
        let code = if let Some(start) = response.find("```rust") {
            let start = start + "```rust".len();
            let code_block = &response[start..];
            let end = code_block.find("```").unwrap_or(code_block.len());
            code_block[..end].trim().to_string()
        } else {
            String::new()
        };
        
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
    println!("{}\n", result);
    Ok(()) 
}
