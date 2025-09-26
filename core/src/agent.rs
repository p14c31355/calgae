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
        let start_idx = if let Some(pos) = response.find("```rust") { pos + 7 } else { 0 };
        let end_idx = if let Some(pos) = response.rfind("```") { pos } else { response.len() };
        let code_start = start_idx.min(response.len());
        let code = if code_start < response.len() {
            &response[code_start..min(end_idx, response.len())]
        } else {
            &response[..]
        }.to_string().trim().to_string();
        
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
