use super::inference::LlmInference;
use once_cell::sync::Lazy;
use log::warn;
use regex::Regex;

use anyhow::Result as AnyhowResult;
use std::fs::File;
use std::process;
use std::sync::Arc;
use tempfile::Builder;

use std::io::Write;

unsafe extern "C" {
    fn matrix_mult_c(n: i32, p: i32, q: i32, a: *const f32, b: *const f32, c: *mut f32) -> i32;
}

fn use_codon_kernel(a: &[f32], b: &[f32], q: usize) -> Vec<f32> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    // Assume a: n x q (flattened, n rows, q cols), b: q x p
    assert!(a.len() % q == 0, "Input 'a' length must be a multiple of q");
    assert!(b.len() % q == 0, "Input 'b' length must be a multiple of q");
    let n = (a.len() / q) as i32;
    let p = (b.len() / q) as i32;
    let mut c = vec![0.0f32; (n * p) as usize];
    let status = unsafe {
        matrix_mult_c(n, p, q as i32, a.as_ptr(), b.as_ptr(), c.as_mut_ptr())
    };
    if status != 0 {
        eprintln!("Codon kernel failed with status: {}", status);
        return vec![];
    }
    c
}

#[derive(Clone)]
pub struct Agent {
    inference: Arc<LlmInference>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
}

impl Agent {
    pub async fn new(
        model: std::path::PathBuf, 
        temperature: f32, 
        top_k: usize, 
        top_p: f32,
        quantize_bits: Option<u8>,
        quantize_mode: Option<&str>
    ) -> AnyhowResult<Self> {
        let bits = if let Some(bits_opt) = quantize_bits {
            match quantize_mode {
                Some("awq") => Some(4u8),
                Some("smoothquant") => Some(8u8),
                _ => Some(bits_opt),
            }
        } else {
            None
        };

        if let Some(bits_opt) = quantize_bits {
            if let Some(mode) = quantize_mode {
                match mode {
                    "awq" => if bits_opt != 4 { warn!("Overriding user bits to 4 for AWQ mode"); }
                    "smoothquant" => if bits_opt != 8 { warn!("Overriding user bits to 8 for SmoothQuant mode"); }
                    _ => {}
                }
            }
        }

        let inference = Arc::new(LlmInference::new(model, None, bits)?);
        Ok(Agent { inference, temperature, top_k, top_p })
    }

    async fn infer_async(
        &self,
        prompt: &str,
        tokens: usize,
    ) -> AnyhowResult<String> {
        // Wrap sync inference in blocking task for async
        let inference = self.inference.clone();
        let prompt = prompt.to_string();
        let temperature = self.temperature;
        let top_k = self.top_k;
        let top_p = self.top_p;

        tokio::task::spawn_blocking(move || {
            inference.infer(&prompt, tokens, temperature, top_k, top_p)
        })
        .await
        .map_err(|e| anyhow::anyhow!("blocking task failed: {}", e))?
    }

    /// Generate code for a single prompt asynchronously
    pub async fn generate_code(&self, prompt: &str, tokens: usize) -> AnyhowResult<String> {
        // Enhance prompt for coding agent: Follow instructions to generate code, suggest edits, or explain
        let enhanced_prompt = format!(
            "You are Calgae, a CLI coding agent. The user is giving a programming task for the current Rust project (Calgae LLM runtime). Analyze the instruction and respond with:\n1. Explanation of how to implement it.\n2. Generated or edited Rust code in blocks.\n3. Specific file paths to apply changes (e.g., 'Add to core/src/inference.rs').\n4. Commands to run (e.g., 'cargo build').\nKeep responses concise and actionable.\n\nUser task: {}",
            prompt
        );

        let response = self
            .infer_async(&enhanced_prompt, tokens)
            .await?;

        // Extract code blocks and instructions
        static RUST_CODE_BLOCK_RE: Lazy<Regex> =
            Lazy::new(|| Regex::new(r#"(?s)```rust\s*(.*?)```"#).unwrap());
        static COMMAND_RE: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"Commands?:? ?([^\n]+)").unwrap());

        let mut full_response = response.clone();
        let code_blocks = RUST_CODE_BLOCK_RE.captures_iter(&response)
            .map(|caps| caps.get(1).unwrap().as_str().trim().to_string())
            .collect::<Vec<_>>();
        
        if !code_blocks.is_empty() {
            full_response += "\n\nExtracted code blocks:\n";
            for (i, code) in code_blocks.iter().enumerate() {
                full_response += &format!("Block {}:\n{}\n", i+1, code);
            }
        }

        if let Some(cmd_caps) = COMMAND_RE.captures(&response) {
            full_response += &format!("\nSuggested command: {}", cmd_caps.get(1).unwrap().as_str().trim());
        }

        Ok(full_response.trim().to_string())
    }

    /// Run multiple generation tasks in parallel
    pub async fn generate_codes_parallel(
        &self,
        prompts: Vec<&str>,
        tokens: usize,
    ) -> AnyhowResult<Vec<String>> {
        let tasks = prompts.into_iter().map(|prompt| {
            let agent = self.clone();
            let prompt = prompt.to_string();
            tokio::spawn(async move { agent.generate_code(&prompt, tokens).await })
        });

        let results = futures::future::try_join_all(tasks).await?;
        results.into_iter().collect()
    }
}

fn compile_and_execute(code: &str) -> AnyhowResult<()> {
    println!("Compiling and executing...\n");
    let rs_path = Builder::new()
        .prefix("calgae_generated")
        .suffix(".rs")
        .tempfile()?
        .path()
        .to_path_buf();

    let mut f = File::create(&rs_path)
        .map_err(|e| anyhow::anyhow!("Failed to create temp file: {}", e))?;
    f.write_all(code.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write code: {}", e))?;

    let output = process::Command::new("rustc")
        .arg(&rs_path)
        .output()
        .map_err(|e| anyhow::anyhow!("Failed to compile: {}", e))?;

    if !output.status.success() {
        eprintln!("Compilation failed:\n{}", String::from_utf8_lossy(&output.stderr));
        return Ok(());
    }

    let exe_path = rs_path.with_extension(std::env::consts::EXE_EXTENSION);
    let run_output = process::Command::new(exe_path)
        .output()
        .map_err(|e| anyhow::anyhow!("Failed to run executable: {}", e))?;

    if run_output.status.success() {
        if !run_output.stdout.is_empty() {
            println!("Execution output:\n{}", String::from_utf8_lossy(&run_output.stdout));
        }
    } else {
        println!("Execution failed: {}", run_output.status);
    }

    if !run_output.stderr.is_empty() {
        eprintln!("Runtime stderr:\n{}", String::from_utf8_lossy(&run_output.stderr));
    }

    Ok(())
}

fn run_interactive_loop(
    agent: &Agent,
    tokens: usize,
    execute: bool,
) {
    use std::io::{self, BufRead};

    println!("\nCalgae Interactive Coding Agent Ready!");
    println!("Enter programming task (e.g., 'Add a function to compute fibonacci in inference.rs'):");
    println!("Provide instructions; I'll generate code, suggest changes, and commands.\n");
    println!("Type 'exit' to quit, 'exec' to toggle code execution, 'help' for commands.\n");
    let mut execute_mode = execute;
    if execute_mode {
        println!("\n\x1b[93;1mWARNING: Code execution is enabled. Code generated by the LLM will be compiled and run on your machine, which can be a security risk. Use 'exec' to toggle this setting.\x1b[0m");
    }
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let input = match line {
            Ok(input) => input.trim().to_string(),
            Err(_) => continue,
        };
        match input.to_lowercase().as_str() {
            "exit" => {
                println!("Goodbye!");
                break;
            }
            "exec" => {
                execute_mode = !execute_mode;
                println!("Code execution {}abled.", if execute_mode { "en" } else { "dis" });
                continue;
            }
            "help" => {
                println!("Commands: 'task description' for code generation/suggestions, 'exec' to toggle execution, 'exit' to quit.");
                continue;
            }
            "" | "#" => continue,
            _ => {}
        }
        println!("Analyzing and generating for: {}\n", input);
        let agent_clone = agent.clone();
        let input_clone = input.clone();
        let code_result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                agent_clone.generate_code(&input_clone, tokens).await
            })
        });
        match code_result {
            Ok(response) => {
                println!("\nCalgae Response:\n---");
                println!("{}", response);
                println!("---\n");

                if execute_mode && response.contains("cargo ") || response.contains("rustc ") {
                    // Simple command extraction and exec if flagged
                    if let Some(cmd_start) = response.find("Command: ") {
                        let cmd = &response[cmd_start + 9..].lines().next().unwrap_or("").trim();
                        println!("Executing suggested command: {}", cmd);
                        if let Err(e) = std::process::Command::new("sh").arg("-c").arg(cmd).output() {
                            eprintln!("Command execution error: {}", e);
                        }
                    }
                }

                if execute_mode {
                    // Extract and execute any standalone code if present
                    if let Some(code) = response.lines().find(|l| l.starts_with("fn ")) {
                        println!("Executing extracted function code...");
                        if let Err(e) = compile_and_execute(code) {
                            eprintln!("Execution error: {}", e);
                        }
                    }
                } else {
                    println!("(Execution disabled. Use 'exec' to enable.)\nNext input:");
                }
                println!("Next input:");
            }
            Err(e) => {
                eprintln!("Generation failed: {}", e);
                println!("\nNext input:");
            }
        }
    }
}

pub async fn run_agent(
    model: std::path::PathBuf,
    prompt: String,
    tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    execute: bool,
    interactive: bool,
    quantize_bits: Option<u8>,
    quantize_mode: Option<String>,
) -> AnyhowResult<()> {
    let agent = Agent::new(model, temperature, top_k, top_p, quantize_bits, quantize_mode.as_deref()).await?;

    if interactive {
        run_interactive_loop(&agent, tokens, execute);
    } else {
        let response = agent.generate_code(&prompt, tokens).await?;

        println!("Calgae Response for '{}':\n---\n{}\n---", prompt, response);

        // Suggest applying changes manually or execute if simple
        if execute {
            if response.contains("fn ") || response.contains("impl ") {
                compile_and_execute(&response).map_err(|e| anyhow::anyhow!("Execution failed: {}", e))?;
            }
        }
    }

    Ok(())
}
