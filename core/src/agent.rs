use super::inference::LlmInference;
use once_cell::sync::Lazy;
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
}

impl Agent {
    pub async fn new(model: std::path::PathBuf) -> AnyhowResult<Self> {
        let inference = Arc::new(LlmInference::new(model, None)?);
        Ok(Agent { inference })
    }

    async fn infer_async(
        &self,
        prompt: &str,
        tokens: usize,
    ) -> AnyhowResult<String> {
        // Wrap sync inference in blocking task for async
        let inference = self.inference.clone();
        let prompt = prompt.to_string();

        tokio::task::spawn_blocking(move || {
            inference.infer(&prompt, tokens)
        })
        .await
        .map_err(|e| anyhow::anyhow!("blocking task failed: {}", e))?
    }

    /// Generate code for a single prompt asynchronously
    pub async fn generate_code(&self, prompt: &str, tokens: usize) -> AnyhowResult<String> {
        // Enhance prompt for code generation
        let enhanced_prompt = format!(
            "You are a helpful coding assistant. Generate only Rust code for: {}",
            prompt
        );

        let response = self
            .infer_async(&enhanced_prompt, tokens)
            .await?;

        // Robust extraction of code block using lazy-compiled regexes
        static RUST_CODE_BLOCK_RE: Lazy<Regex> =
            Lazy::new(|| Regex::new(r#"(?s)```rust\s*(.*?)```"#).unwrap());
        static ANY_CODE_BLOCK_RE: Lazy<Regex> =
            Lazy::new(|| Regex::new(r#"(?s)```(?:\w+)?\s*(.*?)```"#).unwrap());

        let code = RUST_CODE_BLOCK_RE
            .captures(&response)
            .and_then(|caps| caps.get(1))
            .or_else(||
                ANY_CODE_BLOCK_RE
                    .captures(&response)
                    .and_then(|caps| caps.get(1))
            )
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or(response.trim().to_string());

        Ok(code)
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
    println!("Enter programming task (e.g., 'Write a Rust function to compute fibonacci'):");
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
                println!("Commands: 'task description' to generate code, 'exec' to toggle execution, 'exit' to quit.");
                continue;
            }
            "" | "#" => continue,
            _ => {}
        }
        println!("Generating code for: {}", input);
        let agent_clone = agent.clone();
        let input_clone = input.clone();
        let code_result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                agent_clone.generate_code(&input_clone, tokens).await
            })
        });
        match code_result {
            Ok(code) => {
                println!("\nGenerated code:\n---");
                println!("{}", code);
                println!("---\n");

                if execute_mode {
                    if let Err(e) = compile_and_execute(&code) {
                        eprintln!("Execution error: {}", e);
                    } else {
                        println!("\nNext input:");
                    }
                } else {
                    println!("(Execution disabled. Use 'exec' to enable.)\nNext input:");
                }
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
    execute: bool,
    interactive: bool,
) -> AnyhowResult<()> {
    let agent = Agent::new(model).await?;

    if interactive {
        run_interactive_loop(&agent, tokens, execute);
    } else {
        let code = agent.generate_code(&prompt, tokens).await?;

        println!("Generated code:\n{}\n", code);

        if execute {
            compile_and_execute(&code).map_err(|e| anyhow::anyhow!("Execution failed: {}", e))?;
        }
    }

    Ok(())
}
