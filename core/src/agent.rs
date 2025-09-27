use super::inference::LlmInference;
use once_cell::sync::Lazy;
use regex::Regex;

use anyhow::Result as AnyhowResult;
use std::fs::File;
use std::io::Write;
use std::process;
use std::sync::Arc;
use tempfile::Builder;

unsafe extern "C" {
    fn codon_matrix_mult(n: i32, p: i32, q: i32, a: *const f32, b: *const f32, c: *mut f32) -> i32;
}

fn use_codon_kernel(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len() as i32 / 1; // Assume flattened
    let p = b.len() as i32 / 1;
    let q = 1; // Square for simplicity
    let mut c = vec![0.0f32; (n * p) as usize];
    unsafe {
        codon_matrix_mult(n, p, q, a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
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
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> AnyhowResult<String> {
        // Wrap sync inference in blocking task for async
        let inference = self.inference.clone();
        let prompt = prompt.to_string();

        tokio::task::spawn_blocking(move || {
            inference.infer(&prompt, tokens, temperature, top_k, top_p)
        })
        .await
        .map_err(|e| anyhow::anyhow!("blocking task failed: {}", e))?
    }

    /// Generate code for a single prompt asynchronously
    pub async fn generate_code(&self, prompt: &str, tokens: usize, temperature: f32, top_k: usize, top_p: f32) -> AnyhowResult<String> {
        // Enhance prompt for code generation
        let enhanced_prompt = format!(
            "You are a helpful coding assistant. Generate only Rust code for: {}",
            prompt
        );

        let response = self
            .infer_async(&enhanced_prompt, tokens, temperature, top_k, top_p)
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

        // Integrate Codon kernel for example computation in orchestrator
        let a = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let b = vec![5.0f32, 6.0f32, 7.0f32, 8.0f32];
        let result = use_codon_kernel(&a, &b);
        println!("Codon kernel integrated result: {:?}", result);

        Ok(code)
    }

    /// Run multiple generation tasks in parallel
    pub async fn generate_codes_parallel(
        &self,
        prompts: Vec<&str>,
        tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> AnyhowResult<Vec<String>> {
        let tasks = prompts.into_iter().map(|prompt| {
            let agent = self.clone();
            let prompt = prompt.to_string();
            tokio::spawn(async move { agent.generate_code(&prompt, tokens, temperature, top_k, top_p).await })
        });

        let results = futures::future::try_join_all(tasks).await?;
        results.into_iter().collect()
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
) -> AnyhowResult<()> {
    let agent = Agent::new(model).await?;

    if interactive {
        use std::io::{self, BufRead};

        println!("Interactive mode: Enter prompts (type 'exit' to quit).");
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let input = line?.trim().to_string();
            if input.to_lowercase() == "exit" {
                break;
            }
            if input.is_empty() {
                continue;
            }
            let code = agent.generate_code(&input, tokens, temperature, top_k, top_p).await?;

            if execute {
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
                    continue;
                }

                let exe_path = rs_path.with_extension("");
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
            } else {
                println!("Generated code:\n{}\n", code);
            }
        }
    } else {
        let code = agent.generate_code(&prompt, tokens, temperature, top_k, top_p).await?;

        if execute {
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

            let exe_path = rs_path.with_extension("");
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
        } else {
            println!("Generated code:\n{}\n", code);
        }
    }

    Ok(())
}
