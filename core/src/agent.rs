use super::inference::LlmInference;
use once_cell::sync::Lazy;
use regex::Regex;

use anyhow::Result as AnyhowResult;
use std::sync::Arc;

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
            "You are a helpful coding assistant. Generate Rust code for: {}",
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
            .or_else(|| {
                ANY_CODE_BLOCK_RE
                    .captures(&response)
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
    agent_path: std::path::PathBuf,
    prompt: String,
    tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> AnyhowResult<()> {
    let agent = Agent::new(agent_path).await?;
    let result = agent.generate_code(&prompt, tokens, temperature, top_k, top_p).await?;
    println!("{}", result);
    Ok(())
}
