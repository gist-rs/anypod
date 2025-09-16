use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

// --- LLM API Data Structures ---

#[derive(Debug, Deserialize)]
struct Message {
    content: String,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Debug, Deserialize)]
pub struct LlmResponse {
    choices: Vec<Choice>,
}

/// Calls the LLM to determine the podcast format.
pub async fn determine_format(client: &Client, llm_url: &str, content: &str) -> Result<String> {
    let system_prompt = "Analyze the following content and determine the best podcast format from this list: 'news_summary', 'explainer', 'paper_deep_dive', 'open_source_summary'. Respond with only the chosen format name and nothing else.";
    let user_prompt = &content[..content.len().min(4000)];

    let response: LlmResponse = client
        .post(llm_url)
        .json(&json!({
            "model": "qwen3-coder-30b-a3b-instruct-mlx",
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ],
            "temperature": 0.2,
            "max_tokens": 20,
            "stream": false
        }))
        .send()
        .await
        .with_context(|| format!("Failed to send format selection request to LLM at {llm_url}"))?
        .json()
        .await
        .with_context(|| "Failed to parse format selection response from LLM".to_string())?;

    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("LLM response did not contain any choices"))?;

    Ok(choice.message.content.trim().to_string())
}

/// Calls the LLM to generate a YouTube description.
pub async fn generate_youtube_description(
    client: &Client,
    llm_url: &str,
    content: &str,
    slogan: &str,
) -> Result<String> {
    let system_prompt = format!(
        "Based on the following content, generate a concise and engaging YouTube video description in English. The description should:\n1. Start with a one-sentence hook that grabs attention.\n2. Summarize the key topics discussed in 2-3 bullet points, with each bullet point starting with a relevant emoji (e.g., 🚀, ✨, 🤖).\n3. Use the following slogan as the friendly closing sentence: \"{slogan}\"\n4. Do NOT include hashtags or links."
    );
    let user_prompt = &content[..content.len().min(4000)];

    let response: LlmResponse = client
        .post(llm_url)
        .json(&json!({
            "model": "qwen3-coder-30b-a3b-instruct-mlx",
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ],
            "temperature": 0.7,
            "stream": false
        }))
        .send()
        .await
        .with_context(|| format!("Failed to send YouTube description request to LLM at {llm_url}"))?
        .json()
        .await
        .with_context(|| "Failed to parse YouTube description response from LLM".to_string())?;

    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("LLM response did not contain any choices"))?;

    Ok(choice.message.content.trim().to_string())
}
