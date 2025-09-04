use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

// --- LLM API Data Structures ---

#[derive(Debug, Deserialize)]
struct LlmResult {
    text: String,
}

#[derive(Debug, Deserialize)]
pub struct LlmResponse {
    result: LlmResult,
}

/// Calls the LLM to determine the podcast format.
pub async fn determine_format(client: &Client, llm_url: &str, content: &str) -> Result<String> {
    let prompt = format!(
        "Analyze the following content and determine the best podcast format from this list: 'news_summary', 'explainer', 'paper_deep_dive', 'open_source_summary'. Respond with only the chosen format name and nothing else.\n\n---\n\n{}",
        &content[..content.len().min(4000)]
    );

    let response: LlmResponse = client
        .post(llm_url)
        .json(&json!({ "prompt": prompt }))
        .send()
        .await
        .with_context(|| format!("Failed to send format selection request to LLM at {llm_url}"))?
        .json()
        .await
        .with_context(|| "Failed to parse format selection response from LLM".to_string())?;

    Ok(response.result.text.trim().to_string())
}

/// Calls the LLM to generate a YouTube description.
pub async fn generate_youtube_description(
    client: &Client,
    llm_url: &str,
    content: &str,
    slogan: &str,
) -> Result<String> {
    let prompt = format!(
        "Based on the following content, generate a concise and engaging YouTube video description in English. The description should:\n1. Start with a one-sentence hook that grabs attention.\n2. Summarize the key topics discussed in 2-3 bullet points, with each bullet point starting with a relevant emoji (e.g., 🚀, ✨, 🤖).\n3. Use the following slogan as the friendly closing sentence: \"{}\"\n4. Do NOT include hashtags or links.\n\n---\n\n{}",
        slogan,
        &content[..content.len().min(4000)]
    );

    let response: LlmResponse = client
        .post(llm_url)
        .json(&json!({ "prompt": prompt }))
        .send()
        .await
        .with_context(|| format!("Failed to send YouTube description request to LLM at {llm_url}"))?
        .json()
        .await
        .with_context(|| "Failed to parse YouTube description response from LLM".to_string())?;

    Ok(response.result.text.trim().to_string())
}
