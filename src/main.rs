mod cli;
mod config;
mod llm;
mod prompt;

use anyhow::{Context, Result};
use chrono::Local;
use clap::Parser;
use reqwest::Client;
use tokio::fs;

use crate::cli::Cli;
use crate::config::load_components;
use crate::llm::{determine_format, generate_youtube_description};
use crate::prompt::assemble_prompt;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("Initializing prompt generation...");
    println!("-> Source File: {}", cli.file_path.display());
    println!("-> Output Dir:  {}", cli.output_dir.display());
    println!("-> LLM URL:     {}", cli.llm_url);

    // --- Execution Flow ---

    // 1. Create output directory and read source file
    fs::create_dir_all(&cli.output_dir).await.with_context(|| {
        format!(
            "Failed to create output directory: {}",
            cli.output_dir.display()
        )
    })?;

    let source_content = fs::read_to_string(&cli.file_path)
        .await
        .with_context(|| format!("Failed to read source file: {}", cli.file_path.display()))?;

    // 2. Determine format based on file path hint or LLM, then generate YouTube description
    let client = Client::new();
    println!("\n-> Determining podcast format...");
    let file_path_str = cli.file_path.to_string_lossy();
    let format_name = if file_path_str.contains("news") {
        "news_summary".to_string()
    } else if file_path_str.contains("paper") {
        "paper_deep_dive".to_string()
    } else {
        determine_format(&client, &cli.llm_url, &source_content).await?
    };
    println!("-> Selected format: {format_name}");

    println!("-> Generating YouTube description...");
    let mut youtube_desc =
        generate_youtube_description(&client, &cli.llm_url, &source_content).await?;
    println!("-> YouTube description generated.");

    let source_filename = cli
        .file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("source");

    // Prepend title to youtube description
    let title = match format_name.as_str() {
        "news_summary" => format!("Noob Vibe News {source_filename}\n\n"),
        "paper_deep_dive" => format!("Noob Vibe Paper: {source_filename}\n\n"),
        _ => String::new(),
    };
    youtube_desc.insert_str(0, &title);

    // 3. Load all necessary JSON components
    println!("-> Loading all prompt components...");
    let components = load_components(&format_name).await?;
    println!("-> All components loaded successfully.");

    // 4. Assemble the final prompt string.
    let final_prompt = assemble_prompt(&components);
    println!("-> Final prompt assembled successfully.");

    // 5. Append social link to the description.
    if !components.core.social_links.facebook.is_empty() {
        youtube_desc.push_str("\n\n---\n\n");
        youtube_desc.push_str(&components.core.social_links.facebook);
    }

    // 6. Generate descriptive filenames and save both files.
    let timestamp = Local::now().format("%Y-%m-%d");

    let prompt_filename = format!("{timestamp}-{source_filename}-{format_name}-prompt.md");
    let desc_filename = format!("{timestamp}-{source_filename}-{format_name}-youtube-desc.txt");

    let prompt_path = cli.output_dir.join(&prompt_filename);
    let desc_path = cli.output_dir.join(&desc_filename);

    fs::write(&prompt_path, &final_prompt)
        .await
        .with_context(|| format!("Failed to write final prompt to {}", prompt_path.display()))?;

    fs::write(&desc_path, &youtube_desc)
        .await
        .with_context(|| {
            format!(
                "Failed to write YouTube description to {}",
                desc_path.display()
            )
        })?;

    println!(
        "\n✅ Success! Files saved:\n- Prompt: {}\n- Description: {}",
        prompt_path.display(),
        desc_path.display()
    );

    Ok(())
}
