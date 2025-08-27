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

    // 2. Concurrently call LLM for format selection and YouTube description.
    println!("\n-> Asking LLM to choose format and generate YouTube description...");
    let client = Client::new();
    let (format_name_result, youtube_desc_result) = tokio::join!(
        determine_format(&client, &cli.llm_url, &source_content),
        generate_youtube_description(&client, &cli.llm_url, &source_content)
    );

    let format_name = format_name_result?;
    let mut youtube_desc = youtube_desc_result?;

    println!("-> LLM selected format: {format_name}");
    println!("-> YouTube description generated.");

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
    let source_filename = cli
        .file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("source");

    let base_filename = format!("{source_filename}-{format_name}");

    let prompt_filename = format!("{base_filename}-prompt-{timestamp}.md");
    let desc_filename = format!("{base_filename}-youtube-desc-{timestamp}.txt");

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
