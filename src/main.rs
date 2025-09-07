mod cli;
mod config;
mod llm;
mod prompt;
mod youtube;

use anyhow::{Context, Result};
use chrono::Local;
use clap::Parser;
use reqwest::Client;
use tokio::fs;

use crate::cli::Cli;
use crate::config::load_components;
use crate::llm::{determine_format, generate_youtube_description};
use crate::prompt::assemble_prompt;
use crate::youtube::{add_video_to_playlist, authenticate, upload_thumbnail, upload_video};

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
    } else if file_path_str.contains("open_sources") {
        "open_source_summary".to_string()
    } else {
        determine_format(&client, &cli.llm_url, &source_content).await?
    };
    println!("-> Selected format: {format_name}");

    // 3. Load all necessary JSON components
    println!("-> Loading all prompt components...");
    let components = load_components(&format_name).await?;
    println!("-> All components loaded successfully.");

    let source_filename = cli
        .file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("source");

    // 4. Determine Title: Use manual title if provided, otherwise generate it.
    let title = if let Some(manual_title) = cli.title {
        println!("-> Using manual title.");
        manual_title
    } else {
        match format_name.as_str() {
            "news_summary" => format!("Noob Vibe News {source_filename}"),
            "paper_deep_dive" => format!("Noob Vibe Paper: {source_filename}"),
            "open_source_summary" => format!("Noob Vibe Open Source: {source_filename}"),
            _ => source_filename.to_string(),
        }
    };

    // 5. Determine Description: Use manual description if provided, otherwise generate it.
    let base_description = if let Some(manual_desc) = cli.description {
        println!("-> Using manual description.");
        manual_desc
    } else {
        println!("-> Generating YouTube description...");
        let desc = generate_youtube_description(
            &client,
            &cli.llm_url,
            &source_content,
            &components.core.slogan,
        )
        .await?;
        println!("-> YouTube description generated.");
        desc
    };

    // 6. Assemble the final prompt string.
    let final_prompt = assemble_prompt(&components);
    println!("-> Final prompt assembled successfully.");

    // 7. Prepare descriptions for file and upload.
    let mut description_for_file = base_description.clone();
    if !components.core.social_links.facebook.is_empty() {
        description_for_file.push_str("\n\n---\n\n");
        description_for_file.push_str(&components.core.social_links.facebook);
    }
    let youtube_file_content = format!("{title}\n\n{description_for_file}");

    // 8. Generate descriptive filenames and save both files.
    let timestamp = Local::now().format("%Y-%m-%d");
    let prompt_filename = format!("{timestamp}-{source_filename}-{format_name}-prompt.md");
    let desc_filename = format!("{timestamp}-{source_filename}-{format_name}-youtube-desc.txt");
    let prompt_path = cli.output_dir.join(&prompt_filename);
    let desc_path = cli.output_dir.join(&desc_filename);

    fs::write(&prompt_path, &final_prompt)
        .await
        .with_context(|| format!("Failed to write final prompt to {}", prompt_path.display()))?;

    fs::write(&desc_path, &youtube_file_content)
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

    // --- (Optional) YouTube Upload ---
    if let (Some(video_file), Some(playlist_id)) = (&cli.video_file, &cli.playlist_id) {
        println!("\n-> Starting YouTube upload process...");

        // 1. Authenticate
        let hub = authenticate()
            .await
            .context("YouTube authentication failed")?;
        println!("-> YouTube authentication successful.");

        // 2. Upload Video
        let video_id = upload_video(&hub, video_file, &title, &base_description)
            .await
            .context("YouTube video upload failed")?;

        // 3. Add to Playlist
        add_video_to_playlist(&hub, playlist_id, &video_id)
            .await
            .context("Failed to add video to YouTube playlist")?;
        println!(
            "-> Successfully added video to playlist: https://www.youtube.com/playlist?list={playlist_id}"
        );

        // 4. (Optional) Upload Thumbnail
        if let Some(thumbnail_file) = &cli.thumbnail_file {
            upload_thumbnail(&hub, &video_id, thumbnail_file)
                .await
                .context("Failed to upload YouTube thumbnail")?;
        }

        println!("✅ YouTube upload complete.");
    }

    Ok(())
}
