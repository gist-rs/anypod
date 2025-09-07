use clap::Parser;
use std::path::PathBuf;

/// A CLI tool to intelligently generate podcast prompts for LLMs.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// Path to the source content file (e.g., a news article).
    #[arg(long)]
    pub file_path: PathBuf,

    /// Directory to save the generated prompt file.
    #[arg(long, default_value = "./generated_prompts")]
    pub output_dir: PathBuf,

    /// URL of the local LLM endpoint for format selection.
    #[arg(long, default_value = "http://localhost:9090/prompt")]
    pub llm_url: String,

    /// (Optional) Path to the video file to upload to YouTube.
    #[arg(long)]
    pub video_file: Option<PathBuf>,

    /// (Optional) YouTube playlist ID to add the video to.
    #[arg(long)]
    pub playlist_id: Option<String>,

    /// (Optional) Path to the thumbnail file to upload to YouTube.
    #[arg(long)]
    pub thumbnail_file: Option<PathBuf>,

    /// (Optional) Manually specify the YouTube video title.
    #[arg(long)]
    pub title: Option<String>,

    /// (Optional) Manually specify the YouTube video description.
    #[arg(long)]
    pub description: Option<String>,
}
