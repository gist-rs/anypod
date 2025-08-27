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
}
