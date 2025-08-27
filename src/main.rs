use anyhow::{Context, Result};
use chrono::Local;
use clap::Parser;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;
use tokio::fs;

// --- LLM API Data Structures ---

#[derive(Debug, Deserialize)]
struct LlmResult {
    text: String,
}

#[derive(Debug, Deserialize)]
struct LlmResponse {
    result: LlmResult,
}

// --- Data Structures for JSON Components ---

#[derive(Debug, Deserialize)]
struct SocialLinks {
    facebook: String,
}

#[derive(Debug, Deserialize)]
struct CoreConfig {
    podcast_name: String,
    slogan: String,
    target_audience: String,
    social_links: SocialLinks,
}

#[derive(Debug, Deserialize)]
struct Persona {
    name: String,
    gender: String,
    role: String,
    persona: String,
    common_phrases: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct BlueprintStep {
    step: u8,
    name: String,
    technique: String,
}

#[derive(Debug, Deserialize)]
struct Format {
    format_name: String,
    goal: String,
    blueprint: Vec<BlueprintStep>,
}

#[derive(Debug, Deserialize)]
struct IntroSegment {
    base_script: Vec<String>,
    dynamic_element_prompt: String,
    topic_transition_line: String,
}

#[derive(Debug, Deserialize)]
struct OutroSegment {
    summary_prompt: String,
    call_to_action: String,
    sign_off: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct GeneralRules {
    language: String,
    acronyms: String,
    tone: String,
    style: String,
}

// --- Prompt Assembly Structures ---

// A container for all the deserialized prompt components.
#[derive(Debug)]
struct PromptComponents {
    core: CoreConfig,
    ying: Persona,
    katopz: Persona,
    format: Format,
    intro: IntroSegment,
    outro: OutroSegment,
    rules: GeneralRules,
}

// Generic helper function to load and deserialize a JSON file into a given type.
async fn load_and_deserialize<T: for<'de> Deserialize<'de>>(path: PathBuf) -> Result<T> {
    let content = fs::read_to_string(&path)
        .await
        .with_context(|| format!("Failed to read component file: {}", path.display()))?;
    let component: T = serde_json::from_str(&content)
        .with_context(|| format!("Failed to deserialize component file: {}", path.display()))?;
    Ok(component)
}

/// Loads all JSON components from disk concurrently.
async fn load_components(format_name: &str) -> Result<PromptComponents> {
    let base_path = PathBuf::from("./prompt_components");

    // Define paths for each component
    let core_path = base_path.join("core/noob_learning.json");
    let ying_path = base_path.join("personas/ying.json");
    let katopz_path = base_path.join("personas/katopz.json");
    let intro_filename = if format_name == "news_summary" {
        "segments/intro_news.json"
    } else {
        "segments/intro.json"
    };
    let intro_path = base_path.join(intro_filename);
    let outro_path = base_path.join("segments/outro.json");
    let rules_path = base_path.join("rules/general.json");
    let format_path = base_path
        .join("formats")
        .join(format!("{format_name}.json"));

    // Create async tasks for loading each component
    let core_task = load_and_deserialize::<CoreConfig>(core_path);
    let ying_task = load_and_deserialize::<Persona>(ying_path);
    let katopz_task = load_and_deserialize::<Persona>(katopz_path);
    let intro_task = load_and_deserialize::<IntroSegment>(intro_path);
    let outro_task = load_and_deserialize::<OutroSegment>(outro_path);
    let rules_task = load_and_deserialize::<GeneralRules>(rules_path);
    let format_task = load_and_deserialize::<Format>(format_path);

    // Concurrently await all tasks
    let (core, ying, katopz, intro, outro, rules, format) = tokio::try_join!(
        core_task,
        ying_task,
        katopz_task,
        intro_task,
        outro_task,
        rules_task,
        format_task
    )?;

    Ok(PromptComponents {
        core,
        ying,
        katopz,
        format,
        intro,
        outro,
        rules,
    })
}

/// A CLI tool to intelligently generate podcast prompts for LLMs.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Path to the source content file (e.g., a news article).
    #[arg(long)]
    file_path: PathBuf,

    /// Directory to save the generated prompt file.
    #[arg(long, default_value = "./generated_prompts")]
    output_dir: PathBuf,

    /// URL of the local LLM endpoint for format selection.
    #[arg(long, default_value = "http://localhost:9090/prompt")]
    llm_url: String,
}

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

/// Calls the LLM to determine the podcast format.
async fn determine_format(client: &Client, llm_url: &str, content: &str) -> Result<String> {
    let prompt = format!(
        "Analyze the following content and determine the best podcast format from this list: 'news_summary', 'explainer', 'paper_deep_dive'. Respond with only the chosen format name and nothing else.\n\n---\n\n{}",
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
async fn generate_youtube_description(
    client: &Client,
    llm_url: &str,
    content: &str,
) -> Result<String> {
    let prompt = format!(
        "Based on the following content, generate a concise and engaging YouTube video description. The description should:\n1. Start with a one-sentence hook that grabs attention.\n2. Summarize the key topics discussed in 2-3 bullet points, with each bullet point starting with a relevant emoji (e.g., 🚀, ✨, 🤖).\n3. Include a friendly closing sentence.\n4. Do NOT include hashtags or links.\n\n---\n\n{}",
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

/// Assembles the final prompt string from all loaded components.
fn assemble_prompt(components: &PromptComponents) -> String {
    let mut prompt = String::new();

    // Core Identity
    prompt.push_str("# CORE IDENTITY\n");
    prompt.push_str(&format!("Podcast Name: {}\n", components.core.podcast_name));
    prompt.push_str(&format!("Slogan: {}\n", components.core.slogan));
    prompt.push_str(&format!(
        "Target Audience: {}\n\n",
        components.core.target_audience
    ));

    // Host Personas
    prompt.push_str("# HOST PERSONAS\n");
    let personas = [&components.ying, &components.katopz];
    for p in personas.iter() {
        prompt.push_str(&format!("## {} ({})\n", p.name, p.gender));
        prompt.push_str(&format!("- Role: {}\n", p.role));
        prompt.push_str(&format!("- Persona: {}\n", p.persona));
        prompt.push_str(&format!(
            "- Common Phrases: {}\n",
            p.common_phrases.join(", ")
        ));
    }
    prompt.push('\n');

    // Golden Rules
    prompt.push_str("# GOLDEN RULES\n");
    prompt.push_str(&format!("- Language: {}\n", components.rules.language));
    prompt.push_str(&format!("- Acronyms: {}\n", components.rules.acronyms));
    prompt.push_str(&format!("- Tone: {}\n", components.rules.tone));
    prompt.push_str(&format!("- Style: {}\n\n", components.rules.style));

    // Episode Blueprint
    prompt.push_str(&format!(
        "# EPISODE BLUEPRINT: {}\n",
        components.format.format_name
    ));
    prompt.push_str(&format!("Goal: {}\n\n", components.format.goal));

    prompt.push_str("## Intro\n");
    prompt.push_str(&format!("- {}\n", components.intro.dynamic_element_prompt));
    for line in &components.intro.base_script {
        prompt.push_str(&format!("- {line}\n"));
    }
    prompt.push_str(&format!("- {}\n\n", components.intro.topic_transition_line));

    prompt.push_str("## Main Content\n");
    for step in &components.format.blueprint {
        prompt.push_str(&format!("### Step {}: {}\n", step.step, step.name));
        prompt.push_str(&format!("- Technique: {}\n", step.technique));
    }
    prompt.push('\n');

    prompt.push_str("## Outro\n");
    prompt.push_str(&format!("- {}\n", components.outro.summary_prompt));
    prompt.push_str(&format!("- {}\n", components.outro.call_to_action));
    for line in &components.outro.sign_off {
        prompt.push_str(&format!("- {line}\n"));
    }
    prompt.push('\n');

    prompt
}
