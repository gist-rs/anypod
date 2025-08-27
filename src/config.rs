use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;
use tokio::fs;

// --- Data Structures for JSON Components ---

#[derive(Debug, Deserialize)]
pub struct SocialLinks {
    pub facebook: String,
}

#[derive(Debug, Deserialize)]
pub struct CoreConfig {
    pub podcast_name: String,
    pub slogan: String,
    pub target_audience: String,
    pub social_links: SocialLinks,
}

#[derive(Debug, Deserialize)]
pub struct Persona {
    pub name: String,
    pub gender: String,
    pub role: String,
    pub persona: String,
    pub common_phrases: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct BlueprintStep {
    pub step: u8,
    pub name: String,
    pub technique: String,
}

#[derive(Debug, Deserialize)]
pub struct Format {
    pub format_name: String,
    pub goal: String,
    pub blueprint: Vec<BlueprintStep>,
}

#[derive(Debug, Deserialize)]
pub struct IntroSegment {
    pub base_script: Vec<String>,
    pub dynamic_element_prompt: String,
    pub topic_transition_line: String,
}

#[derive(Debug, Deserialize)]
pub struct OutroSegment {
    pub summary_prompt: String,
    pub call_to_action: String,
    pub sign_off: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct GeneralRules {
    pub language: String,
    pub acronyms: String,
    pub tone: String,
    pub style: String,
}

// --- Prompt Assembly Structures ---

// A container for all the deserialized prompt components.
#[derive(Debug)]
pub struct PromptComponents {
    pub core: CoreConfig,
    pub ying: Persona,
    pub katopz: Persona,
    pub format: Format,
    pub intro: IntroSegment,
    pub outro: OutroSegment,
    pub rules: GeneralRules,
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
pub async fn load_components(format_name: &str) -> Result<PromptComponents> {
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
