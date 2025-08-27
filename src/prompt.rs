use crate::config::PromptComponents;

/// Assembles the final prompt string from all loaded components.
pub fn assemble_prompt(components: &PromptComponents) -> String {
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
