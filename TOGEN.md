### Final Plan: The "TOGEN" AI Podcast Prompt Engineer

The application is a command-line tool that intelligently generates a customized, high-quality prompt file. This prompt is designed to be used in a separate LLM (like NotebookLM) to generate a podcast script.

---

#### Phase 1: Project Setup & CLI Definition

1.  **`Cargo.toml` Dependencies**:
    *   `tokio`: For the asynchronous runtime.
    *   `serde` & `serde_json`: For robust JSON serialization and deserialization of our component files.
    *   `clap`: For creating a clean and powerful command-line interface.
    *   `reqwest`: A reliable HTTP client for communicating with the local LLM API.
    *   `thiserror` & `anyhow``: For ergonomic and easy-to-manage error handling.

2.  **CLI Structure (`src/main.rs`)**:
    *   The CLI will accept the following arguments:
        *   `file_path`: Mandatory path to the source content file (e.g., `raw/news/some-ai-news.html`).
        *   `--output-dir`: An optional path to a directory for saving the generated prompt. Defaults to a new `./generated_prompts` directory.
        *   `--llm-url`: An optional URL for the local LLM endpoint, defaulting to `http://localhost:9090/prompt`.

---

#### Phase 2: Core Logic & Execution Flow

The tool will follow a "Reason -> Assemble -> Save" workflow.

1.  **Read Source File**: Read the full content of the file specified by `file_path`.

2.  **Intelligent Format Selection (Single LLM Call)**:
    *   The tool sends a lightweight "meta" prompt to the local LLM.
    *   **Prompt**: *"Analyze the following content and determine the best podcast format: 'news_summary', 'explainer', or 'paper_deep_dive'. Respond with only the chosen format name."*
    *   This determines which blueprint from the `prompt_components/formats/` directory to use.

3.  **Load All Prompt Components**:
    *   Based on the format chosen by the LLM, the application reads and deserializes all the required JSON files (`core`, `personas`, `segments`, `rules`, and the selected `format`) into Rust structs.

4.  **Final Prompt Assembly**:
    *   A builder function assembles the final prompt string. The structure will be `podcast_template + instructions + file_content` to properly prime the target LLM.
    *   The generated Markdown will be clean and token-efficient, using simple headers (e.g., `# Title`, `## Subtitle`).
    *   The final prompt will look like this:
        ```markdown
        # CORE IDENTITY
        Podcast Name: Noob Learning
        Slogan: Noob Learning: Let's vibe learning together!
        ...

        # HOST PERSONAS
        ...

        # GOLDEN RULES
        ...

        # EPISODE BLUEPRINT: AI News Summary
        Your goal is to summarize a recent AI news story...
        ## The News Hook (30s)
        ...

        ---
        # SOURCE CONTENT
        [The full content from the input file is placed here]
        ```

5.  **Save the Final Prompt**:
    *   The assembled prompt is saved to a new `.md` file.
    *   The filename will be descriptive, combining the source filename, the chosen format, and the current date, e.g., `some-ai-news-news_summary-prompt-2023-10-27.md`.
    *   The file will be saved in the specified output directory.
    *   The application will print a confirmation message to the console indicating where the file was saved.