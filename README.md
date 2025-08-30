# ANYPOD

1. Paper to podcast. // weekly
2. News to podcast. // daily

## Title
```bash
export TITLE=2025-08-30
```

## Convert
```bash
ffmpeg -loop 1 -i assets/${TITLE}.png -i "bin/${TITLE}.m4a" -c:v libx264 -tune stillimage -c:a copy -pix_fmt yuv420p -shortest bin/${TITLE}.mp4
```

---

## How to Run the Prompt Generator

This project includes a command-line tool to intelligently generate podcast prompts from source files.

### 1. Prerequisites

-   Ensure you have Rust and Cargo installed (`rustup`).
-   You need a local Large Language Model (LLM) running and accessible at the specified URL (default is `http://localhost:9090/prompt`). This endpoint is used to intelligently select the podcast format.

### 2. Running the Application

You can run the application directly using Cargo.

#### Basic Usage

Pass the path to your source content file using the `file-path` argument. Note the `--` which separates the cargo arguments from your application's arguments.

```bash
cargo run -- --file-path path/to/your/source-file.html
```

**Example:**
```bash
cargo run -- --file-path "raw/news/${TITLE}.md"
cargo run -- --file-path "raw/papers/${TITLE}.md"
```

The generated prompt will be saved in the `./generated_prompts` directory by default.

#### Specifying an Output Directory

You can specify a different directory to save the prompt files using the `--output-dir` flag.

```bash
cargo run -- --file-path "raw/news/${TITLE}" --output-dir ./my_prompts
```

#### Specifying the LLM URL

If your local LLM is running on a different address, use the `--llm-url` flag.

```bash
cargo run -- --file-path "raw/news/${TITLE}" --llm-url http://localhost:9090/prompt
```

### 3. Building for Production

To build a release binary, run:

```bash
cargo build --release
```

The optimized executable will be located at `anypod/target/release/anypod`. You can then run it directly:

```bash
./target/release/anypod --file-path "raw/news/${TITLE}"
```
