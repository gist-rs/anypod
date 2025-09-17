# ANYPOD

1. Paper to podcast. // weekly
2. News to podcast. // daily
3. Open source to podcast. // weekly

## Title
```bash
export TITLE=2025-09-17
```

## Convert
```bash
ffmpeg -loop 1 -i assets/${TITLE}.png -i "bin/${TITLE}.audio.mp4" -c:v libx264 -tune stillimage -c:a copy -pix_fmt yuv420p -shortest bin/${TITLE}.mp4
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
cargo run -- --file-path "raw/open_sources/${TITLE}.md"
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

### 3. (Optional) YouTube Upload

The tool can also upload a video file directly to YouTube and add it to a specified playlist.

#### One-Time Setup

Before you can upload, you need to authorize the application to access your YouTube account.

1.  **Create a Google Cloud Project**: Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a new project.
2.  **Enable the API**: In your project, enable the "YouTube Data API v3".
3.  **Create OAuth Credentials**:
    -   Go to "Credentials" -> "Create Credentials" -> "OAuth client ID".
    -   Select **"Desktop app"** as the application type.
    -   Download the JSON file.
4.  **Save Credentials**: Rename the downloaded file to `client_secrets.json` and place it in a configuration directory, for example: `~/.config/anypod/client_secrets.json`. The application will look for it there.

The first time you run an upload command, your web browser will open, asking you to log in to your Google account and grant permission. After you approve, a `token.json` file will be saved in the same directory as your secrets file, allowing for automatic authentication in the future.

#### Upload Command

To upload a video, provide the path to the video file and the target playlist ID.

```bash
cargo run -- \
  --file-path "raw/news/${TITLE}.md" \
  --video-file "bin/${TITLE}.mp4" \
  --playlist-id "YOUR_YOUTUBE_PLAYLIST_ID"
```

The application will first generate the description file locally and then proceed with the upload.

### 4. Building for Production

To build a release binary, run:

```bash
cargo build --release
```

The optimized executable will be located at `anypod/target/release/anypod`. You can then run it directly:

```bash
./target/release/anypod --file-path "raw/news/${TITLE}"
```
