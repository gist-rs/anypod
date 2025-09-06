# Plan: YouTube Upload Feature

This document outlines the steps to add functionality for uploading a video and its generated description directly to YouTube.

## Phase 1: Core YouTube Upload Logic

### 1. New CLI Arguments
- Modify `src/cli.rs` to accept YouTube-related arguments.
- We need arguments for:
  - `--video-file`: Path to the video file to upload.
  - `--playlist-id`: The ID of the YouTube playlist to add the video to.
  - `--no-upload`: A flag to prevent uploading, allowing the script to only generate local files. This will be the default behavior to maintain existing functionality.

### 2. Create `youtube.rs` Module
- Create a new file `src/youtube.rs`.
- This module will encapsulate all logic for interacting with the YouTube API.
- It will handle authentication, token management, video upload, and adding to a playlist.

### 3. Implement OAuth 2.0 Authentication
- Add `oauth2` and `url` crates to `Cargo.toml`.
- In `youtube.rs`, implement a function to handle the Google OAuth 2.0 flow for a desktop application.
- The flow should:
  - Read `client_secrets.json` from the configuration directory.
  - Generate an authorization URL and prompt the user to visit it.
  - Listen on a local port for the redirect to get the authorization code.
  - Exchange the code for an access token and a refresh token.
  - Store the token (e.g., in `~/.config/anypod/token.json`) for future use.
  - Implement logic to automatically use the refresh token if the access token has expired.

### 4. Implement Video Upload
- In `youtube.rs`, create a function `upload_video`.
- This function will take the access token, video file path, title, and description as arguments.
- It will perform a `multipart/related` POST request to the YouTube Data API v3 `videos.insert` endpoint.
- It should return the ID of the newly uploaded video upon success.

### 5. Implement "Add to Playlist"
- In `youtube.rs`, create a function `add_video_to_playlist`.
- This function will take the access token, video ID, and playlist ID.
- It will perform a POST request to the `playlistItems.insert` endpoint.

## Phase 2: Integration and Refinement

### 1. Update `main.rs`
- Modify the `main` function to incorporate the new upload logic.
- After generating the description, check if the `--video-file` and `--playlist-id` arguments were provided.
- If they are, initiate the YouTube upload flow:
  1. Call the authentication function from `youtube.rs` to get a valid token.
  2. Call `upload_video` with the necessary details.
  3. Call `add_video_to_playlist` with the returned video ID.
  4. Print progress and success messages to the console.

### 2. Configuration Management
- Decide on a standard location for configuration files (`client_secrets.json`, `token.json`). A good choice would be `~/.config/anypod/`.
- Ensure the application creates this directory if it doesn't exist.

### 3. Error Handling
- Add robust error handling for all API interactions.
- Provide clear, user-friendly error messages for common issues like:
  - `client_secrets.json` not found.
  - Invalid token.
  - API quota errors.
  - File not found for video.

## Phase 3: Documentation

### 1. Update `README.md`
- Add a new section called "YouTube Upload".
- Detail the one-time setup process:
  - How to create a Google Cloud project.
  - How to enable the YouTube Data API v3.
  - How to create OAuth 2.0 credentials for a "Desktop app".
  - Where to save the downloaded `client_secrets.json`.
- Document the new CLI flags (`--video-file`, `--playlist-id`).
- Provide a complete example command for uploading a video.