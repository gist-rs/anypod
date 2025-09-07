use anyhow::{Context, Result};
use google_youtube3::api::{
    PlaylistItem, PlaylistItemSnippet, ResourceId, Video, VideoSnippet, VideoStatus,
};
use google_youtube3::yup_oauth2::InstalledFlowAuthenticator;
use google_youtube3::yup_oauth2::{read_application_secret, InstalledFlowReturnMethod};
use google_youtube3::{hyper_rustls, YouTube};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use std::default::Default;
use std::fs as std_fs;
use std::path::PathBuf;
use tokio::fs;

const CONFIG_DIR_NAME: &str = "anypod";
const CLIENT_SECRET_FILE: &str = "client_secrets.json";
const TOKEN_CACHE_FILE: &str = "token.json";

/// Establishes an authenticated connection to the YouTube API.
/// Handles the OAuth2.0 flow, including token caching and refreshing.
pub async fn authenticate() -> Result<YouTube<hyper_rustls::HttpsConnector<HttpConnector>>> {
    let config_dir = dirs::config_dir()
        .context("Failed to find user's config directory")?
        .join(CONFIG_DIR_NAME);

    fs::create_dir_all(&config_dir).await.with_context(|| {
        format!(
            "Failed to create config directory at {}",
            config_dir.display()
        )
    })?;

    let secret_path = config_dir.join(CLIENT_SECRET_FILE);
    if !secret_path.exists() {
        anyhow::bail!(
            "YouTube client secrets file not found. Please place it at: {}",
            secret_path.display()
        );
    }

    let app_secret = read_application_secret(secret_path)
        .await
        .context("Failed to read client secrets file")?;

    let token_cache_path = config_dir.join(TOKEN_CACHE_FILE);

    let auth =
        InstalledFlowAuthenticator::builder(app_secret, InstalledFlowReturnMethod::HTTPRedirect)
            .persist_tokens_to_disk(token_cache_path)
            .build()
            .await
            .context("Failed to create YouTube authenticator")?;

    let scopes = &[
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube",
    ];
    // This will trigger the auth flow if needed and cache the token within the authenticator.
    let _token = auth.token(scopes).await.context(
        "Failed to get YouTube authorization token. Please follow the browser instructions.",
    )?;

    let hub = YouTube::new(
        Client::builder(TokioExecutor::new()).build(
            hyper_rustls::HttpsConnectorBuilder::new()
                .with_native_roots()
                .context("Failed to find native root certificates")?
                .https_or_http()
                .enable_http1()
                .build(),
        ),
        auth,
    );
    Ok(hub)
}

/// Uploads a video to YouTube.
pub async fn upload_video(
    hub: &YouTube<hyper_rustls::HttpsConnector<HttpConnector>>,
    video_path: &PathBuf,
    title: &str,
    description: &str,
) -> Result<String> {
    let video_file = std_fs::File::open(video_path)
        .with_context(|| format!("Failed to open video file: {}", video_path.display()))?;

    let video_snippet = VideoSnippet {
        title: Some(title.to_string()),
        description: Some(description.to_string()),
        ..Default::default()
    };

    let video_status = VideoStatus {
        privacy_status: Some("private".to_string()), // Start as private, can be changed later
        ..Default::default()
    };

    let video = Video {
        snippet: Some(video_snippet),
        status: Some(video_status),
        ..Default::default()
    };

    let req = hub
        .videos()
        .insert(video)
        .upload(video_file, "video/*".parse()?);

    println!("-> Starting video upload...");
    let (_response, uploaded_video) = req.await.context("Video upload failed")?;
    println!("-> Video uploaded successfully.");

    uploaded_video
        .id
        .context("YouTube did not return a video ID after upload")
}

/// Adds a video to a specific playlist.
pub async fn add_video_to_playlist(
    hub: &YouTube<hyper_rustls::HttpsConnector<HttpConnector>>,
    playlist_id: &str,
    video_id: &str,
) -> Result<()> {
    let resource_id = ResourceId {
        kind: Some("youtube#video".to_string()),
        video_id: Some(video_id.to_string()),
        ..Default::default()
    };

    let playlist_item_snippet = PlaylistItemSnippet {
        playlist_id: Some(playlist_id.to_string()),
        resource_id: Some(resource_id),
        ..Default::default()
    };

    let playlist_item = PlaylistItem {
        snippet: Some(playlist_item_snippet),
        ..Default::default()
    };

    hub.playlist_items()
        .insert(playlist_item)
        .doit()
        .await
        .with_context(|| format!("Failed to add video '{video_id}' to playlist '{playlist_id}'"))?;

    Ok(())
}

/// Uploads a custom thumbnail for a video.
pub async fn upload_thumbnail(
    hub: &YouTube<hyper_rustls::HttpsConnector<HttpConnector>>,
    video_id: &str,
    thumbnail_path: &PathBuf,
) -> Result<()> {
    let thumbnail_file = std_fs::File::open(thumbnail_path).with_context(|| {
        format!(
            "Failed to open thumbnail file: {}",
            thumbnail_path.display()
        )
    })?;

    println!("-> Uploading thumbnail...");
    let req = hub
        .thumbnails()
        .set(video_id)
        .upload(thumbnail_file, "image/*".parse()?);

    let upload_result = req.await;
    upload_result.context("Thumbnail upload failed")?;

    println!("-> Thumbnail uploaded successfully.");
    Ok(())
}
