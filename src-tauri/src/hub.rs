use crate::runtime::{model_directory, AppState, DownloadState};
use serde_json::{json, Value};
use std::{
    path::{Component, Path},
    process::Stdio,
    time::Duration,
};
use tokio::{io::AsyncReadExt, process::Command};

struct HfProgress {
    percent: Option<f64>,
    completed: Option<u64>,
    transferred: Option<u64>,
}

fn parse_hf_progress(line: &str) -> Option<HfProgress> {
    use std::sync::OnceLock;
    static PERCENT: OnceLock<regex::Regex> = OnceLock::new();
    static BYTES: OnceLock<regex::Regex> = OnceLock::new();
    let percent = PERCENT
        .get_or_init(|| regex::Regex::new(r"(\d+(?:\.\d+)?)%\|").unwrap())
        .captures(line)
        .and_then(|c| c[1].parse::<f64>().ok())
        .map(|p| p.clamp(0.0, 100.0));
    let transfer = line.contains(": downloading bytes:");
    if percent.is_none() && !transfer {
        return None;
    }
    let tail = line.rsplit('|').next()?;
    let bytes = BYTES
        .get_or_init(|| regex::Regex::new(r"^\s*(\d+(?:\.\d+)?)\s*([kKMGT]?)(i?B)?").unwrap())
        .captures(tail)
        .and_then(|c| {
            let number = c[1].parse::<f64>().ok()?;
            let power = match &c[2] {
                "k" | "K" => 1,
                "M" => 2,
                "G" => 3,
                "T" => 4,
                _ => 0,
            };
            let base: f64 = if c.get(3).is_some_and(|unit| unit.as_str().starts_with('i')) {
                1024.0
            } else {
                1000.0
            };
            Some((number * base.powi(power)).round() as u64)
        });
    Some(HfProgress {
        percent,
        completed: if transfer { None } else { bytes },
        transferred: if transfer { bytes } else { None },
    })
}

async fn track_hf_progress(mut stderr: tokio::process::ChildStderr, state: AppState) {
    let mut chunk = [0u8; 4096];
    let mut pending = Vec::new();
    loop {
        let count = match stderr.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(count) => count,
        };
        pending.extend_from_slice(&chunk[..count]);
        while let Some(end) = pending
            .iter()
            .position(|byte| *byte == b'\r' || *byte == b'\n')
        {
            let line: Vec<_> = pending.drain(..=end).collect();
            if let Some(progress) = parse_hf_progress(&String::from_utf8_lossy(&line)) {
                let mut download = state.download.lock().await;
                if let Some(percent) = progress.percent {
                    download.progress = Some(percent);
                }
                if let Some(bytes) = progress.completed {
                    download.downloaded_bytes = bytes.min(download.total_bytes.unwrap_or(u64::MAX));
                }
                if let Some(bytes) = progress.transferred {
                    download.transferred_bytes = bytes;
                }
                download.message = if download.progress.is_some_and(|p| p >= 100.0) {
                    "Finalizing downloaded file…".into()
                } else {
                    "Downloading with Hugging Face…".into()
                };
            }
        }
        // Ignore unstructured stderr; never surface credential-bearing CLI URLs.
        if pending.len() > 16_384 {
            pending.clear();
        }
    }
}

pub fn decision_metadata(model: &Value) -> bool {
    let tags = model["tags"].as_array();
    tags.is_some_and(|tags| {
        tags.iter().filter_map(Value::as_str).any(|tag| {
            matches!(
                tag,
                "decision" | "decision-model" | "decision-models" | "system-one" | "systemone"
            )
        })
    }) || model["config"]["architectures"]
        .as_array()
        .is_some_and(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .any(|a| a.to_lowercase().contains("decision") || a.starts_with("D1"))
        })
}

fn valid_repo(repo: &str) -> bool {
    let parts: Vec<_> = repo.split('/').collect();
    parts.len() == 2
        && parts.iter().all(|p| {
            !p.is_empty()
                && *p != "."
                && *p != ".."
                && p.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
        })
}

async fn hf_token(state: &AppState) -> Option<String> {
    if let Some(token) = state.token.lock().await.clone() {
        return Some(token);
    }
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.trim().is_empty() {
            return Some(token.trim().into());
        }
    }
    let token_path = std::env::var_os("HF_TOKEN_PATH")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            let base = std::env::var_os("HF_HOME")
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    std::env::var_os("XDG_CACHE_HOME")
                        .map(std::path::PathBuf::from)
                        .or_else(|| std::env::var_os("HOME").map(|p| Path::new(&p).join(".cache")))
                        .map(|p| p.join("huggingface"))
                });
            base.map(|p| p.join("token"))
        });
    if let Some(path) = token_path {
        if let Ok(token) = tokio::fs::read_to_string(path).await {
            if !token.trim().is_empty() {
                return Some(token.trim().into());
            }
        }
    }
    None
}

async fn hub_get(state: &AppState, url: &str) -> Result<Value, String> {
    let mut request = state.client.get(url).timeout(Duration::from_secs(30));
    if let Some(token) = hf_token(state).await {
        request = request.bearer_auth(token);
    }
    let response = request
        .send()
        .await
        .map_err(|e| format!("Hugging Face connection failed: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!(
            "Hugging Face returned {status}. Check the repo, model access, and HF login."
        ));
    }
    response.json().await.map_err(|e| e.to_string())
}

async fn metadata(state: &AppState, repo: &str) -> Result<Value, String> {
    if !valid_repo(repo) {
        return Err("Use a Hugging Face repo ID in owner/model format.".into());
    }
    let model = hub_get(
        state,
        &format!("https://huggingface.co/api/models/{repo}?blobs=true"),
    )
    .await?;
    if !decision_metadata(&model) {
        return Err("This repository does not advertise a native decision model in its Hub tags or architecture. General chat models are excluded. Try its decision-tagged GGUF repo.".into());
    }
    Ok(model)
}

#[tauri::command]
pub async fn local_models(state: tauri::State<'_, AppState>) -> Result<Value, String> {
    list_local_models(state.inner()).await
}

async fn list_local_models(state: &AppState) -> Result<Value, String> {
    let settings = state.settings.lock().await.clone();
    let base = model_directory(&settings, &state.data_dir);
    let original = state.data_dir.join("models");
    tokio::task::spawn_blocking(move || {
        let mut files = Vec::new();
        let mut pending = vec![(base, 0)];
        if pending[0].0 != original { pending.push((original, 0)); }
        while let Some((dir, depth)) = pending.pop() {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.file_name().is_some_and(|s| s.to_string_lossy().starts_with('.')) { continue; }
                    if let Ok(kind) = entry.file_type() {
                        if kind.is_dir() && depth < 5 { pending.push((path, depth + 1)); }
                        else if kind.is_file() && path.extension().is_some_and(|e| e == "gguf") {
                            files.push(json!({"path": path, "name": entry.file_name().to_string_lossy(), "size": entry.metadata().ok().map(|m| m.len())}));
                        }
                    }
                }
            }
        }
        files.sort_by_key(|f| f["path"].as_str().unwrap_or_default().to_owned());
        Value::Array(files)
    }).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn set_hf_token(token: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    *state.token.lock().await = if token.trim().is_empty() {
        None
    } else {
        Some(token.trim().to_owned())
    };
    Ok(())
}

#[tauri::command]
pub async fn hf_session(state: tauri::State<'_, AppState>) -> Result<Value, String> {
    let token = hf_token(&state).await;
    let account = if token.is_some() {
        hub_get(&state, "https://huggingface.co/api/whoami-v2")
            .await
            .ok()
            .and_then(|v| v["name"].as_str().map(str::to_owned))
    } else {
        None
    };
    let hf = state.settings.lock().await.hf_path.clone();
    let cli = tokio::time::timeout(
        Duration::from_secs(8),
        Command::new(hf).arg("--version").output(),
    )
    .await
    .ok()
    .and_then(Result::ok)
    .is_some_and(|o| o.status.success());
    Ok(
        json!({"authenticated": account.is_some(), "account": account, "cli_available": cli, "token_present": token.is_some()}),
    )
}

#[tauri::command]
pub async fn search_models(
    query: String,
    state: tauri::State<'_, AppState>,
) -> Result<Value, String> {
    let mut found = Vec::new();
    for tag in ["decision", "decision-model", "system-one"] {
        let url = reqwest::Url::parse_with_params(
            "https://huggingface.co/api/models",
            &[
                ("search", query.as_str()),
                ("filter", tag),
                ("limit", "40"),
                ("sort", "lastModified"),
                ("direction", "-1"),
                ("full", "true"),
            ],
        )
        .map_err(|e| e.to_string())?;
        let models = hub_get(&state, url.as_str()).await?;
        for model in models
            .as_array()
            .ok_or("Unexpected Hub response")?
            .iter()
            .filter(|m| decision_metadata(m))
        {
            if !found.iter().any(|m: &Value| m["id"] == model["id"]) {
                found.push(json!({"id": model["id"], "downloads": model["downloads"], "likes": model["likes"], "tags": model["tags"]}));
            }
        }
    }
    Ok(Value::Array(found))
}

#[tauri::command]
pub async fn inspect_model(
    repo: String,
    state: tauri::State<'_, AppState>,
) -> Result<Value, String> {
    let model = metadata(&state, &repo).await?;
    let files: Vec<Value> = model["siblings"].as_array().into_iter().flatten().filter(|f| f["rfilename"].as_str().is_some_and(|s| s.ends_with(".gguf"))).map(|f| json!({"name": f["rfilename"], "size": f["size"], "projector": f["rfilename"].as_str().unwrap_or_default().to_lowercase().contains("mmproj")})).collect();
    Ok(json!({"repo": repo, "sha": model["sha"], "files": files, "tags": model["tags"]}))
}

#[tauri::command]
pub async fn download_model(
    repo: String,
    filename: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    if !filename.ends_with(".gguf")
        || Path::new(&filename)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err("Choose a relative GGUF filename from the repository.".into());
    }
    let mut download = state.download.lock().await;
    if download.active {
        return Err("A model download is already in progress.".into());
    }
    let model = metadata(&state, &repo).await?;
    if !model["siblings"]
        .as_array()
        .is_some_and(|a| a.iter().any(|f| f["rfilename"] == filename))
    {
        return Err("This file is not present in the selected repository.".into());
    }
    let revision = model["sha"]
        .as_str()
        .ok_or("Hub did not return a revision")?
        .to_owned();
    let directory = model_directory(&*state.settings.lock().await, &state.data_dir)
        .join(repo.replace('/', "--"));
    tokio::fs::create_dir_all(&directory)
        .await
        .map_err(|e| e.to_string())?;
    let destination = directory.join(&filename).to_string_lossy().into_owned();
    let mut command = Command::new(state.settings.lock().await.hf_path.clone());
    command
        .args([
            "download",
            &repo,
            &filename,
            "--revision",
            &revision,
            "--local-dir",
        ])
        .arg(&directory);
    if let Some(token) = hf_token(&state).await {
        command.env("HF_TOKEN", token);
    }
    command
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "0")
        .env("TQDM_POSITION", "-1")
        .env("TQDM_MININTERVAL", "0.2");
    let mut child = command.stdout(Stdio::null()).stderr(Stdio::piped()).kill_on_drop(true).spawn().map_err(|e| format!("Cannot start hf: {e}. Install the Hugging Face CLI or configure its executable path."))?;
    *download = DownloadState {
        active: true,
        repo: repo.clone(),
        file: filename.clone(),
        message: "Downloading with hf; resumable files are kept on disk…".into(),
        total_bytes: model["siblings"]
            .as_array()
            .and_then(|files| files.iter().find(|file| file["rfilename"] == filename))
            .and_then(|file| file["size"].as_u64()),
        ..Default::default()
    };
    drop(download);
    state
        .log(format!(
            "Downloading {repo}/{filename} at {}",
            &revision[..revision.len().min(12)]
        ))
        .await;
    let state = state.inner().clone();
    tokio::spawn(async move {
        let stderr = child.stderr.take();
        let progress_state = state.clone();
        let reader = tokio::spawn(async move {
            if let Some(stderr) = stderr {
                track_hf_progress(stderr, progress_state).await;
            }
        });
        let result = child.wait().await;
        let _ = reader.await;
        let success = result.is_ok_and(|s| s.success()) && Path::new(&destination).is_file();
        let mut download = state.download.lock().await;
        download.active = false;
        if success {
            let size = tokio::fs::metadata(&destination)
                .await
                .ok()
                .map(|metadata| metadata.len());
            download.downloaded_bytes = size.unwrap_or(download.downloaded_bytes);
            download.total_bytes = size.or(download.total_bytes);
            download.progress = Some(100.0);
            download.path = Some(destination);
            download.message = "Download complete. Ready to load.".into();
            state.log(format!("Download complete: {filename}")).await;
        } else {
            download.error = Some("hf download failed. Check your connection, disk space, and repository access; retry resumes the download.".into());
            download.message = "Download failed; partial files were preserved.".into();
            state
                .log(format!("Download failed: {repo}/{filename}"))
                .await;
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hf_progress_tracks_reconstruction_and_network_bytes_separately() {
        let progress =
            parse_hf_progress("Julia.gguf: reconstructing file:  40%|████ | 67.8MB / 168MB")
                .unwrap();
        assert_eq!(progress.percent, Some(40.0));
        assert_eq!(progress.completed, Some(67_800_000));
        assert_eq!(progress.transferred, None);
        let transfer =
            parse_hf_progress("Julia.gguf: downloading bytes: ███ | 12.3MB, 2.1MB/s").unwrap();
        assert_eq!(transfer.transferred, Some(12_300_000));
        assert_eq!(transfer.percent, None);
        assert!(parse_hf_progress("Warning: a token is not configured").is_none());
    }

    #[tokio::test]
    async fn progress_reader_handles_carriage_return_frames() {
        let base = std::env::temp_dir().join(format!("noullama-progress-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        state.download.lock().await.total_bytes = Some(168_000_000);
        // Exercise the actual stderr framing path, not a file-size approximation.
        let mut child = Command::new("sh").args(["-c", "printf 'model: downloading bytes: | 12.3MB, 2MB/s\\rmodel: reconstructing file: 40%%|#### | 67.8MB / 168MB\\n' >&2"])
            .stderr(Stdio::piped()).stdout(Stdio::null()).spawn().unwrap();
        track_hf_progress(child.stderr.take().unwrap(), state.clone()).await;
        child.wait().await.unwrap();
        let download = state.download.lock().await.clone();
        assert_eq!(download.progress, Some(40.0));
        assert_eq!(download.transferred_bytes, 12_300_000);
        assert_eq!(download.downloaded_bytes, 67_800_000);
        std::fs::remove_dir_all(base).unwrap();
    }
    #[test]
    fn metadata_excludes_chat_and_includes_unlisted_decision_models() {
        assert!(decision_metadata(&json!({"tags":["gguf","decision"]})));
        assert!(decision_metadata(
            &json!({"config":{"architectures":["D1Model"]}})
        ));
        assert!(!decision_metadata(
            &json!({"id":"LiquidAI/LFM3-1.7B", "tags":["text-generation"]})
        ));
        assert!(!valid_repo("../escape"));
        assert!(!valid_repo("org/repo/extra"));
        assert!(valid_repo("LiquidAI/d1-2B-GGUF"));
    }

    #[tokio::test]
    async fn custom_library_also_retains_original_downloads() {
        let base = std::env::temp_dir().join(format!("noullama-library-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let original = base.join("models/repo");
        let custom = base.join("custom directory/repo");
        for dir in [&original, &custom] {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join("model.gguf"), "test weights").unwrap();
            std::fs::write(dir.join("partial.gguf.incomplete"), "partial").unwrap();
        }
        state.settings.lock().await.download_dir =
            base.join("custom directory").to_string_lossy().into_owned();
        let files = list_local_models(&state).await.unwrap();
        assert_eq!(files.as_array().unwrap().len(), 2);
        assert!(original.join("model.gguf").exists());
        std::fs::remove_dir_all(base).unwrap();
    }
}
