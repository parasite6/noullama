use crate::runtime::{model_directory, AppState, DownloadState, Settings};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
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

fn redact_diagnostic(line: &str, secrets: &[String]) -> String {
    use std::sync::OnceLock;
    static SENSITIVE: OnceLock<regex::Regex> = OnceLock::new();
    static ANSI: OnceLock<regex::Regex> = OnceLock::new();
    let mut line = line.to_owned();
    for secret in secrets.iter().filter(|s| !s.is_empty()) {
        line = line.replace(secret, "[redacted]");
    }
    let line = ANSI
        .get_or_init(|| regex::Regex::new(r"\x1b\[[0-?]*[ -/]*[@-~]").unwrap())
        .replace_all(&line, "");
    let line = SENSITIVE
        .get_or_init(|| {
            regex::Regex::new(r"(?i)https?://[^\s]+|bearer\s+[^\s]+|hf_[a-z0-9_]+").unwrap()
        })
        .replace_all(&line, "[redacted]");
    line.chars()
        .filter(|c| !c.is_control())
        .take(512)
        .collect::<String>()
        .trim()
        .to_owned()
}

async fn handle_hf_line(
    line: &[u8],
    state: &AppState,
    secrets: &[String],
    tail: &mut VecDeque<String>,
) {
    let line = String::from_utf8_lossy(line);
    if let Some(progress) = parse_hf_progress(&line) {
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
    } else {
        let line = redact_diagnostic(&line, secrets);
        if !line.is_empty() {
            if tail.len() == 12 {
                tail.pop_front();
            }
            tail.push_back(line);
        }
    }
}

async fn track_hf_progress(
    mut stderr: tokio::process::ChildStderr,
    state: AppState,
    secrets: Vec<String>,
) -> VecDeque<String> {
    let mut chunk = [0u8; 4096];
    let mut pending = Vec::new();
    let mut tail = VecDeque::new();
    let mut oversized = false;
    loop {
        let count = match stderr.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(count) => count,
        };
        for byte in &chunk[..count] {
            if *byte == b'\r' || *byte == b'\n' {
                if !oversized {
                    handle_hf_line(&pending, &state, &secrets, &mut tail).await;
                }
                pending.clear();
                oversized = false;
            } else if !oversized {
                if pending.len() == 16_384 {
                    pending.clear();
                    oversized = true;
                } else {
                    pending.push(*byte);
                }
            }
        }
    }
    if !oversized && !pending.is_empty() {
        handle_hf_line(&pending, &state, &secrets, &mut tail).await;
    }
    tail
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
        && !repo.starts_with('-')
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
    let mut roots = vec![(base.clone(), !settings.download_dir.trim().is_empty())];
    if base != original {
        roots.push((original, false));
    }
    tokio::task::spawn_blocking(move || scan_local_models(roots, 20_000, Duration::from_secs(2)))
        .await
        .map_err(|e| e.to_string())
}

fn scan_local_models(
    roots: Vec<(std::path::PathBuf, bool)>,
    max_entries: usize,
    budget: Duration,
) -> Value {
    let started = std::time::Instant::now();
    let mut files = Vec::new();
    let mut warnings = Vec::new();
    let mut incomplete = false;
    let mut count = 0;
    let mut pending: Vec<_> = roots
        .into_iter()
        .map(|(path, required)| (path, 0, required))
        .collect();
    'scan: while let Some((dir, depth, required)) = pending.pop() {
        if count >= max_entries || started.elapsed() >= budget {
            incomplete = true;
            warnings.push("Model scan reached its entry or time limit. Choose a smaller model directory to see all files.".into());
            break;
        }
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if !required && error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                incomplete = true;
                if warnings.len() < 8 {
                    warnings.push(format!("Cannot scan {}: {error}", dir.display()));
                }
                continue;
            }
        };
        for entry in entries {
            if count >= max_entries || started.elapsed() >= budget {
                incomplete = true;
                warnings.push("Model scan reached its entry or time limit. Choose a smaller model directory to see all files.".into());
                break 'scan;
            }
            count += 1;
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    incomplete = true;
                    if warnings.len() < 8 {
                        warnings.push(format!(
                            "Cannot read an entry in {}: {error}",
                            dir.display()
                        ));
                    }
                    continue;
                }
            };
            let path = entry.path();
            if entry.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => {
                    if depth < 5 {
                        pending.push((path, depth + 1, true));
                    } else {
                        incomplete = true;
                        if warnings.len() < 8 {
                            warnings
                                .push(format!("Scan depth limit reached at {}", path.display()));
                        }
                    }
                }
                Ok(kind)
                    if kind.is_file()
                        && path
                            .extension()
                            .is_some_and(|e| e.eq_ignore_ascii_case("gguf")) =>
                {
                    files.push(json!({"path": path, "name": entry.file_name().to_string_lossy(), "size": entry.metadata().ok().map(|m| m.len())}));
                }
                Err(error) => {
                    incomplete = true;
                    if warnings.len() < 8 {
                        warnings.push(format!("Cannot inspect {}: {error}", path.display()));
                    }
                }
                _ => {}
            }
        }
    }
    files.sort_by_key(|f| f["path"].as_str().unwrap_or_default().to_owned());
    files.dedup_by(|a, b| a["path"] == b["path"]);
    json!({"models": files, "warnings": warnings, "incomplete": incomplete})
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
pub async fn author_avatar(
    author: String,
    pixels: Option<u32>,
    state: tauri::State<'_, AppState>,
) -> Result<Option<String>, String> {
    let source = lookup_author_avatar(state.inner(), &author).await?;
    let pixels = pixels.unwrap_or(64).clamp(24, 192);
    Ok(tokio::task::spawn_blocking(move || {
        source.map(|source| raster_avatar(&source, pixels).unwrap_or(source))
    })
    .await
    .map_err(|e| e.to_string())?)
}

fn raster_avatar(source: &str, pixels: u32) -> Option<String> {
    use base64::Engine;
    let encoded = source
        .strip_prefix("data:image/")?
        .split_once(";base64,")?
        .1;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .ok()?;
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .ok()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(4096);
    limits.max_image_height = Some(4096);
    limits.max_alloc = Some(64 * 1024 * 1024);
    reader.limits(limits);
    let resized =
        reader
            .decode()
            .ok()?
            .resize(pixels, pixels, image::imageops::FilterType::Lanczos3);
    let mut output = std::io::Cursor::new(Vec::new());
    resized
        .write_to(&mut output, image::ImageFormat::Png)
        .ok()?;
    Some(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(output.into_inner())
    ))
}

async fn original_avatar_data(state: &AppState, avatar: &str) -> Option<String> {
    use base64::Engine;
    let url = reqwest::Url::parse(avatar).ok()?;
    if url.scheme() != "https" || url.host_str() != Some("cdn-avatars.huggingface.co") {
        return None;
    }
    let path = url.path().strip_prefix("/v1/production/uploads/")?;
    let mut response = state
        .client
        .get(format!(
            "https://cdn-uploads.huggingface.co/production/uploads/{path}"
        ))
        .timeout(Duration::from_secs(6))
        .send()
        .await
        .ok()?;
    if !response.status().is_success()
        || response
            .content_length()
            .is_some_and(|size| size > 1_048_576)
    {
        return None;
    }
    let mut data = Vec::new();
    while let Some(chunk) = response.chunk().await.ok()? {
        if data.len() + chunk.len() > 1_048_576 {
            return None;
        }
        data.extend_from_slice(&chunk);
    }
    let mime = if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        "image/png"
    } else if data.starts_with(b"\xff\xd8\xff") {
        "image/jpeg"
    } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
        "image/webp"
    } else if data.starts_with(b"GIF8") {
        "image/gif"
    } else {
        return None;
    };
    Some(format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(data)
    ))
}

async fn lookup_author_avatar(state: &AppState, author: &str) -> Result<Option<String>, String> {
    if author.is_empty()
        || author.len() > 96
        || !author
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
    {
        return Ok(None);
    }
    for kind in ["organizations", "users"] {
        let response = state
            .client
            .get(format!(
                "https://huggingface.co/api/{kind}/{author}/overview"
            ))
            .timeout(Duration::from_secs(4))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            continue;
        }
        let profile: Value = response.json().await.map_err(|e| e.to_string())?;
        if let Some(url) = profile["avatarUrl"].as_str().filter(|url| {
            reqwest::Url::parse(url).is_ok_and(|url| {
                url.scheme() == "https"
                    && url.host_str() == Some("cdn-avatars.huggingface.co")
                    && url.username().is_empty()
                    && url.password().is_none()
            })
        }) {
            return Ok(Some(
                original_avatar_data(state, url)
                    .await
                    .unwrap_or_else(|| url.to_owned()),
            ));
        }
    }
    Ok(None)
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
    begin_download(state.inner(), repo, filename).await
}

fn valid_filename(filename: &str) -> bool {
    filename.ends_with(".gguf")
        && !filename.starts_with('-')
        && Path::new(filename)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

fn download_revision(model: &Value) -> Result<&str, String> {
    model["sha"]
        .as_str()
        .filter(|sha| sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or_else(|| "Hub returned an invalid commit revision".into())
}

async fn begin_download(state: &AppState, repo: String, filename: String) -> Result<(), String> {
    if !valid_repo(&repo) || !valid_filename(&filename) {
        return Err("Choose a relative GGUF filename (not starting with '-') from an owner/model repository.".into());
    }
    let settings = {
        let _guard = state.lifecycle.lock().await;
        if state.app.exiting.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("Noullama is shutting down.".into());
        }
        let mut download = state.download.lock().await;
        if download.active {
            return Err("A model download is already in progress.".into());
        }
        *download = DownloadState {
            active: true,
            repo: repo.clone(),
            file: filename.clone(),
            message: "Checking repository access and preparing download…".into(),
            ..Default::default()
        };
        state.settings.lock().await.clone()
    };
    if let Err(error) = prepare_download(state, &settings, &repo, &filename).await {
        let mut download = state.download.lock().await;
        download.active = false;
        download.error = Some(error.clone());
        download.message = "Download could not start.".into();
        return Err(error);
    }
    Ok(())
}

async fn prepare_download(
    state: &AppState,
    settings: &Settings,
    repo: &str,
    filename: &str,
) -> Result<(), String> {
    let model = metadata(state, repo).await?;
    if !model["siblings"]
        .as_array()
        .is_some_and(|a| a.iter().any(|f| f["rfilename"] == filename))
    {
        return Err("This file is not present in the selected repository.".into());
    }
    let revision = download_revision(&model)?;
    let directory = model_directory(settings, &state.data_dir).join(repo.replace('/', "--"));
    tokio::fs::create_dir_all(&directory)
        .await
        .map_err(|e| e.to_string())?;
    let destination = directory.join(&filename).to_string_lossy().into_owned();
    let mut command = Command::new(&settings.hf_path);
    command
        .args([
            "download",
            repo,
            filename,
            "--revision",
            revision,
            "--local-dir",
        ])
        .arg(&directory);
    let token = hf_token(state).await;
    if let Some(token) = &token {
        command.env("HF_TOKEN", token);
    }
    command
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "0")
        .env("TQDM_POSITION", "-1")
        .env("TQDM_MININTERVAL", "0.2");
    let _spawn_guard = state.lifecycle.lock().await;
    if state.app.exiting.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("Noullama is shutting down.".into());
    }
    let mut child = command.stdout(Stdio::null()).stderr(Stdio::piped()).kill_on_drop(true).spawn().map_err(|e| format!("Cannot start hf: {e}. Install the Hugging Face CLI or configure its executable path."))?;
    {
        let mut download = state.download.lock().await;
        download.message = "Downloading with hf; resumable files are kept on disk…".into();
        download.total_bytes = model["siblings"]
            .as_array()
            .and_then(|files| files.iter().find(|file| file["rfilename"] == filename))
            .and_then(|file| file["size"].as_u64());
    }
    state
        .log(format!(
            "Downloading {repo}/{filename} at {}",
            revision.chars().take(12).collect::<String>()
        ))
        .await;
    let state = state.clone();
    let repo = repo.to_owned();
    let filename = filename.to_owned();
    let app = state.app.clone();
    let task = tokio::spawn(async move {
        let mut stop = state.app.api_stop.subscribe();
        let stderr = child.stderr.take();
        let progress_state = state.clone();
        let reader = tokio::spawn(async move {
            if let Some(stderr) = stderr {
                track_hf_progress(stderr, progress_state, token.into_iter().collect()).await
            } else {
                VecDeque::new()
            }
        });
        let completed = tokio::select! {
            result = child.wait() => Some(result),
            _ = stop.wait_for(|stop| *stop) => None,
        };
        let result = match completed {
            Some(result) => result,
            None => {
                let _ = child.kill().await;
                child.wait().await
            }
        };
        let diagnostics = reader
            .await
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>()
            .join("\n");
        let size = tokio::fs::metadata(&destination)
            .await
            .ok()
            .filter(|m| m.is_file())
            .map(|m| m.len());
        let success = result.as_ref().is_ok_and(|s| s.success()) && size.is_some();
        let mut download = state.download.lock().await;
        download.active = false;
        if success {
            download.downloaded_bytes = size.unwrap_or(download.downloaded_bytes);
            download.total_bytes = size.or(download.total_bytes);
            download.progress = Some(100.0);
            download.path = Some(destination);
            download.message = "Download complete. Ready to load.".into();
        } else {
            let status = match result {
                Ok(status) => status.to_string(),
                Err(error) => error.to_string(),
            };
            download.error = Some(if diagnostics.is_empty() {
                format!("hf download failed ({status}). Check connection, disk space, and repository access; retry resumes the download.")
            } else {
                format!("hf download failed ({status}):\n{diagnostics}")
            });
            download.message = "Download failed; partial files were preserved.".into();
        }
        drop(download);
        state
            .log(if success {
                format!("Download complete: {filename}")
            } else {
                format!("Download failed: {repo}/{filename}\n{diagnostics}")
            })
            .await;
    });
    *app.download_task.lock().await = Some(task);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raster_logos_are_antialiased_at_the_requested_device_resolution() {
        use base64::Engine;
        let original = image::RgbaImage::from_fn(950, 950, |x, y| {
            if x < y {
                image::Rgba([0, 0, 0, 255])
            } else {
                image::Rgba([255, 255, 255, 255])
            }
        });
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(original)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        let source = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes.into_inner())
        );
        let result = raster_avatar(&source, 32).unwrap();
        let encoded = result.split_once(',').unwrap().1;
        let decoded = image::load_from_memory(
            &base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .unwrap(),
        )
        .unwrap()
        .to_rgba8();
        assert_eq!(decoded.dimensions(), (32, 32));
        assert!(
            decoded.pixels().any(|pixel| pixel[0] > 0 && pixel[0] < 255),
            "Downscaling must retain filtered edge pixels"
        );
    }
    #[tokio::test]
    #[ignore = "reads the public Liquid AI logo to verify original image resolution"]
    async fn author_logo_uses_original_resolution_instead_of_avatar_thumbnail() {
        use base64::Engine;
        let base =
            std::env::temp_dir().join(format!("noullama-logo-test-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let image = lookup_author_avatar(&state, "LiquidAI")
            .await
            .unwrap()
            .unwrap();
        let encoded = image
            .strip_prefix("data:image/png;base64,")
            .expect("Original PNG should load through the backend");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
        let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
        assert!(
            width > 200 && height > 200,
            "Expected original upload, got {width}x{height}"
        );
        println!(
            "Original Liquid AI logo: {width}x{height}, {} bytes",
            bytes.len()
        );
        std::fs::remove_dir_all(base).unwrap();
    }
    #[test]
    fn invalid_hub_revisions_return_errors_without_utf8_slicing() {
        for sha in [
            "",
            "main",
            "aaaaaaaaaaaéaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "----------------------------------------",
        ] {
            assert!(download_revision(&json!({"sha":sha})).is_err());
        }
        assert!(download_revision(&json!({})).is_err());
        assert_eq!(
            download_revision(&json!({"sha":"0123456789abcdef0123456789abcdef01234567"})).unwrap(),
            "0123456789abcdef0123456789abcdef01234567"
        );
    }
    #[test]
    fn option_like_and_escaping_download_paths_are_rejected() {
        for name in [
            "--revision=main.gguf",
            "../weights.gguf",
            "/tmp/weights.gguf",
            "folder/../../weights.gguf",
        ] {
            assert!(!valid_filename(name), "{name}");
        }
        assert!(valid_filename("weights.gguf"));
        assert!(valid_filename("folder/weights.gguf"));
        assert!(!valid_repo("--help/model"));
    }

    #[tokio::test]
    async fn diagnostics_keep_a_bounded_redacted_tail_and_final_unterminated_line() {
        let base =
            std::env::temp_dir().join(format!("noullama-diagnostics-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let mut child = Command::new("sh").args(["-c", "i=0; while [ $i -lt 20 ]; do printf 'old diagnostic %s\\n' $i >&2; i=$((i+1)); done; printf 'Disk full: custom-secret hf_fakeToken Bearer other-secret https://example.com/file?token=private' >&2"])
            .stderr(Stdio::piped()).stdout(Stdio::null()).spawn().unwrap();
        let tail = track_hf_progress(
            child.stderr.take().unwrap(),
            state,
            vec!["custom-secret".into()],
        )
        .await;
        child.wait().await.unwrap();
        assert_eq!(tail.len(), 12);
        let final_line = tail.back().unwrap();
        assert!(final_line.contains("Disk full"));
        for secret in [
            "custom-secret",
            "hf_fakeToken",
            "other-secret",
            "https://",
            "private",
        ] {
            assert!(!final_line.contains(secret), "{secret}");
        }
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn discovery_preserves_variants_and_reports_limits_and_missing_custom_roots() {
        let base = std::env::temp_dir().join(format!("noullama-scan-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).unwrap();
        for name in ["Q4.gguf", "Q8.gguf", "mmproj.gguf"] {
            std::fs::write(base.join(name), "weights").unwrap();
        }
        let scan = scan_local_models(vec![(base.clone(), true)], 100, Duration::from_secs(2));
        assert_eq!(scan["models"].as_array().unwrap().len(), 3);
        assert_eq!(scan["incomplete"], false);
        let limited = scan_local_models(vec![(base.clone(), true)], 1, Duration::from_secs(2));
        assert_eq!(limited["models"].as_array().unwrap().len(), 1);
        assert_eq!(limited["incomplete"], true);
        assert!(!limited["warnings"].as_array().unwrap().is_empty());
        let missing = scan_local_models(
            vec![(base.join("unmounted"), true)],
            100,
            Duration::from_secs(2),
        );
        assert_eq!(missing["incomplete"], true);
        assert!(!missing["warnings"].as_array().unwrap().is_empty());
        std::fs::remove_dir_all(base).unwrap();
    }

    #[tokio::test]
    async fn slow_metadata_does_not_block_snapshots_or_settings_and_reserves_one_download() {
        use tokio::io::AsyncWriteExt;
        let base =
            std::env::temp_dir().join(format!("noullama-download-lock-{}", uuid::Uuid::new_v4()));
        let mut state = AppState::new(base.clone()).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        state.client = reqwest::Client::builder()
            .proxy(
                reqwest::Proxy::https(format!("http://{}", listener.local_addr().unwrap()))
                    .unwrap(),
            )
            .build()
            .unwrap();
        *state.token.lock().await = Some("fake-test-token".into());
        let worker_state = state.clone();
        let worker = tokio::spawn(async move {
            begin_download(
                &worker_state,
                "LiquidAI/d1-2B-GGUF".into(),
                "model.gguf".into(),
            )
            .await
        });
        let (mut connection, _) = tokio::time::timeout(Duration::from_secs(2), listener.accept())
            .await
            .unwrap()
            .unwrap();
        let snapshot = tokio::time::timeout(
            Duration::from_millis(500),
            crate::runtime::state_snapshot(&state),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(snapshot["download"]["active"], true);
        let mut settings = state.settings.lock().await.clone();
        settings.hf_path = "/custom/hf".into();
        tokio::time::timeout(
            Duration::from_millis(500),
            crate::runtime::persist_settings(&state, settings.clone()),
        )
        .await
        .unwrap()
        .unwrap();
        settings.download_dir = base.join("different").to_string_lossy().into_owned();
        assert!(crate::runtime::persist_settings(&state, settings)
            .await
            .is_err());
        assert!(tokio::time::timeout(
            Duration::from_millis(500),
            begin_download(&state, "LiquidAI/d1-2B-GGUF".into(), "second.gguf".into())
        )
        .await
        .unwrap()
        .is_err());
        connection
            .write_all(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            .await
            .unwrap();
        drop(connection);
        assert!(tokio::time::timeout(Duration::from_secs(2), worker)
            .await
            .unwrap()
            .unwrap()
            .is_err());
        assert!(!state.download.lock().await.active);
        std::fs::remove_dir_all(base).unwrap();
    }
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
        track_hf_progress(child.stderr.take().unwrap(), state.clone(), Vec::new()).await;
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
        assert_eq!(files["models"].as_array().unwrap().len(), 2);
        assert!(original.join("model.gguf").exists());
        std::fs::remove_dir_all(base).unwrap();
    }
}
