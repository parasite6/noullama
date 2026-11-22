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

/// The entry for one download attempt; None once a retry has replaced it.
fn entry<'a>(downloads: &'a mut [DownloadState], key: &DownloadKey) -> Option<&'a mut DownloadState> {
    downloads.iter_mut().find(|d| d.id == key.id && d.attempt == key.attempt)
}

#[derive(Clone)]
struct DownloadKey {
    id: String,
    attempt: u64,
}

async fn handle_hf_line(
    line: &[u8],
    state: &AppState,
    key: &DownloadKey,
    secrets: &[String],
    tail: &mut VecDeque<String>,
) {
    let line = String::from_utf8_lossy(line);
    if let Some(progress) = parse_hf_progress(&line) {
        let mut downloads = state.downloads.lock().await;
        let Some(download) = entry(&mut downloads, key) else { return };
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
    key: DownloadKey,
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
                    handle_hf_line(&pending, &state, &key, &secrets, &mut tail).await;
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
        handle_hf_line(&pending, &state, &key, &secrets, &mut tail).await;
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

/// How the selected model can use MTP: its own layers, or a head: the one
/// models.yml lists for it, or else a fitting head in its folder. Works for a
/// model chosen from outside the scanned folders.
#[tauri::command]
pub async fn model_mtp(path: String, state: tauri::State<'_, AppState>) -> Result<crate::gguf::ModelMtp, String> {
    crate::library::refresh(state.inner()).await;
    let listed = state.library.lock().await.entry(&path).map(|e| e.mtp.clone()).unwrap_or_default();
    let mut found = tokio::task::spawn_blocking(move || crate::gguf::model_mtp(&path))
        .await
        .map_err(|e| e.to_string())?;
    if !listed.is_empty() {
        found.head = Some(listed);
    }
    Ok(found)
}

async fn list_local_models(state: &AppState) -> Result<Value, String> {
    let settings = state.settings.lock().await.clone();
    let base = model_directory(&settings, &state.data_dir);
    let original = state.data_dir.join("models");
    let mut roots = vec![(base.clone(), !settings.download_dir.trim().is_empty())];
    if base != original {
        roots.push((original, false));
    }
    let scan = tokio::task::spawn_blocking(move || scan_local_models(roots, 20_000, Duration::from_secs(2)))
        .await
        .map_err(|e| e.to_string())?;
    // Every model found gets an entry in models.yml; projectors and MTP heads do not.
    let models: Vec<String> = scan["models"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|m| m["mtp_head"] != true && !m["name"].as_str().unwrap_or_default().to_lowercase().contains("mmproj"))
        .filter_map(|m| m["path"].as_str().map(str::to_owned))
        .collect();
    crate::library::ensure(state, &models).await;
    Ok(scan)
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
                    // MTP support is read from each file's metadata (cached per
                    // file version): a standalone head, or a model that can use
                    // its own MTP layers or a fitting head in its folder.
                    let name = entry.file_name().to_string_lossy().into_owned();
                    let text = path.to_string_lossy();
                    let projector = name.to_lowercase().contains("mmproj");
                    let mtp_head = !projector && crate::gguf::is_mtp_head(&text);
                    let mtp = !projector && !mtp_head && {
                        let found = crate::gguf::model_mtp(&text);
                        found.built_in || found.head.is_some()
                    };
                    // How decisions would be read from it, before it is loaded.
                    let decision = (!projector && !mtp_head)
                        .then(|| crate::decision::resolve("auto", &text))
                        .flatten()
                        .map(crate::decision::Layout::name);
                    files.push(json!({"path": path, "name": name, "size": entry.metadata().ok().map(|m| m.len()), "mtp": mtp, "mtp_head": mtp_head, "decision": decision}));
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
    let cli = crate::runtime::probe_output(Command::new(hf).arg("--version"), Duration::from_secs(8))
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
    context_size: u32,
    gpu_layers: i32,
    state: tauri::State<'_, AppState>,
) -> Result<Value, String> {
    let mut found = Vec::new();
    let hardware = crate::hardware::hardware();
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
                ("blobs", "true"),
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
                let fit = model["siblings"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|file| {
                        file["rfilename"].as_str().is_some_and(|name| {
                            name.ends_with(".gguf")
                                && !name.to_lowercase().contains("mmproj")
                                && !crate::gguf::named_like_mtp_head(name)
                        })
                    })
                    .filter_map(|file| file["size"].as_u64())
                    .min()
                    .and_then(|size| {
                        crate::hardware::fit_model_bytes(size, context_size, gpu_layers, hardware)
                    });
                found.push(json!({"id": model["id"], "downloads": model["downloads"], "likes": model["likes"], "tags": model["tags"], "fit": fit}));
            }
        }
    }
    Ok(Value::Array(found))
}

fn inspection_files(
    model: &Value,
    context_size: u32,
    gpu_layers: i32,
    hardware: &crate::hardware::Hardware,
) -> Vec<Value> {
    model["siblings"].as_array().into_iter().flatten()
        .filter(|file| file["rfilename"].as_str().is_some_and(|name| name.ends_with(".gguf")))
        .map(|file| {
            let name = file["rfilename"].as_str().unwrap_or_default();
            let projector = name.to_lowercase().contains("mmproj");
            let mtp_head = crate::gguf::named_like_mtp_head(name);
            let fit = if projector || mtp_head { None } else {
                file["size"].as_u64().and_then(|size| crate::hardware::fit_model_bytes(size, context_size, gpu_layers, hardware))
            };
            json!({"name": file["rfilename"], "size": file["size"], "projector": projector, "mtp_head": mtp_head, "fit": fit})
        }).collect()
}

#[tauri::command]
pub async fn inspect_model(
    repo: String,
    context_size: Option<u32>,
    gpu_layers: Option<i32>,
    state: tauri::State<'_, AppState>,
) -> Result<Value, String> {
    let model = metadata(&state, &repo).await?;
    let settings = state.settings.lock().await.clone();
    let files = inspection_files(
        &model,
        context_size.unwrap_or(settings.context_size),
        gpu_layers.unwrap_or(settings.gpu_layers),
        crate::hardware::hardware(),
    );
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

#[tauri::command]
pub async fn cancel_download(id: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    cancel_one(state.inner(), &id).await;
    Ok(())
}

// Checked under the downloads lock so a cancel cannot leak into the next attempt.
async fn cancel_one(state: &AppState, id: &str) {
    let downloads = state.downloads.lock().await;
    if downloads.iter().any(|d| d.id == id && d.active) {
        if let Some((_, cancel)) = state.app.download_cancels.lock().await.get(id) {
            cancel.send_replace(true);
        }
    }
}

/// Removes a finished, failed or cancelled download from the list.
#[tauri::command]
pub async fn dismiss_download(id: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    state.downloads.lock().await.retain(|d| d.id != id || d.active);
    Ok(())
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

/// Finished entries kept for the downloads list; older ones drop off.
const KEPT_FINISHED_DOWNLOADS: usize = 8;

async fn begin_download(state: &AppState, repo: String, filename: String) -> Result<(), String> {
    if !valid_repo(&repo) || !valid_filename(&filename) {
        return Err("Choose a relative GGUF filename (not starting with '-') from an owner/model repository.".into());
    }
    let id = format!("{repo}/{filename}");
    let (settings, key, cancel) = {
        let _guard = state.lifecycle.lock().await;
        if state.app.exiting.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("Noullama is shutting down.".into());
        }
        let mut downloads = state.downloads.lock().await;
        if downloads.iter().any(|d| d.id == id && d.active) {
            return Err(format!("{filename} is already downloading."));
        }
        let attempt = state.app.download_attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        let (sender, receiver) = tokio::sync::watch::channel(false);
        state.app.download_cancels.lock().await.insert(id.clone(), (attempt, sender));
        downloads.retain(|d| d.id != id);
        downloads.push(DownloadState {
            id: id.clone(),
            attempt,
            active: true,
            repo: repo.clone(),
            file: filename.clone(),
            message: "Checking repository access and preparing download…".into(),
            ..Default::default()
        });
        while downloads.iter().filter(|d| !d.active).count() > KEPT_FINISHED_DOWNLOADS {
            if let Some(oldest) = downloads.iter().position(|d| !d.active) {
                downloads.remove(oldest);
            }
        }
        (state.settings.lock().await.clone(), DownloadKey { id, attempt }, receiver)
    };
    let mut waiter = cancel.clone();
    let prepared = tokio::select! {
        result = prepare_download(state, &settings, &repo, &filename, &key, cancel.clone()) => result,
        _ = waiter.wait_for(|cancelled| *cancelled) => Err("Download cancelled.".into()),
    };
    if let Err(error) = prepared {
        release_cancel(state, &key).await;
        let mut downloads = state.downloads.lock().await;
        let Some(download) = entry(&mut downloads, &key) else { return Err(error) };
        download.active = false;
        if *cancel.borrow() {
            download.cancelled = true;
            download.message = "Download cancelled before it started.".into();
            return Ok(());
        }
        download.error = Some(error.clone());
        download.message = "Download could not start.".into();
        return Err(error);
    }
    Ok(())
}

async fn release_cancel(state: &AppState, key: &DownloadKey) {
    let mut cancels = state.app.download_cancels.lock().await;
    if cancels.get(&key.id).is_some_and(|(attempt, _)| *attempt == key.attempt) {
        cancels.remove(&key.id);
    }
}

async fn prepare_download(
    state: &AppState,
    settings: &Settings,
    repo: &str,
    filename: &str,
    key: &DownloadKey,
    cancel: tokio::sync::watch::Receiver<bool>,
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
    let destination = directory.join(filename).to_string_lossy().into_owned();
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
    if *cancel.borrow() {
        return Err("Download cancelled.".into());
    }
    let mut child = command.stdout(Stdio::null()).stderr(Stdio::piped()).kill_on_drop(true).spawn().map_err(|e| format!("Cannot start hf: {e}. Install the Hugging Face CLI or configure its executable path."))?;
    {
        let mut downloads = state.downloads.lock().await;
        if let Some(download) = entry(&mut downloads, key) {
            download.message = "Downloading with hf; resumable files are kept on disk…".into();
            download.total_bytes = model["siblings"]
                .as_array()
                .and_then(|files| files.iter().find(|file| file["rfilename"] == filename))
                .and_then(|file| file["size"].as_u64());
        }
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
    let key = key.clone();
    let app = state.app.clone();
    // hf keeps resume state under <local-dir>/.cache/huggingface/download/<filename>.*
    let partial_base = directory
        .join(".cache/huggingface/download")
        .join(&filename);
    // A retry resumes from these files, so they no longer belong to a cancelled download.
    state
        .app
        .cancelled_partials
        .lock()
        .await
        .retain(|(base, _)| *base != partial_base);
    let mut tasks = app.download_tasks.lock().await;
    tasks.retain(|task| !task.is_finished());
    let task = tokio::spawn(async move {
        let mut stop = state.app.api_stop.subscribe();
        let mut cancel = cancel;
        let stderr = child.stderr.take();
        let progress_state = state.clone();
        let progress_key = key.clone();
        let reader = tokio::spawn(async move {
            if let Some(stderr) = stderr {
                track_hf_progress(stderr, progress_state, progress_key, token.into_iter().collect()).await
            } else {
                VecDeque::new()
            }
        });
        let mut cancelled = false;
        let completed = tokio::select! {
            result = child.wait() => Some(result),
            _ = stop.wait_for(|stop| *stop) => None,
            _ = cancel.wait_for(|cancelled| *cancelled) => { cancelled = true; None }
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
        release_cancel(&state, &key).await;
        if cancelled {
            // Keep the partial files until the app exits so an immediate retry can resume.
            let files = cancelled_partial_files(&partial_base);
            let mut partials = state.app.cancelled_partials.lock().await;
            partials.retain(|(base, _)| *base != partial_base);
            partials.push((partial_base, files));
        }
        let model_path = destination.clone();
        let mut downloads = state.downloads.lock().await;
        if let Some(download) = entry(&mut downloads, &key) {
            download.active = false;
            if cancelled {
                download.cancelled = true;
                download.message =
                    "Download cancelled; partial files are kept until you quit Noullama.".into();
            } else if success {
                download.downloaded_bytes = size.unwrap_or(download.downloaded_bytes);
                download.total_bytes = size.or(download.total_bytes);
                download.progress = Some(100.0);
                download.path = Some(destination);
                download.message = "Download complete. Ready to load.".into();
            } else {
                let status = match &result {
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
        }
        drop(downloads);
        if success && !cancelled {
            if filename.to_lowercase().contains("mmproj") || crate::gguf::is_mtp_head(&model_path) {
                crate::library::attach(&state, &model_path).await;
            } else {
                crate::library::ensure(&state, &[model_path.clone()]).await;
            }
        }
        state
            .log(if success {
                format!("Download complete: {filename}")
            } else {
                format!("Download failed: {repo}/{filename}\n{diagnostics}")
            })
            .await;
    });
    tasks.push(task);
    Ok(())
}

/// huggingface_hub's name for a file's partial data: the URL-safe base64 of the
/// SHA-1 of `<file>.metadata`, then the etag (`_local_folder.py`, `_short_hash`).
fn partial_data_prefix(file_name: &str) -> String {
    use base64::Engine;
    use sha1::Digest;
    let digest = sha1::Sha1::digest(format!("{file_name}.metadata").as_bytes());
    format!("{}.", base64::engine::general_purpose::URL_SAFE.encode(digest))
}

// Several downloads from one repository share this cache folder, so the
// partial data is matched by the name hf derives from this file only.
fn cancelled_partial_files(base: &Path) -> Vec<std::path::PathBuf> {
    let with_suffix = |suffix: &str| {
        let mut name = base.as_os_str().to_owned();
        name.push(suffix);
        std::path::PathBuf::from(name)
    };
    let mut files = vec![with_suffix(".metadata"), with_suffix(".lock")];
    let Some(file_name) = base.file_name().map(|n| n.to_string_lossy().into_owned()) else {
        return files;
    };
    let prefix = partial_data_prefix(&file_name);
    if let Some(entries) = base.parent().and_then(|dir| std::fs::read_dir(dir).ok()) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with(&prefix) && name.ends_with(".incomplete") {
                files.push(entry.path());
            }
        }
    }
    files
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
    #[test]
    fn inspection_rates_each_quantization_and_excludes_projectors() {
        let hardware = crate::hardware::Hardware {
            ram_bytes: 32_000_000_000,
            vram_bytes: Some(12_000_000_000),
            gpu: None,
            cpu_cores: 8,
        };
        let model = json!({"siblings": [
            {"rfilename": "model-Q4.gguf", "size": 4_000_000_000_u64},
            {"rfilename": "model-Q8.gguf", "size": 28_600_000_000_u64},
            {"rfilename": "mmproj-F16.gguf", "size": 930_000_000},
            {"rfilename": "unknown.gguf"},
            {"rfilename": "mtp-model-Q8.gguf", "size": 400_000_000},
            {"rfilename": "README.md", "size": 50}
        ]});
        let files = inspection_files(&model, 8192, 99, &hardware);
        assert_eq!(files.len(), 5);
        assert_eq!(files[4]["mtp_head"], true);
        assert!(files[4]["fit"].is_null());
        assert_eq!(files[0]["mtp_head"], false);
        assert_eq!(files[0]["fit"]["model_bytes"], 4_000_000_000_u64);
        assert_eq!(files[1]["fit"]["model_bytes"], 28_600_000_000_u64);
        assert_ne!(files[0]["fit"]["level"], files[1]["fit"]["level"]);
        assert_eq!(files[2]["projector"], true);
        assert!(files[2]["fit"].is_null());
        assert!(files[3]["fit"].is_null());
        let larger_context = inspection_files(&model, 32768, 99, &hardware);
        assert!(
            larger_context[0]["fit"]["required_bytes"].as_u64().unwrap()
                > files[0]["fit"]["required_bytes"].as_u64().unwrap()
        );
    }

    #[test]
    fn invalid_hub_revisions_return_errors_without_utf8_slicing() {
        for sha in [
            "",
            "main",
            "aaaaaaaaaaaéaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "----------------------------------------",
        ] {
            assert_eq!(download_revision(&json!({"sha":sha})).unwrap_err(), "Hub returned an invalid commit revision");
        }
        assert_eq!(download_revision(&json!({})).unwrap_err(), "Hub returned an invalid commit revision");
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
            DownloadKey { id: "acme/m/model.gguf".into(), attempt: 1 },
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
    async fn slow_metadata_does_not_block_snapshots_or_settings_and_reserves_each_file() {
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
                "acme/decision-model-GGUF".into(),
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
        assert_eq!(snapshot["downloads"][0]["active"], true);
        assert_eq!(snapshot["downloads"][0]["id"], "acme/decision-model-GGUF/model.gguf");
        tokio::time::timeout(
            Duration::from_millis(500),
            crate::runtime::update_settings(&state, |s| { s.hf_path = "/custom/hf".into(); Ok(()) }),
        )
        .await
        .unwrap()
        .unwrap();
        let different = base.join("different").to_string_lossy().into_owned();
        assert_eq!(crate::runtime::update_settings(&state, |s| { s.download_dir = different; Ok(()) }).await.unwrap_err(), "Wait for downloads to finish before changing their directory.");
        // The same file cannot start twice while it is active.
        let duplicate = tokio::time::timeout(
            Duration::from_millis(500),
            begin_download(&state, "acme/decision-model-GGUF".into(), "model.gguf".into()),
        )
        .await
        .unwrap();
        assert!(duplicate.unwrap_err().contains("already downloading"));
        // A different file starts alongside it.
        let second_state = state.clone();
        let second = tokio::spawn(async move {
            begin_download(&second_state, "acme/decision-model-GGUF".into(), "second.gguf".into()).await
        });
        let (mut second_connection, _) = tokio::time::timeout(Duration::from_secs(2), listener.accept())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(state.downloads.lock().await.iter().filter(|d| d.active).count(), 2);
        second_connection
            .write_all(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            .await
            .unwrap();
        drop(second_connection);
        assert!(tokio::time::timeout(Duration::from_secs(2), second).await.unwrap().unwrap().is_err());
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
        let downloads = state.downloads.lock().await.clone();
        assert_eq!(downloads.len(), 2);
        assert!(downloads.iter().all(|d| !d.active && d.error.is_some()));
        assert!(state.app.download_cancels.lock().await.is_empty());
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
        let key = DownloadKey { id: "acme/m/model.gguf".into(), attempt: 1 };
        state.downloads.lock().await.push(DownloadState { id: key.id.clone(), attempt: 1, total_bytes: Some(168_000_000), ..Default::default() });
        // Exercise the actual stderr framing path, not a file-size approximation.
        let mut child = Command::new("sh").args(["-c", "printf 'model: downloading bytes: | 12.3MB, 2MB/s\\rmodel: reconstructing file: 40%%|#### | 67.8MB / 168MB\\n' >&2"])
            .stderr(Stdio::piped()).stdout(Stdio::null()).spawn().unwrap();
        track_hf_progress(child.stderr.take().unwrap(), state.clone(), key, Vec::new()).await;
        child.wait().await.unwrap();
        let download = state.downloads.lock().await[0].clone();
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
            &json!({"id":"acme/chat-model", "tags":["text-generation"]})
        ));
        assert!(!valid_repo("../escape"));
        assert!(!valid_repo("org/repo/extra"));
        assert!(valid_repo("acme/decision-model-GGUF"));
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

    #[test]
    fn partial_data_names_match_huggingface_hub() {
        // base64.urlsafe_b64encode(hashlib.sha1(b"model.gguf.metadata").digest())
        assert_eq!(partial_data_prefix("model.gguf"), "oXa0XAG8ZeXAcRMIPdQRwbfnQrg=.");
    }

    #[test]
    fn cancelled_cleanup_keeps_partials_of_other_downloads_in_the_same_folder() {
        let base = std::env::temp_dir().join(format!("noullama-partials-{}", uuid::Uuid::new_v4()));
        let cache = base.join(".cache/huggingface/download");
        std::fs::create_dir_all(&cache).unwrap();
        let mine = cache.join(format!("{}0123abcd.incomplete", partial_data_prefix("model.gguf")));
        let other = cache.join(format!("{}4567.incomplete", partial_data_prefix("mmproj.gguf")));
        std::fs::write(&mine, b"partial").unwrap();
        std::fs::write(&other, b"still downloading").unwrap();
        let files = cancelled_partial_files(&cache.join("model.gguf"));
        assert!(files.contains(&mine));
        assert!(!files.contains(&other));
        assert!(files.contains(&cache.join("model.gguf.metadata")));
        assert!(files.contains(&cache.join("model.gguf.lock")));
        std::fs::remove_dir_all(base).unwrap();
    }

    #[tokio::test]
    async fn cancel_only_applies_to_the_named_active_download() {
        let base = std::env::temp_dir().join(format!("noullama-cancel-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let switch = |attempt| {
            let (sender, receiver) = tokio::sync::watch::channel(false);
            (attempt, sender, receiver)
        };
        let (a, a_tx, mut a_rx) = switch(1);
        let (b, b_tx, b_rx) = switch(2);
        state.app.download_cancels.lock().await.insert("r/a.gguf".into(), (a, a_tx));
        state.app.download_cancels.lock().await.insert("r/b.gguf".into(), (b, b_tx));
        // An idle entry must not be cancelled, so a cancel cannot leak into its next attempt.
        state.downloads.lock().await.push(DownloadState { id: "r/a.gguf".into(), attempt: a, active: false, ..Default::default() });
        state.downloads.lock().await.push(DownloadState { id: "r/b.gguf".into(), attempt: b, active: true, ..Default::default() });
        cancel_one(&state, "r/a.gguf").await;
        assert!(!*a_rx.borrow());
        state.downloads.lock().await[0].active = true;
        cancel_one(&state, "r/a.gguf").await;
        tokio::time::timeout(Duration::from_millis(100), a_rx.wait_for(|c| *c)).await.unwrap().unwrap();
        // The other download keeps going.
        assert!(!*b_rx.borrow());
        std::fs::remove_dir_all(base).unwrap();
    }
}
