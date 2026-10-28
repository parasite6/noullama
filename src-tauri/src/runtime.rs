use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::{Child, Command},
    sync::Mutex,
};

pub const DEFAULT_RUNTIME: &str =
    "/home/andrewunknown/llamacpp-new-patch/llama-b11954-bin-ubuntu-cuda-13.4-x64/llama-b11954";

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub runtime_path: String,
    pub model_path: String,
    pub mmproj_path: String,
    pub context_size: u32,
    pub gpu_layers: i32,
    pub api_port: u16,
    pub runtime_port: u16,
    pub hf_path: String,
    pub download_dir: String,
    pub library_path: String,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            runtime_path: DEFAULT_RUNTIME.into(),
            model_path: String::new(),
            mmproj_path: String::new(),
            context_size: 8192,
            gpu_layers: 99,
            api_port: 11435,
            runtime_port: 11436,
            hf_path: "hf".into(),
            download_dir: String::new(),
            library_path: String::new(),
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    pub settings: Arc<Mutex<Settings>>,
    pub child: Arc<Mutex<Option<Child>>>,
    pub lifecycle: Arc<Mutex<()>>,
    pub logs: Arc<Mutex<VecDeque<String>>>,
    pub token: Arc<Mutex<Option<String>>>,
    pub download: Arc<Mutex<DownloadState>>,
    pub ready: Arc<Mutex<bool>>,
    pub api_error: Arc<Mutex<Option<String>>>,
    pub loaded_model: Arc<Mutex<Option<LoadedModel>>>,
    pub presets_lock: Arc<Mutex<()>>,
    pub local_client: reqwest::Client,
    pub data_dir: PathBuf,
    pub api_port: u16,
    pub client: reqwest::Client,
}

#[derive(Clone, Serialize)]
pub struct LoadedModel {
    pub path: String,
    pub device: String,
}

#[derive(Clone, Default, Serialize)]
pub struct DownloadState {
    pub active: bool,
    pub repo: String,
    pub file: String,
    pub message: String,
    pub path: Option<String>,
    pub error: Option<String>,
    pub progress: Option<f64>,
    pub downloaded_bytes: u64,
    pub transferred_bytes: u64,
    pub total_bytes: Option<u64>,
}

impl AppState {
    pub fn new(data_dir: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        std::fs::create_dir_all(&data_dir)?;
        let settings_path = data_dir.join("settings.json");
        let settings: Settings = if settings_path.exists() {
            serde_json::from_slice(&std::fs::read(settings_path)?)?
        } else {
            Settings::default()
        };
        Ok(Self {
            api_port: settings.api_port,
            settings: Arc::new(Mutex::new(settings)),
            child: Default::default(),
            lifecycle: Default::default(),
            logs: Default::default(),
            token: Default::default(),
            download: Default::default(),
            ready: Default::default(),
            api_error: Default::default(),
            loaded_model: Default::default(),
            presets_lock: Default::default(),
            local_client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(120))
                .build()?,
            data_dir,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()?,
        })
    }
    pub async fn log(&self, line: String) {
        let mut logs = self.logs.lock().await;
        if logs.len() >= 160 {
            logs.pop_front();
        }
        logs.push_back(line);
    }
    pub async fn shutdown(&self) {
        *self.ready.lock().await = false;
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        *self.loaded_model.lock().await = None;
    }
    pub async fn infer(&self, payload: Value) -> Result<Value, String> {
        validate_request(&payload)?;
        if !*self.ready.lock().await {
            return Err("Load a decision model before running inference.".into());
        }
        let port = self.settings.lock().await.runtime_port;
        let response = self
            .local_client
            .post(format!("http://127.0.0.1:{port}/v1/systemone"))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Runtime connection failed: {e}"))?;
        let status = response.status();
        let body = response.text().await.map_err(|e| e.to_string())?;
        let value: Value = serde_json::from_str(&body).map_err(|_| {
            format!(
                "Runtime returned non-JSON ({status}): {}",
                body.chars().take(500).collect::<String>()
            )
        })?;
        if !status.is_success() {
            return Err(format!("Runtime {status}: {value}"));
        }
        Ok(value)
    }
}

pub fn validate_request(payload: &Value) -> Result<(), String> {
    if !payload.is_object() || payload.get("state").is_none() {
        return Err("Request needs a state field and a questions object.".into());
    }
    let questions = payload
        .get("questions")
        .and_then(Value::as_object)
        .filter(|q| !q.is_empty())
        .ok_or("Add at least one named question to the JSON editor.")?;
    for (name, q) in questions {
        if q.get("multiple").and_then(Value::as_bool) == Some(true) {
            return Err(format!(
                "{name}: multiple-choice selection is not supported by the pinned runtime."
            ));
        }
        if !q
            .get("instructions")
            .is_some_and(|v| v.is_string() || v.is_object() || v.is_array())
        {
            return Err(format!("{name}: instructions must contain the question."));
        }
        match q.get("type").and_then(Value::as_str) {
            Some("noul") => {}
            Some("choice")
                if q.get("criteria")
                    .and_then(Value::as_object)
                    .is_some_and(|c| {
                        c.len() >= 2 && c.values().all(|v| v.is_string() || v.is_null())
                    }) => {}
            Some("score")
                if q.get("criteria")
                    .and_then(Value::as_array)
                    .is_some_and(|c| {
                        (2..=10).contains(&c.len()) && c.iter().all(Value::is_string)
                    }) => {}
            Some("choice") => {
                return Err(format!(
                    "{name}: choice needs at least two criteria with string or null descriptions."
                ))
            }
            Some("score") => {
                return Err(format!(
                    "{name}: score needs 2–10 descriptions, ordered lowest first."
                ))
            }
            _ => {
                return Err(format!(
                    "{name}: supported question types are choice, score, and noul."
                ))
            }
        }
    }
    Ok(())
}

pub fn server_binary(path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_dir() {
        path.join(if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        })
    } else {
        path
    }
}

pub fn model_directory(settings: &Settings, data_dir: &Path) -> PathBuf {
    if settings.download_dir.trim().is_empty() {
        data_dir.join("models")
    } else {
        PathBuf::from(settings.download_dir.trim())
    }
}

pub fn runtime_library_dirs(settings: &Settings, binary: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(parent) = binary.parent().filter(|p| !p.as_os_str().is_empty()) {
        paths.push(parent.to_path_buf());
    }
    paths.extend(std::env::split_paths(&settings.library_path));
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    #[cfg(target_os = "linux")]
    for path in [
        "/usr/local/cuda/lib64",
        "/opt/cuda/lib64",
        "/usr/local/lib/ollama/cuda_v13",
        "/usr/local/lib/ollaya/cuda_v13",
    ] {
        let path = PathBuf::from(path);
        if ["libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"]
            .iter()
            .all(|file| path.join(file).is_file())
            && !paths.contains(&path)
        {
            paths.push(path);
        }
    }
    paths.retain(|path| !path.as_os_str().is_empty());
    paths
}

fn runtime_command(settings: &Settings) -> Command {
    let binary = server_binary(&settings.runtime_path);
    let mut command = Command::new(&binary);
    if let Ok(paths) = std::env::join_paths(runtime_library_dirs(settings, &binary)) {
        command.env("LD_LIBRARY_PATH", paths);
    }
    command
}

fn runtime_settings_changed(a: &Settings, b: &Settings) -> bool {
    a.runtime_path != b.runtime_path
        || a.model_path != b.model_path
        || a.mmproj_path != b.mmproj_path
        || a.context_size != b.context_size
        || a.gpu_layers != b.gpu_layers
        || a.runtime_port != b.runtime_port
        || a.library_path != b.library_path
}

#[tauri::command]
pub async fn snapshot(state: tauri::State<'_, AppState>) -> Result<Value, String> {
    let mut child = state.child.lock().await;
    let running = if let Some(c) = child.as_mut() {
        c.try_wait().map_err(|e| e.to_string())?.is_none()
    } else {
        false
    };
    if !running {
        *child = None;
        *state.ready.lock().await = false;
        *state.loaded_model.lock().await = None;
    }
    drop(child);
    let settings = state.settings.lock().await.clone();
    let logs = state.logs.lock().await.clone();
    let download = state.download.lock().await.clone();
    let loaded_model = state.loaded_model.lock().await.clone();
    let ready = *state.ready.lock().await;
    let api_error = state.api_error.lock().await.clone();
    Ok(
        json!({"settings": settings, "running": running, "ready": ready,
        "logs": logs, "download": download, "loaded_model":loaded_model,
        "download_dir": model_directory(&settings, &state.data_dir),
        "api_port": state.api_port, "api_error": api_error, "data_dir": state.data_dir}),
    )
}

#[tauri::command]
pub async fn save_settings(
    settings: Settings,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    persist_settings(state.inner(), settings).await
}

pub async fn persist_settings(state: &AppState, settings: Settings) -> Result<(), String> {
    let _guard = state.lifecycle.lock().await;
    if state.child.lock().await.is_some()
        && runtime_settings_changed(&*state.settings.lock().await, &settings)
    {
        return Err("Stop the runtime before changing its configuration.".into());
    }
    if settings.api_port == 0
        || settings.runtime_port == 0
        || settings.api_port == settings.runtime_port
    {
        return Err("Choose two different, nonzero ports.".into());
    }
    if !(512..=131072).contains(&settings.context_size) || settings.gpu_layers < -1 {
        return Err("Use a context size of 512–131072 and GPU layers of -1 or higher.".into());
    }
    if !settings.download_dir.trim().is_empty()
        && !Path::new(settings.download_dir.trim()).is_absolute()
    {
        return Err("Choose an absolute path for the download directory.".into());
    }
    if state.download.lock().await.active
        && state.settings.lock().await.download_dir != settings.download_dir
    {
        return Err(
            "Wait for the current download to finish before changing its directory.".into(),
        );
    }
    if !settings.download_dir.trim().is_empty() {
        tokio::fs::create_dir_all(settings.download_dir.trim())
            .await
            .map_err(|e| format!("Cannot create download directory: {e}"))?;
    }
    let bytes = serde_json::to_vec_pretty(&settings).map_err(|e| e.to_string())?;
    let temp = state.data_dir.join("settings.json.tmp");
    tokio::fs::write(&temp, bytes)
        .await
        .map_err(|e| e.to_string())?;
    tokio::fs::rename(temp, state.data_dir.join("settings.json"))
        .await
        .map_err(|e| e.to_string())?;
    *state.settings.lock().await = settings;
    Ok(())
}

#[tauri::command]
pub async fn probe_runtime(path: String, library_path: String) -> Result<String, String> {
    let settings = Settings {
        runtime_path: path,
        library_path,
        ..Settings::default()
    };
    let output = tokio::time::timeout(
        Duration::from_secs(10),
        runtime_command(&settings).arg("--version").output(),
    )
    .await
    .map_err(|_| "Runtime version check timed out")?
    .map_err(|e| format!("Cannot run llama-server: {e}"))?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() {
        return Err(text);
    }
    let devices = tokio::time::timeout(
        Duration::from_secs(15),
        runtime_command(&settings).arg("--list-devices").output(),
    )
    .await
    .map_err(|_| "Device detection timed out")?
    .map_err(|e| e.to_string())?;
    Ok(format!(
        "{text}\n{}{}",
        String::from_utf8_lossy(&devices.stdout),
        String::from_utf8_lossy(&devices.stderr)
    ))
}

#[tauri::command]
pub async fn start_runtime(state: tauri::State<'_, AppState>) -> Result<(), String> {
    load_model(state.inner()).await
}

pub async fn load_model(state: &AppState) -> Result<(), String> {
    let _guard = state.lifecycle.lock().await;
    if state.child.lock().await.is_some() {
        return Err("Runtime is already started. Stop it before loading another model.".into());
    }
    let settings = state.settings.lock().await.clone();
    if !Path::new(&settings.model_path).is_file()
        || !settings.model_path.to_lowercase().ends_with(".gguf")
    {
        return Err("Select a downloaded or local .gguf decision model first.".into());
    }
    if !settings.mmproj_path.is_empty() && !Path::new(&settings.mmproj_path).is_file() {
        return Err("The multimodal projector path does not exist.".into());
    }
    // Never attach to or kill an unrelated process already using this port.
    let reservation =
        tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, settings.runtime_port))
            .await
            .map_err(|e| format!("Private runtime port is unavailable: {e}"))?;
    let binary = server_binary(&settings.runtime_path);
    let devices = tokio::time::timeout(
        Duration::from_secs(15),
        runtime_command(&settings).arg("--list-devices").output(),
    )
    .await
    .map_err(|_| "GPU detection timed out")?
    .map_err(|e| format!("Cannot detect runtime devices: {e}"))?;
    let devices = format!(
        "{}{}",
        String::from_utf8_lossy(&devices.stdout),
        String::from_utf8_lossy(&devices.stderr)
    );
    let gpu = devices
        .lines()
        .find(|line| {
            line.trim_start().starts_with("CUDA")
                || line.trim_start().starts_with("Vulkan")
                || line.trim_start().starts_with("Metal")
                || line.trim_start().starts_with("ROCm")
                || line.trim_start().starts_with("SYCL")
        })
        .map(|line| {
            line.trim()
                .split(" (")
                .next()
                .unwrap_or(line.trim())
                .to_owned()
        });
    if settings.gpu_layers != 0 && gpu.is_none() {
        return Err("No GPU backend detected by this runtime. Set the CUDA/shared-library path in Runtime, use a compatible GPU build, or choose GPU layers = 0 for CPU inference.".into());
    }
    let device = if settings.gpu_layers == 0 {
        "CPU".to_owned()
    } else {
        gpu.unwrap_or_default()
    };
    let mut command = runtime_command(&settings);
    command.args([
        "--model",
        &settings.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        &settings.runtime_port.to_string(),
        "--alias",
        "noullama",
        "--ctx-size",
        &settings.context_size.to_string(),
        "--batch-size",
        &settings.context_size.to_string(),
        "--ubatch-size",
        &settings.context_size.to_string(),
        "--n-gpu-layers",
        &settings.gpu_layers.to_string(),
        "--parallel",
        "1",
    ]);
    if !settings.mmproj_path.is_empty() {
        command.args(["--mmproj", &settings.mmproj_path]);
    }
    drop(reservation);
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("Could not start {}: {e}", binary.display()))?;
    for pipe in [
        child
            .stdout
            .take()
            .map(|p| Box::pin(p) as std::pin::Pin<Box<dyn tokio::io::AsyncRead + Send>>),
        child
            .stderr
            .take()
            .map(|p| Box::pin(p) as std::pin::Pin<Box<dyn tokio::io::AsyncRead + Send>>),
    ]
    .into_iter()
    .flatten()
    {
        let state = state.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(pipe).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                state.log(line).await;
            }
        });
    }
    *state.child.lock().await = Some(child);
    *state.loaded_model.lock().await = Some(LoadedModel {
        path: settings.model_path.clone(),
        device: device.clone(),
    });
    state
        .log(format!("Loading {} on {device}", settings.model_path))
        .await;
    state
        .log("Loading model; waiting for the runtime health check…".into())
        .await;
    let state = state.clone();
    tokio::spawn(async move {
        // Serialize startup against Stop and configuration changes.
        let _guard = state.lifecycle.lock().await;
        for _ in 0..180 {
            let mut child = state.child.lock().await;
            match child.as_mut().map(|c| c.try_wait()) {
                Some(Ok(None)) => {}
                _ => {
                    *child = None;
                    *state.loaded_model.lock().await = None;
                    state.log("Runtime exited during startup. Check the log for model/runtime compatibility.".into()).await;
                    return;
                }
            }
            drop(child);
            let url = format!("http://127.0.0.1:{}", settings.runtime_port);
            if state
                .local_client
                .get(format!("{url}/health"))
                .timeout(Duration::from_secs(1))
                .send()
                .await
                .is_ok_and(|r| r.status().is_success())
            {
                let models = state
                    .local_client
                    .get(format!("{url}/v1/models"))
                    .timeout(Duration::from_secs(3))
                    .send()
                    .await;
                let decision = match models {
                    Ok(r) => r.json::<Value>().await.ok().is_some_and(|v| {
                        v["data"].as_array().is_some_and(|m| {
                            m.iter().any(|m| {
                                m["architecture"]["output_modalities"]
                                    .as_array()
                                    .is_some_and(|a| a.iter().any(|x| x == "decisions"))
                            })
                        })
                    }),
                    Err(_) => false,
                };
                if !decision {
                    state.log("Runtime did not advertise native decision support. Use a decision GGUF and a compatible llama.cpp build (b11954 for d1-2B).".into()).await;
                    state.shutdown().await;
                    return;
                }
                *state.ready.lock().await = true;
                state
                    .log("Decision model is ready. Local API is accepting requests.".into())
                    .await;
                return;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        state
            .log("Model loading timed out after 90 seconds.".into())
            .await;
        state.shutdown().await;
    });
    Ok(())
}

#[tauri::command]
pub async fn stop_runtime(state: tauri::State<'_, AppState>) -> Result<(), String> {
    // Startup may be in progress; kill the owned child first so its polling exits.
    state.shutdown().await;
    let _guard = state.lifecycle.lock().await;
    state.shutdown().await;
    state.log("Runtime stopped; model unloaded.".into()).await;
    Ok(())
}

#[tauri::command]
pub async fn decide(payload: Value, state: tauri::State<'_, AppState>) -> Result<Value, String> {
    state.infer(payload).await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_settings_keep_default_downloads_and_allow_custom_roots() {
        let settings: Settings = serde_json::from_value(json!({"gpu_layers":0})).unwrap();
        let base = PathBuf::from("/tmp/noullama-data");
        assert_eq!(model_directory(&settings, &base), base.join("models"));
        let custom = Settings {
            download_dir: "/tmp/custom models".into(),
            ..settings
        };
        assert_eq!(
            model_directory(&custom, &base),
            PathBuf::from("/tmp/custom models")
        );
    }
    #[test]
    fn runtime_library_search_keeps_custom_and_inherited_paths_first() {
        let settings = Settings {
            library_path: "/custom/cuda:/custom/other".into(),
            ..Settings::default()
        };
        let paths = runtime_library_dirs(&settings, Path::new("/runtime/llama-server"));
        assert_eq!(
            &paths[..3],
            &[
                PathBuf::from("/runtime"),
                PathBuf::from("/custom/cuda"),
                PathBuf::from("/custom/other")
            ]
        );
    }

    #[tokio::test]
    async fn saving_download_directory_creates_it_and_rejects_relative_paths() {
        let base =
            std::env::temp_dir().join(format!("noullama-directory-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let mut settings = Settings::default();
        settings.download_dir = "relative/models".into();
        assert!(persist_settings(&state, settings.clone()).await.is_err());
        settings.download_dir = base.join("custom models").to_string_lossy().into_owned();
        persist_settings(&state, settings.clone()).await.unwrap();
        assert!(Path::new(&settings.download_dir).is_dir());
        let stored: Settings =
            serde_json::from_slice(&std::fs::read(base.join("settings.json")).unwrap()).unwrap();
        assert_eq!(stored.download_dir, settings.download_dir);
        std::fs::remove_dir_all(base).unwrap();
    }
}
