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
            runtime_path: String::new(),
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
    pub app: Arc<crate::lifecycle::AppControl>,
    pub runtimes: Arc<crate::runtimes::RuntimeManager>,
    pub settings: Arc<Mutex<Settings>>,
    pub child: Arc<Mutex<Option<Child>>>,
    pub lifecycle: Arc<Mutex<()>>,
    pub session: Arc<Mutex<Option<Arc<RuntimeSession>>>>,
    pub startup_warning: Option<String>,
    pub logs: Arc<Mutex<VecDeque<String>>>,
    pub token: Arc<Mutex<Option<String>>>,
    pub download: Arc<Mutex<DownloadState>>,
    pub ready: Arc<Mutex<bool>>,
    pub api_error: Arc<Mutex<Option<String>>>,
    pub api_notice: Arc<Mutex<Option<String>>>,
    pub loaded_model: Arc<Mutex<Option<LoadedModel>>>,
    pub presets_lock: Arc<Mutex<()>>,
    pub local_client: reqwest::Client,
    pub data_dir: PathBuf,
    pub api_port: u16,
    pub client: reqwest::Client,
}

// A launch owns its credentials until all its requests and polling tasks finish.
// Never serialize or log this structure.
pub struct RuntimeSession {
    alias: String,
    key: String,
    key_path: PathBuf,
    port: u16,
    cancel: tokio::sync::Notify,
}

impl Drop for RuntimeSession {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.key_path);
    }
}

impl RuntimeSession {
    fn new(data_dir: &Path, port: u16) -> Result<Arc<Self>, String> {
        use std::io::Write;
        let alias = format!("noullama-{}", uuid::Uuid::new_v4());
        let key_path = data_dir.join(format!("runtime-{}.key", uuid::Uuid::new_v4()));
        let key = format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&key_path)
            .map_err(|e| format!("Cannot create runtime credentials: {e}"))?;
        if let Err(error) = writeln!(file, "{key}") {
            let _ = std::fs::remove_file(&key_path);
            return Err(format!("Cannot write runtime credentials: {error}"));
        }
        Ok(Arc::new(Self {
            alias,
            key,
            key_path,
            port,
            cancel: tokio::sync::Notify::new(),
        }))
    }
}

#[derive(Clone, Serialize)]
pub struct LoadedModel {
    pub path: String,
    pub device: String,
    pub runtime_path: String,
}

#[derive(Clone, Default, Serialize)]
pub struct DownloadState {
    pub active: bool,
    pub cancelled: bool,
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
        let (settings, startup_warning) = match std::fs::read(&settings_path) {
            Ok(bytes) => match serde_json::from_slice::<Settings>(&bytes) {
                Ok(settings) => (settings, None),
                Err(error) => {
                    use std::io::Write;
                    let backup =
                        data_dir.join(format!("settings-{}.json.bak", uuid::Uuid::new_v4()));
                    let mut file = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&backup)?;
                    file.write_all(&bytes)?;
                    (Settings::default(), Some(format!("Settings could not be read ({error}). Using defaults. Your original settings were preserved at {}.", backup.display())))
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                (Settings::default(), None)
            }
            Err(error) => {
                return Err(format!("Cannot read {}: {error}", settings_path.display()).into())
            }
        };
        Ok(Self {
            app: Default::default(),
            runtimes: Default::default(),
            api_port: settings.api_port,
            settings: Arc::new(Mutex::new(settings)),
            child: Default::default(),
            lifecycle: Default::default(),
            session: Default::default(),
            startup_warning,
            logs: Default::default(),
            token: Default::default(),
            download: Default::default(),
            ready: Default::default(),
            api_error: Default::default(),
            api_notice: Default::default(),
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
    pub fn listening_port(&self) -> u16 {
        let active = self
            .app
            .active_api_port
            .load(std::sync::atomic::Ordering::SeqCst);
        if active == 0 {
            self.api_port
        } else {
            active
        }
    }
    pub async fn shutdown(&self) {
        let _guard = self.lifecycle.lock().await;
        self.shutdown_locked().await;
    }
    async fn shutdown_locked(&self) {
        if let Some(session) = self.session.lock().await.take() {
            session.cancel.notify_one();
        }
        *self.ready.lock().await = false;
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        *self.loaded_model.lock().await = None;
    }
    pub async fn infer(&self, mut payload: Value) -> Result<Value, String> {
        validate_request(&payload)?;
        if !*self.ready.lock().await {
            return Err("Load a decision model before running inference.".into());
        }
        let session = self
            .session
            .lock()
            .await
            .clone()
            .ok_or("The runtime has stopped.")?;
        payload["model"] = json!(session.alias);
        let response = self
            .local_client
            .post(format!("http://127.0.0.1:{}/v1/systemone", session.port))
            .bearer_auth(&session.key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Runtime connection failed: {e}"))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| e.to_string())?
            .replace(&session.key, "[redacted]");
        let mut value: Value = serde_json::from_str(&body).map_err(|_| {
            format!(
                "Runtime returned non-JSON ({status}): {}",
                body.chars().take(500).collect::<String>()
            )
        })?;
        if !status.is_success() {
            return Err(format!("Runtime {status}: {value}"));
        }
        if let Some(model) = value.get_mut("model") {
            *model = json!("noullama");
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
        // Distribution packages and the usual CUDA toolkit locations.
        "/usr/lib64",
        "/usr/lib64/nvidia",
        "/usr/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/targets/aarch64-linux/lib",
        "/opt/cuda/lib64",
        "/opt/cuda/targets/x86_64-linux/lib",
        // Shared CUDA runtimes shipped beside an Ollama install.
        "/usr/local/lib/ollama/cuda_v12",
        "/usr/local/lib/ollama/cuda_v13",
        "/usr/lib/ollama",
    ] {
        let path = PathBuf::from(path);
        if has_cuda_runtime(&path) && !paths.contains(&path) {
            paths.push(path);
        }
    }
    paths.retain(|path| !path.as_os_str().is_empty());
    paths
}

#[cfg(target_os = "linux")]
fn has_cuda_runtime(directory: &Path) -> bool {
    ["12", "13"].iter().any(|major| {
        ["libcudart", "libcublas", "libcublasLt"]
            .iter()
            .all(|name| directory.join(format!("{name}.so.{major}")).is_file())
    })
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
    state_snapshot(state.inner()).await
}

pub async fn state_snapshot(state: &AppState) -> Result<Value, String> {
    let _guard = state.lifecycle.lock().await;
    let mut child = state.child.lock().await;
    let running = if let Some(c) = child.as_mut() {
        c.try_wait().map_err(|e| e.to_string())?.is_none()
    } else {
        state.session.lock().await.is_some()
    };
    if !running {
        *child = None;
        *state.ready.lock().await = false;
        *state.loaded_model.lock().await = None;
        if let Some(session) = state.session.lock().await.take() {
            session.cancel.notify_one();
        }
    }
    drop(child);
    let settings = state.settings.lock().await.clone();
    let logs = state.logs.lock().await.clone();
    let download = state.download.lock().await.clone();
    let loaded_model = state.loaded_model.lock().await.clone();
    let ready = *state.ready.lock().await;
    let api_error = state.api_error.lock().await.clone();
    let api_notice = state.api_notice.lock().await.clone();
    let runtime_download = state.runtimes.progress.lock().await.clone();
    let hardware = crate::hardware::hardware();
    let fit_path = loaded_model
        .as_ref()
        .map(|model| model.path.clone())
        .unwrap_or_else(|| settings.model_path.clone());
    let fit = crate::hardware::fit(&fit_path, settings.context_size, settings.gpu_layers);
    Ok(
        json!({"settings": settings, "running": running, "ready": ready,
        "logs": logs, "download": download, "loaded_model":loaded_model,
        "download_dir": model_directory(&settings, &state.data_dir),
        "api_port": state.listening_port(), "api_error": api_error, "api_notice": api_notice, "runtime_download": runtime_download, "startup_warning": state.startup_warning, "quit_confirmation":state.app.quit_confirmation.load(std::sync::atomic::Ordering::SeqCst), "data_dir": state.data_dir, "hardware": hardware, "fit": fit}),
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
    if state.session.lock().await.is_some()
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
    let active_api = state
        .app
        .active_api_port
        .load(std::sync::atomic::Ordering::SeqCst);
    if active_api != 0 && settings.runtime_port == active_api {
        return Err(format!(
            "The private runtime port must differ from the active local API port ({active_api})."
        ));
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
    if state.app.exiting.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("Noullama is shutting down.".into());
    }
    if state.session.lock().await.is_some() || state.child.lock().await.is_some() {
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
    let session = RuntimeSession::new(&state.data_dir, settings.runtime_port)?;
    *state.session.lock().await = Some(session.clone());
    *state.ready.lock().await = false;
    state
        .log("Loading model; checking the runtime and waiting for readiness…".into())
        .await;
    let state = state.clone();
    tokio::spawn(async move {
        let result = tokio::select! {
            result = tokio::time::timeout(Duration::from_secs(90), prepare_and_poll(&state, &settings, &session)) => result,
            _ = session.cancel.notified() => return,
        };
        let error = match result {
            Ok(Ok(())) => return,
            Ok(Err(error)) => error,
            Err(_) => "Model loading timed out after 90 seconds.".into(),
        };
        let _guard = state.lifecycle.lock().await;
        if is_current_session(&state, &session).await {
            state.log(error.replace(&session.key, "[redacted]")).await;
            state.shutdown_locked().await;
        }
    });
    Ok(())
}

async fn is_current_session(state: &AppState, session: &RuntimeSession) -> bool {
    state
        .session
        .lock()
        .await
        .as_ref()
        .is_some_and(|current| current.alias == session.alias)
}

fn runtime_args(settings: &Settings, session: &RuntimeSession) -> Vec<String> {
    let mut args = vec![
        "--model".into(),
        settings.model_path.clone(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        session.port.to_string(),
        "--alias".into(),
        session.alias.clone(),
        "--api-key-file".into(),
        session.key_path.to_string_lossy().into_owned(),
        "--ctx-size".into(),
        settings.context_size.to_string(),
        "--batch-size".into(),
        settings.context_size.min(2048).to_string(),
        "--ubatch-size".into(),
        settings.context_size.min(512).to_string(),
        "--n-gpu-layers".into(),
        settings.gpu_layers.to_string(),
        "--parallel".into(),
        "1".into(),
    ];
    if !settings.mmproj_path.is_empty() {
        args.extend(["--mmproj".into(), settings.mmproj_path.clone()]);
    }
    args
}

async fn prepare_and_poll(
    state: &AppState,
    settings: &Settings,
    session: &Arc<RuntimeSession>,
) -> Result<(), String> {
    // Keep the port reserved during preflight, then verify the launch's random
    // alias over an authenticated endpoint before ever accepting inference.
    let reservation = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, session.port))
        .await
        .map_err(|e| format!("Private runtime port is unavailable: {e}"))?;
    let devices = tokio::time::timeout(
        Duration::from_secs(15),
        runtime_command(settings)
            .kill_on_drop(true)
            .arg("--list-devices")
            .output(),
    )
    .await
    .map_err(|_| "GPU detection timed out")?
    .map_err(|e| format!("Cannot detect runtime devices: {e}"))?;
    if !devices.status.success() {
        return Err(
            "Runtime device detection failed. Check the runtime path and shared libraries.".into(),
        );
    }
    let devices = format!(
        "{}{}",
        String::from_utf8_lossy(&devices.stdout),
        String::from_utf8_lossy(&devices.stderr)
    );
    let gpu = devices
        .lines()
        .find(|line| {
            ["CUDA", "Vulkan", "Metal", "ROCm", "SYCL"]
                .iter()
                .any(|prefix| line.trim_start().starts_with(prefix))
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
    {
        let _guard = state.lifecycle.lock().await;
        if !is_current_session(state, session).await {
            return Ok(());
        }
        drop(reservation);
        let mut child = runtime_command(settings)
            .args(runtime_args(settings, session))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                format!(
                    "Could not start {}: {e}",
                    server_binary(&settings.runtime_path).display()
                )
            })?;
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
            let key = session.key.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(pipe).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    state.log(line.replace(&key, "[redacted]")).await;
                }
            });
        }
        *state.child.lock().await = Some(child);
        *state.loaded_model.lock().await = Some(LoadedModel {
            path: settings.model_path.clone(),
            device: device.clone(),
            runtime_path: settings.runtime_path.clone(),
        });
        state
            .log(format!("Loading {} on {device}", settings.model_path))
            .await;
    }
    loop {
        {
            let _guard = state.lifecycle.lock().await;
            if !is_current_session(state, session).await {
                return Ok(());
            }
            let mut child = state.child.lock().await;
            if !matches!(child.as_mut().map(|c| c.try_wait()), Some(Ok(None))) {
                return Err(
                    "Runtime exited during startup. Check the log for model/runtime compatibility."
                        .into(),
                );
            }
        }
        // /health may be public in llama.cpp. Only the authenticated model
        // listing and the unpredictable per-launch alias establish readiness.
        let models = state
            .local_client
            .get(format!("http://127.0.0.1:{}/v1/models", session.port))
            .bearer_auth(&session.key)
            .timeout(Duration::from_secs(2))
            .send()
            .await;
        if let Ok(response) = models {
            if response.status().is_success() {
                let models = response
                    .json::<Value>()
                    .await
                    .map_err(|_| "Runtime returned an invalid model listing")?;
                if !models["data"]
                    .as_array()
                    .is_some_and(|models| models.iter().any(|model| model["id"] == session.alias))
                {
                    return Err(
                        "Runtime identity did not match this launch; refusing to connect.".into(),
                    );
                }
                let decision = models["data"].as_array().is_some_and(|models| {
                    models.iter().any(|model| {
                        model["id"] == session.alias
                            && model["architecture"]["output_modalities"]
                                .as_array()
                                .is_some_and(|a| a.iter().any(|x| x == "decisions"))
                    })
                });
                if !decision {
                    return Err("Runtime did not advertise native decision support. Use a decision GGUF and a compatible llama.cpp build.".into());
                }
                let _guard = state.lifecycle.lock().await;
                if !is_current_session(state, session).await {
                    return Ok(());
                }
                let mut child = state.child.lock().await;
                if !matches!(child.as_mut().map(|c| c.try_wait()), Some(Ok(None))) {
                    return Err("Runtime exited before becoming ready.".into());
                }
                *state.ready.lock().await = true;
                state
                    .log("Decision model is ready. Local API is accepting requests.".into())
                    .await;
                return Ok(());
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

#[tauri::command]
pub async fn stop_runtime(state: tauri::State<'_, AppState>) -> Result<(), String> {
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
    #[cfg(unix)]
    #[tokio::test]
    async fn runtime_credentials_are_private_temporary_and_absent_from_snapshots() {
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("noullama-key-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let session = RuntimeSession::new(&base, 18439).unwrap();
        let key_path = session.key_path.clone();
        assert_eq!(
            std::fs::metadata(&key_path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            std::fs::read_to_string(&key_path).unwrap().trim(),
            session.key
        );
        *state.session.lock().await = Some(session.clone());
        let snapshot = state_snapshot(&state).await.unwrap().to_string();
        assert!(!snapshot.contains(&session.key));
        let args = runtime_args(&Settings::default(), &session);
        assert!(!args.iter().any(|arg| arg.contains(&session.key)));
        state.shutdown().await;
        drop(session);
        assert!(!key_path.exists());
        std::fs::remove_dir_all(base).unwrap();
    }
    #[cfg(unix)]
    fn fake_runtime(base: &Path, name: &str, script: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = base.join(name);
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stop_cancels_preflight_without_blocking_settings_or_a_replacement_launch() {
        let base = std::env::temp_dir().join(format!("noullama-cancel-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let runtime = fake_runtime(&base, "slow-runtime", "#!/bin/sh\nexec sleep 60\n");
        let model = base.join("model.gguf");
        std::fs::write(&model, "test").unwrap();
        let port = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        {
            let mut settings = state.settings.lock().await;
            settings.runtime_path = runtime.to_string_lossy().into_owned();
            settings.model_path = model.to_string_lossy().into_owned();
            settings.runtime_port = port;
            settings.gpu_layers = 0;
        }
        load_model(&state).await.unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        let snapshot = tokio::time::timeout(Duration::from_millis(500), state_snapshot(&state))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(snapshot["running"], true);
        let old = state.session.lock().await.clone().unwrap();
        let mut settings = state.settings.lock().await.clone();
        settings.download_dir = base.join("downloads").to_string_lossy().into_owned();
        tokio::time::timeout(
            Duration::from_millis(500),
            persist_settings(&state, settings),
        )
        .await
        .unwrap()
        .unwrap();
        tokio::time::timeout(Duration::from_millis(500), state.shutdown())
            .await
            .unwrap();
        assert!(!is_current_session(&state, &old).await);
        drop(old);
        load_model(&state).await.unwrap();
        let replacement = state.session.lock().await.clone().unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(is_current_session(&state, &replacement).await);
        assert!(state.child.lock().await.is_none());
        state.shutdown().await;
        drop(replacement);
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert!(!std::fs::read_dir(&base)
            .unwrap()
            .flatten()
            .any(|entry| entry.path().extension().is_some_and(|e| e == "key")));
        std::fs::remove_dir_all(base).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unrelated_listener_cannot_satisfy_runtime_readiness() {
        let base = std::env::temp_dir().join(format!("noullama-identity-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let runtime = fake_runtime(
            &base,
            "idle-runtime",
            "#!/bin/sh\nif [ \"$1\" = \"--list-devices\" ]; then exit 0; fi\nexec sleep 60\n",
        );
        let model = base.join("model.gguf");
        std::fs::write(&model, "test").unwrap();
        let port = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        {
            let mut settings = state.settings.lock().await;
            settings.runtime_path = runtime.to_string_lossy().into_owned();
            settings.model_path = model.to_string_lossy().into_owned();
            settings.runtime_port = port;
            settings.gpu_layers = 0;
        }
        load_model(&state).await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while state.child.lock().await.is_none() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
            .await
            .unwrap();
        let app = axum::Router::new().route("/v1/models", axum::routing::get(|| async {
            axum::Json(json!({"data":[{"id":"noullama", "architecture":{"output_modalities":["decisions"]}}]}))
        }));
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        tokio::time::timeout(Duration::from_secs(3), async {
            while state.session.lock().await.is_some() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(!*state.ready.lock().await);
        assert!(state.child.lock().await.is_none());
        assert!(state
            .logs
            .lock()
            .await
            .iter()
            .any(|line| line.contains("identity did not match")));
        server.abort();
        let _ = server.await;
        state.shutdown().await;
        std::fs::remove_dir_all(base).unwrap();
    }
    #[test]
    fn malformed_settings_are_preserved_and_defaults_recover() {
        for contents in [r#"{"context_size":"8192"}"#, r#"{"model_path":"truncated"#] {
            let base =
                std::env::temp_dir().join(format!("noullama-recovery-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&base).unwrap();
            std::fs::write(base.join("settings.json"), contents).unwrap();
            let state =
                AppState::new(base.clone()).expect("Malformed settings must not prevent startup");
            assert_eq!(state.settings.try_lock().unwrap().context_size, 8192);
            let backups: Vec<_> = std::fs::read_dir(&base)
                .unwrap()
                .flatten()
                .filter(|entry| entry.file_name().to_string_lossy().ends_with(".bak"))
                .collect();
            assert_eq!(backups.len(), 1);
            assert_eq!(
                std::fs::read_to_string(backups[0].path()).unwrap(),
                contents
            );
            std::fs::remove_dir_all(base).unwrap();
        }
    }
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
    #[cfg(target_os = "linux")]
    #[test]
    fn cuda_runtime_detection_accepts_each_supported_major_version() {
        let base = std::env::temp_dir().join(format!("noullama-cuda-{}", uuid::Uuid::new_v4()));
        for major in ["12", "13"] {
            let directory = base.join(major);
            std::fs::create_dir_all(&directory).unwrap();
            for name in ["libcudart", "libcublas", "libcublasLt"] {
                std::fs::write(directory.join(format!("{name}.so.{major}")), "").unwrap();
            }
            assert!(has_cuda_runtime(&directory));
        }
        // A partially installed toolkit must not shadow the system loader path.
        assert!(!has_cuda_runtime(&base));
        std::fs::remove_dir_all(base).unwrap();
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
    async fn settings_survive_an_application_restart() {
        let base = std::env::temp_dir().join(format!("noullama-persist-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let mut saved = state.settings.lock().await.clone();
        saved.runtime_path = "/opt/llama.cpp/llama-server".into();
        saved.model_path = "/models/demo.gguf".into();
        saved.context_size = 4096;
        saved.gpu_layers = 12;
        saved.api_port = 12111;
        saved.runtime_port = 12112;
        saved.hf_path = "/usr/local/bin/hf".into();
        saved.library_path = "/opt/cuda/lib64".into();
        persist_settings(&state, saved.clone()).await.unwrap();
        drop(state);

        // A second AppState reads the same directory the way a fresh launch does.
        let restarted = AppState::new(base.clone()).unwrap();
        let loaded = restarted.settings.lock().await.clone();
        assert!(restarted.startup_warning.is_none());
        assert_eq!(loaded.runtime_path, saved.runtime_path);
        assert_eq!(loaded.model_path, saved.model_path);
        assert_eq!(loaded.context_size, saved.context_size);
        assert_eq!(loaded.gpu_layers, saved.gpu_layers);
        assert_eq!(loaded.api_port, saved.api_port);
        assert_eq!(loaded.runtime_port, saved.runtime_port);
        assert_eq!(loaded.hf_path, saved.hf_path);
        assert_eq!(loaded.library_path, saved.library_path);
        std::fs::remove_dir_all(base).unwrap();
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
