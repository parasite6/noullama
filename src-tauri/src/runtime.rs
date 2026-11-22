use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, VecDeque},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
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
    // Multi-token prediction: speculative decoding with the model's own MTP
    // layers, or with a separate `mtp-*.gguf` head when `mtp_path` is set.
    pub mtp: bool,
    pub mtp_path: String,
    // How decisions are read: "auto" (from the GGUF's metadata), "native"
    // (llama.cpp's /v1/systemone) or "winnow" (Winnow's prompt layout).
    // The UI labels them "Auto", "Built-in" and "Prompt layout".
    pub decision_mode: String,
    pub context_size: u32,
    pub gpu_layers: i32,
    pub api_port: u16,
    pub runtime_port: u16,
    pub hf_path: String,
    pub download_dir: String,
    pub library_path: String,
    // Aliases saved by an earlier version, keyed by model path. They moved to
    // models.yml (see library.rs) and are only read once to create it.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub model_routes: BTreeMap<String, Vec<String>>,
    // sha256 checks before a load: "off", "once", "every_load" or "warn".
    // Changed only by set_sha256_check.
    pub sha256_check: String,
    // Keep context for new loads: whether a new instance starts carrying
    // earlier decisions into each request's state (see context.rs). Each
    // loaded instance then has its own switch (Instance::keeps_context).
    // Saved by the Local API load panel.
    pub keep_context: bool,
    pub allow_force_fit: bool,
    pub hide_crowded_load_warning: bool,
    // When slot 1 of a file is not loaded, its unsuffixed names reach the
    // loaded instance with the lowest slot instead of answering 503.
    // Changed only by set_fall_through.
    pub fall_through_to_other_instances: bool,
    // What a request for `any-available` does when every ready instance is
    // running a request: "reject" (503) or "wait" (up to ANY_AVAILABLE_WAIT).
    // Changed only by set_any_available_when_busy.
    pub any_available_when_busy: String,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            runtime_path: String::new(),
            model_path: String::new(),
            mmproj_path: String::new(),
            mtp: false,
            mtp_path: String::new(),
            decision_mode: "auto".into(),
            context_size: 8192,
            gpu_layers: 99,
            api_port: 11435,
            runtime_port: 11436,
            hf_path: "hf".into(),
            download_dir: String::new(),
            library_path: String::new(),
            model_routes: BTreeMap::new(),
            sha256_check: "once".into(),
            keep_context: false,
            allow_force_fit: false,
            hide_crowded_load_warning: false,
            fall_through_to_other_instances: false,
            any_available_when_busy: "reject".into(),
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    pub app: Arc<crate::lifecycle::AppControl>,
    pub runtimes: Arc<crate::runtimes::RuntimeManager>,
    pub settings: Arc<Mutex<Settings>>,
    pub lifecycle: Arc<Mutex<()>>,
    // Loaded models, oldest first. The Playground uses the first ready one
    // until another is picked; API requests always name theirs.
    pub instances: Arc<Mutex<Vec<Arc<Instance>>>>,
    /// Wakes the supervisor (`start_memory_guard`) when a load registers an instance.
    pub instances_added: Arc<tokio::sync::Notify>,
    pub startup_warning: Option<String>,
    pub logs: Arc<Mutex<VecDeque<String>>>,
    pub token: Arc<Mutex<Option<String>>>,
    // Model downloads, oldest first; several can run at once, one per file.
    pub downloads: Arc<Mutex<Vec<DownloadState>>>,
    pub api_error: Arc<Mutex<Option<String>>>,
    pub api_notice: Arc<Mutex<Option<String>>>,
    // Why recent loads failed or models stopped on their own, newest last.
    pub load_errors: Arc<Mutex<Vec<LoadError>>>,
    pub presets_lock: Arc<Mutex<()>>,
    // models.yml, and the sha256 work around it.
    pub library: Arc<Mutex<crate::library::Library>>,
    pub hashing: Arc<Mutex<std::collections::BTreeSet<String>>>,
    // Model paths whose file no longer matches its recorded sha256, with the new sum.
    pub mismatches: Arc<Mutex<BTreeMap<String, String>>>,
    pub hash_worker: Arc<AtomicBool>,
    pub hash_cache_lock: Arc<Mutex<()>>,
    pub local_client: reqwest::Client,
    pub data_dir: PathBuf,
    pub api_port: u16,
    pub client: reqwest::Client,
    /// Stamps each request an instance starts, so `any-available` can pick
    /// the instance idle the longest.
    pub request_seq: Arc<AtomicU64>,
    /// Wakes `any-available` requests waiting for an instance: one when a
    /// request finishes or an instance becomes ready, all when one unloads.
    pub instance_freed: Arc<tokio::sync::Notify>,
}

// A launch owns its credentials until all its requests and polling tasks finish.
// Never serialize or log this structure.
pub struct RuntimeSession {
    alias: String,
    key: String,
    key_path: PathBuf,
    port: u16,
    cancel: tokio::sync::Notify,
    /// When the runtime last wrote any output; loading only times out on silence.
    last_output: std::sync::Mutex<std::time::Instant>,
    /// Set when the model is read through a prompt layout instead of native
    /// decision support (see `decision.rs`).
    readout: std::sync::OnceLock<crate::decision::Readout>,
    /// One readout request at a time on the single runtime slot.
    readout_lock: Mutex<()>,
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
            last_output: std::sync::Mutex::new(std::time::Instant::now()),
            readout: std::sync::OnceLock::new(),
            readout_lock: Mutex::new(()),
        }))
    }
}

#[derive(Clone, Serialize)]
pub struct LoadedModel {
    pub path: String,
    pub device: String,
    pub runtime_path: String,
    // Loaded with a multimodal projector, so requests may carry images.
    pub projector: bool,
    // Loaded with multi-token prediction (speculative decoding) on.
    pub mtp: bool,
    // How decisions are read once ready: "native" or a layout such as "winnow-v1".
    pub layout: Option<String>,
}

#[derive(Clone, Serialize)]
pub struct LoadError {
    /// The instance that failed or stopped; errors are dismissed by it.
    pub id: String,
    pub path: String,
    pub slot: u32,
    pub name: String,
    pub message: String,
}

/// The model identifier that reaches any ready instance not running a request.
pub const ANY_AVAILABLE: &str = "any-available";

/// How long a request for `any-available` waits for a free instance when
/// `any_available_when_busy` is "wait".
pub const ANY_AVAILABLE_WAIT: Duration = Duration::from_secs(120);

/// Whether a name is `any-available`, ignoring case like every other name collision.
pub fn is_reserved(name: &str) -> bool {
    name.eq_ignore_ascii_case(ANY_AVAILABLE)
}

/// Why a name, alias or identifier cannot be `any-available`.
pub fn reserved_message(what: &str) -> String {
    format!("\"{ANY_AVAILABLE}\" is reserved: a request that sends it goes to any free loaded instance. Choose another {what}.")
}

/// The suffix a new instance in `slot` starts with: none for slot 1, `-i-N` after.
pub fn default_suffix(slot: u32) -> String {
    if slot <= 1 { String::new() } else { format!("-i-{slot}") }
}

/// The lowest slot no loaded instance of the file holds.
pub fn free_slot(taken: &[u32]) -> u32 {
    (1..).find(|slot| !taken.contains(slot)).unwrap_or(1)
}

/// `preferred` (a reload's old slot) when it is free, else the lowest free slot.
pub fn pick_slot(taken: &[u32], preferred: Option<u32>) -> u32 {
    preferred.filter(|s| *s >= 1 && !taken.contains(s)).unwrap_or_else(|| free_slot(taken))
}

/// A log label unique among loaded models: the file name for slot 1, else
/// `name#N`, starting at the slot number.
pub fn instance_label(stem: &str, slot: u32, taken: &[String]) -> String {
    (slot.max(1)..)
        .map(|n| if n == 1 { stem.to_owned() } else { format!("{stem}#{n}") })
        .find(|label| !taken.contains(label))
        .unwrap_or_else(|| stem.to_owned())
}

/// The identifiers one instance answers to: each of the file's names with the
/// suffix added, names first. Each is `(identifier, kind, base name)`.
pub fn identifiers(name: &str, aliases: &[String], suffix: &str) -> Vec<(String, &'static str, String)> {
    std::iter::once((name, "name"))
        .chain(aliases.iter().map(|a| (a.as_str(), "alias")))
        .map(|(base, kind)| (format!("{base}{suffix}"), kind, base.to_owned()))
        .collect()
}

/// Checks a new suffix for the instance in `slot` whose file answers to
/// `name` and `aliases`. `taken` holds every other name a loaded instance
/// answers to, with a description of who; `library` every models.yml name
/// and alias with its entry's name. Returns the trimmed suffix.
pub fn check_suffix(slot: u32, suffix: &str, name: &str, aliases: &[String], taken: &[(String, String)], library: &[(String, String)]) -> Result<String, String> {
    if slot <= 1 {
        return Err("Instance 1 answers to the plain names; its suffix cannot be changed.".into());
    }
    let suffix = suffix.trim();
    if suffix.is_empty() {
        return Err(format!("Enter a suffix for instance {slot}, such as {}.", default_suffix(slot)));
    }
    // The slot's own default is always accepted, as when it was loaded.
    if suffix == default_suffix(slot) {
        return Ok(suffix.to_owned());
    }
    for (identifier, _, _) in identifiers(name, aliases, suffix) {
        if !crate::library::valid_name(&identifier) {
            return Err(format!("Model name \"{identifier}\" can use letters, digits and . _ : / - only, up to 64 characters."));
        }
        if is_reserved(&identifier) {
            return Err(reserved_message("suffix"));
        }
        if let Some((_, entry)) = library.iter().find(|(other, _)| other.eq_ignore_ascii_case(&identifier)) {
            return Err(format!("\"{identifier}\" is already a name or alias of {entry} in models.yml. Choose another suffix."));
        }
        if let Some((_, owner)) = taken.iter().find(|(other, _)| other.eq_ignore_ascii_case(&identifier)) {
            return Err(format!("\"{identifier}\" already reaches {owner}. Choose another suffix."));
        }
    }
    Ok(suffix.to_owned())
}

/// A model file's name without `.gguf`, which the local API also accepts.
pub fn file_stem(path: &str) -> String {
    let name = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    match name.len().checked_sub(5) {
        Some(end) if name[end..].eq_ignore_ascii_case(".gguf") => name[..end].to_owned(),
        _ => name,
    }
}

/// One loaded model: its own llama-server on its own private port.
pub struct Instance {
    pub id: String,
    /// Unique among loaded models; prefixes this model's log lines.
    pub label: String,
    /// Which instance of its file this is: 1, 2, 3… (lowest free at load).
    pub slot: u32,
    /// Added to each of the file's names to make this instance's API
    /// identifiers. Empty for slot 1; held only while it is loaded.
    suffix: std::sync::Mutex<String>,
    /// The settings it was loaded with. Later edits only apply to new loads.
    pub settings: Settings,
    session: Arc<RuntimeSession>,
    child: Mutex<Option<Child>>,
    ready: AtomicBool,
    model: std::sync::Mutex<LoadedModel>,
    /// Earlier decisions for Keep context, kept per model.
    pub context: Mutex<crate::context::History>,
    /// This instance's Keep context switch. Starts from the `keep_context`
    /// setting at load; held only while it is loaded, never saved.
    keep_context: AtomicBool,
    /// Extra exact names for this instance only (no suffix added). Held only
    /// while it is loaded, never saved; a reload starts with none.
    aliases: std::sync::Mutex<Vec<String>>,
    /// Requests running on this instance right now (API and Playground).
    in_flight: AtomicUsize,
    /// `request_seq` when it last started a request; 0 when it never has.
    last_started: AtomicU64,
}

impl Instance {
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }
    pub fn path(&self) -> &str {
        &self.settings.model_path
    }
    pub fn stem(&self) -> String {
        file_stem(&self.settings.model_path)
    }
    pub fn suffix(&self) -> String {
        self.suffix.lock().map(|s| s.clone()).unwrap_or_else(|e| e.into_inner().clone())
    }
    fn set_suffix(&self, suffix: String) {
        *self.suffix.lock().unwrap_or_else(|e| e.into_inner()) = suffix;
    }
    pub fn keeps_context(&self) -> bool {
        self.keep_context.load(Ordering::SeqCst)
    }
    /// This instance's own aliases, in the order they were added.
    pub fn aliases(&self) -> Vec<String> {
        self.aliases.lock().map(|a| a.clone()).unwrap_or_else(|e| e.into_inner().clone())
    }
    fn set_aliases(&self, aliases: Vec<String>) {
        *self.aliases.lock().unwrap_or_else(|e| e.into_inner()) = aliases;
    }
    /// Whether a request is running on it right now.
    pub fn is_busy(&self) -> bool {
        self.in_flight.load(Ordering::SeqCst) > 0
    }
    /// Turns this instance's Keep context on or off; off empties its history.
    pub async fn set_keep_context(&self, on: bool) {
        let mut history = self.context.lock().await;
        self.keep_context.store(on, Ordering::SeqCst);
        if !on {
            history.clear();
        }
    }
    pub fn model(&self) -> LoadedModel {
        self.model.lock().map(|m| m.clone()).unwrap_or_else(|e| e.into_inner().clone())
    }
    pub fn prefix(&self) -> String {
        format!("[{}] ", self.label)
    }
    fn load_marker(&self) -> String {
        format!("{}{LOAD_MARKER}", self.prefix())
    }
    /// Stops the runtime and forgets this model's kept context.
    async fn stop(&self) {
        self.session.cancel.notify_one();
        self.ready.store(false, Ordering::SeqCst);
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        self.context.lock().await.clear();
    }
}

/// A request running on one instance. It counts toward the instance's
/// `in_flight` from creation until it is dropped, including when the caller
/// goes away mid-request, and then wakes one waiting `any-available` request.
pub struct Claim {
    pub instance: Arc<Instance>,
    freed: Arc<tokio::sync::Notify>,
}

impl Claim {
    /// Counts a request on `instance` whether or not it is busy already.
    fn take(state: &AppState, instance: &Arc<Instance>) -> Self {
        instance.in_flight.fetch_add(1, Ordering::SeqCst);
        Self::stamped(state, instance)
    }
    /// Claims `instance` only when no request is running on it.
    fn take_idle(state: &AppState, instance: &Arc<Instance>) -> Option<Self> {
        instance.in_flight.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst).ok()?;
        Some(Self::stamped(state, instance))
    }
    fn stamped(state: &AppState, instance: &Arc<Instance>) -> Self {
        instance.last_started.store(state.request_seq.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
        Self { instance: instance.clone(), freed: state.instance_freed.clone() }
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        self.instance.in_flight.fetch_sub(1, Ordering::SeqCst);
        self.freed.notify_one();
    }
}

/// Why `any-available` found no instance.
pub enum AnyError {
    /// No loaded instance is ready.
    NoneReady(String),
    /// Every ready instance is running a request (setting "reject", or the wait ran out).
    Busy(String),
}

/// The idle, ready instance that started a request least recently (one that
/// never has comes first, then load order).
fn pick_idle(instances: &[Arc<Instance>]) -> Vec<&Arc<Instance>> {
    let mut idle: Vec<&Arc<Instance>> = instances.iter().filter(|i| i.is_ready() && !i.is_busy()).collect();
    idle.sort_by_key(|i| i.last_started.load(Ordering::SeqCst));
    idle
}

/// Why a model name could not be routed to a loaded model.
pub enum RouteError {
    /// The request named no model.
    Missing(String),
    /// The named model is not loaded or still loading.
    Unavailable(String),
    Unknown(String),
    /// An identifier that reaches more than one model.
    Ambiguous(String),
}

/// One name the local API knows, and where it goes.
#[derive(Clone, Serialize)]
pub struct Route {
    /// The identifier a request sends: a name or alias plus the instance suffix.
    pub name: String,
    /// "name" (its name in models.yml), "alias" (a models.yml alias) or
    /// "instance_alias" (one instance's own alias, never suffixed).
    pub kind: &'static str,
    /// The name or alias without the suffix.
    pub base: String,
    pub path: String,
    /// The loaded model it reaches, if any.
    pub instance: Option<String>,
    /// That instance's slot, or the slot this name belongs to (1 when unsuffixed).
    pub slot: u32,
    pub suffix: String,
    pub ready: bool,
    /// An unsuffixed name whose slot 1 is not loaded: it reaches the
    /// lowest-slot instance (setting on) or nothing (setting off).
    pub fallthrough: bool,
    /// Reaches more than one model, so unusable.
    pub conflict: bool,
}

/// One loaded instance, as the route table needs it.
pub struct RouteInstance<'a> {
    pub id: &'a str,
    pub path: &'a str,
    pub slot: u32,
    pub suffix: String,
    pub ready: bool,
    /// The instance's own aliases, exact.
    pub aliases: Vec<String>,
}

/// Every identifier the local API knows, in load order: each loaded
/// instance's names then aliases (suffixed) then its instance aliases, the unsuffixed names of a file
/// whose slot 1 is not loaded, then models.yml entries that are not loaded.
/// `names` gives a path's (name, aliases); `entries` lists models.yml paths.
pub fn route_table(instances: &[RouteInstance], entries: &[String], names: impl Fn(&str) -> (String, Vec<String>), fall_through: bool) -> Vec<Route> {
    let mut routes = Vec::new();
    for instance in instances {
        let (name, aliases) = names(instance.path);
        let siblings: Vec<&RouteInstance> = instances.iter().filter(|i| i.path == instance.path).collect();
        let lowest = siblings.iter().min_by_key(|i| i.slot).map(|i| i.id);
        if !siblings.iter().any(|i| i.slot == 1) && lowest == Some(instance.id) {
            for (identifier, kind, base) in identifiers(&name, &aliases, "") {
                routes.push(Route {
                    name: identifier, kind, base, path: instance.path.to_owned(),
                    instance: fall_through.then(|| instance.id.to_owned()), slot: 1, suffix: String::new(),
                    ready: fall_through && instance.ready, fallthrough: true, conflict: false,
                });
            }
        }
        for (identifier, kind, base) in identifiers(&name, &aliases, &instance.suffix) {
            routes.push(Route {
                name: identifier, kind, base, path: instance.path.to_owned(), instance: Some(instance.id.to_owned()),
                slot: instance.slot, suffix: instance.suffix.clone(), ready: instance.ready, fallthrough: false, conflict: false,
            });
        }
        for alias in &instance.aliases {
            routes.push(Route {
                name: alias.clone(), kind: "instance_alias", base: alias.clone(), path: instance.path.to_owned(), instance: Some(instance.id.to_owned()),
                slot: instance.slot, suffix: instance.suffix.clone(), ready: instance.ready, fallthrough: false, conflict: false,
            });
        }
    }
    for path in entries {
        if instances.iter().any(|i| i.path == path) {
            continue;
        }
        let (name, aliases) = names(path);
        for (identifier, kind, base) in identifiers(&name, &aliases, "") {
            routes.push(Route { name: identifier, kind, base, path: path.clone(), instance: None, slot: 1, suffix: String::new(), ready: false, fallthrough: false, conflict: false });
        }
    }
    // `any-available` is answered before the table is consulted; a file
    // named that (with models.yml unusable) must not list it a second time.
    routes.retain(|r| !is_reserved(&r.name));
    // An identifier listed twice reaches two models (two files sharing a
    // name, or a suffix that matches another model's name): unusable.
    let counts: Vec<usize> = routes.iter().map(|r| routes.iter().filter(|o| o.name == r.name).count()).collect();
    for (route, count) in routes.iter_mut().zip(counts) {
        route.conflict = count > 1;
    }
    routes
}

/// The identifiers `/v1/models` lists: those that reach a ready model without conflict.
pub fn listed_names(routes: &[Route]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for route in routes {
        if route.ready && !route.conflict && !names.contains(&route.name) {
            names.push(route.name.clone());
        }
    }
    names
}

#[derive(Clone, Default, Serialize)]
pub struct DownloadState {
    /// `repo/file`; one entry per file.
    pub id: String,
    /// Which attempt owns this entry, so a slow old task cannot overwrite a retry.
    #[serde(skip)]
    pub attempt: u64,
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
            lifecycle: Default::default(),
            instances: Default::default(),
            startup_warning,
            logs: Default::default(),
            token: Default::default(),
            downloads: Default::default(),
            api_error: Default::default(),
            api_notice: Default::default(),
            load_errors: Default::default(),
            presets_lock: Default::default(),
            library: Default::default(),
            hashing: Default::default(),
            mismatches: Default::default(),
            hash_worker: Default::default(),
            hash_cache_lock: Default::default(),
            instances_added: Default::default(),
            request_seq: Default::default(),
            instance_freed: Default::default(),
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
        let all = std::mem::take(&mut *self.instances.lock().await);
        for instance in all {
            instance.stop().await;
        }
        self.instance_freed.notify_waiters();
    }
    /// Removes one loaded model and stops its runtime. Hold the lifecycle lock.
    async fn unload_locked(&self, id: &str) -> Option<Arc<Instance>> {
        let removed = {
            let mut list = self.instances.lock().await;
            let index = list.iter().position(|i| i.id == id)?;
            list.remove(index)
        };
        removed.stop().await;
        // Waiting `any-available` requests recheck whether any instance is still ready.
        self.instance_freed.notify_waiters();
        Some(removed)
    }
    async fn record_load_error(&self, instance: &Instance, message: String) {
        let mut errors = self.load_errors.lock().await;
        errors.retain(|e| !(e.path == instance.path() && e.slot == instance.slot));
        errors.push(LoadError { id: instance.id.clone(), path: instance.path().to_owned(), slot: instance.slot, name: instance.label.clone(), message });
        let extra = errors.len().saturating_sub(8);
        errors.drain(..extra);
    }
    pub async fn instance(&self, id: &str) -> Option<Arc<Instance>> {
        self.instances.lock().await.iter().find(|i| i.id == id).cloned()
    }
    pub async fn any_ready(&self) -> bool {
        self.instances.lock().await.iter().any(|i| i.is_ready())
    }

    /// A model's name and aliases from models.yml; a file with no entry
    /// answers to its file name.
    pub async fn api_names(&self, path: &str) -> (String, Vec<String>) {
        match self.library.lock().await.entry(path) {
            Some(entry) => (entry.name.clone(), entry.aliases.clone()),
            None => (file_stem(path), Vec::new()),
        }
    }

    /// Every identifier the local API knows (see `route_table`).
    pub async fn routes(&self) -> Vec<Route> {
        crate::library::refresh(self).await;
        let fall_through = self.settings.lock().await.fall_through_to_other_instances;
        let instances = self.instances.lock().await.clone();
        let library = self.library.lock().await;
        let names = |path: &str| match library.entry(path) {
            Some(entry) => (entry.name.clone(), entry.aliases.clone()),
            None => (file_stem(path), Vec::new()),
        };
        let table: Vec<RouteInstance> = instances
            .iter()
            .map(|i| RouteInstance { id: &i.id, path: i.path(), slot: i.slot, suffix: i.suffix(), ready: i.is_ready(), aliases: i.aliases() })
            .collect();
        let entries: Vec<String> = library.entries.iter().map(|e| e.path.clone()).collect();
        route_table(&table, &entries, names, fall_through)
    }
    /// Names that reach a ready model right now, as `/v1/models` lists them:
    /// the route table's, then `any-available` when any instance is ready.
    pub async fn model_names(&self) -> Vec<String> {
        let routes = self.routes().await;
        self.with_any_available(listed_names(&routes)).await
    }
    async fn with_any_available(&self, mut names: Vec<String>) -> Vec<String> {
        if self.any_ready().await {
            names.push(ANY_AVAILABLE.to_owned());
        }
        names
    }
    /// A request for `any-available`: claims the ready instance that has been
    /// idle the longest. When every ready instance is busy, answers at once
    /// (setting "reject") or waits for the first to free up (setting "wait",
    /// up to ANY_AVAILABLE_WAIT). The claim is held until the request ends.
    pub async fn claim_any(&self) -> Result<Claim, AnyError> {
        let wait = self.settings.lock().await.any_available_when_busy == "wait";
        let deadline = tokio::time::Instant::now() + ANY_AVAILABLE_WAIT;
        loop {
            // Registered before looking, so a request that finishes in between still wakes it.
            let freed = self.instance_freed.notified();
            tokio::pin!(freed);
            freed.as_mut().enable();
            let instances = self.instances.lock().await.clone();
            if !instances.iter().any(|i| i.is_ready()) {
                return Err(AnyError::NoneReady(format!("No loaded instance is ready for \"{ANY_AVAILABLE}\". Load a model in noullama, or wait for one to finish loading.")));
            }
            if let Some(claim) = pick_idle(&instances).into_iter().find_map(|i| Claim::take_idle(self, i)) {
                self.log(format!("{}Took a request for {ANY_AVAILABLE}.", claim.instance.prefix())).await;
                return Ok(claim);
            }
            if !wait {
                return Err(AnyError::Busy(format!("Every loaded instance is busy with another request, so \"{ANY_AVAILABLE}\" has none to send this one to. Try again, or set \"When every instance is busy\" to Wait in Settings.")));
            }
            if tokio::time::timeout_at(deadline, freed).await.is_err() {
                return Err(AnyError::Busy(format!("Every loaded instance stayed busy for {} seconds, so \"{ANY_AVAILABLE}\" gave up on this request. Try again.", ANY_AVAILABLE_WAIT.as_secs())));
            }
        }
    }
    /// The loaded model a request's `model` field names: one of its file's
    /// names or aliases in models.yml plus the instance's suffix (a loaded
    /// file with no entry answers to its file name). Every request must name
    /// one, exactly as written.
    pub async fn route(&self, requested: Option<&str>) -> Result<Arc<Instance>, RouteError> {
        use RouteError::*;
        let Some(name) = requested.filter(|n| !n.is_empty()) else {
            let names = self.model_names().await;
            return Err(Missing(if names.is_empty() {
                "Set \"model\" to a loaded model's name or alias. No model is loaded yet; load one in noullama.".into()
            } else {
                format!("Set \"model\" to a loaded model's name or alias: {}.", names.join(", "))
            }));
        };
        let routes = self.routes().await;
        let instances = self.instances.lock().await.clone();
        let matches: Vec<&Route> = routes.iter().filter(|r| r.name == name).collect();
        let still_loading = |instance: &Instance| Unavailable(if instance.slot <= 1 {
            format!("{} is still loading.", instance.stem())
        } else {
            format!("{} instance {} is still loading.", instance.stem(), instance.slot)
        });
        match matches.as_slice() {
            [] => {
                let hint = routes
                    .iter()
                    .find(|r| r.name.eq_ignore_ascii_case(name))
                    .map(|r| r.name.as_str())
                    .or_else(|| is_reserved(name).then_some(ANY_AVAILABLE))
                    .map(|n| format!(" Names are case-sensitive; did you mean '{n}'?"))
                    .unwrap_or_default();
                let names = self.with_any_available(listed_names(&routes)).await;
                Err(Unknown(if names.is_empty() {
                    format!("Unknown model '{name}'.{hint} No model is ready; load one in noullama.")
                } else {
                    format!("Unknown model '{name}'.{hint} Loaded models answer to: {}.", names.join(", "))
                }))
            }
            [one] => match one.instance.as_deref().and_then(|id| instances.iter().find(|i| i.id == id)) {
                Some(instance) if instance.is_ready() => Ok(instance.clone()),
                Some(instance) => Err(still_loading(instance)),
                None if one.fallthrough => {
                    // Suffixed names never fall through; name the ones that work.
                    let others: Vec<&str> = routes
                        .iter()
                        .filter(|r| r.path == one.path && r.base == one.base && r.kind == one.kind && r.instance.is_some() && !r.fallthrough)
                        .map(|r| r.name.as_str())
                        .collect();
                    Err(Unavailable(format!(
                        "'{name}' is instance 1 of {}, which is not loaded. Other loaded instances answer to: {}. Load it again, or turn on Fall through to another instance in Settings.",
                        file_stem(&one.path),
                        others.join(", ")
                    )))
                }
                None => Err(Unavailable(format!("'{name}' is {}, which is not loaded. Load it in noullama.", file_stem(&one.path)))),
            },
            many => {
                let suffixed = many.iter().any(|r| r.slot > 1);
                let instance_alias = many.iter().any(|r| r.kind == "instance_alias");
                Err(Ambiguous(format!(
                    "'{name}' is the name of {} loaded models ({}). {}",
                    many.len(),
                    many.iter().map(|r| if r.slot > 1 { format!("{} instance {}", r.path, r.slot) } else { r.path.clone() }).collect::<Vec<_>>().join(", "),
                    if instance_alias { "Remove the instance alias in Local API, or give them different names in Library." } else if suffixed { "Give them different names in Library, or change the instance suffix in Local API." } else { "Give them different names in Library." }
                )))
            }
        }
    }
    /// Runs a decision on one loaded model, with earlier decisions added when
    /// Keep context is on.
    /// The request counts as running on it (see `Claim`) until it ends.
    pub async fn infer_on(&self, instance: &Arc<Instance>, payload: Value) -> Result<Value, String> {
        let claim = Claim::take(self, instance);
        self.infer_claimed(&claim, payload).await
    }
    /// `infer_on` for a request that already holds its instance's claim.
    pub async fn infer_claimed(&self, claim: &Claim, payload: Value) -> Result<Value, String> {
        crate::context::infer(self, &claim.instance, payload).await
    }

    /// One request to the runtime. `with_history` marks a state that carries
    /// earlier decisions, which must fit whole rather than be cut short.
    pub async fn infer_once(&self, instance: &Instance, mut payload: Value, with_history: bool) -> Result<Value, crate::context::Failure> {
        use crate::context::Failure::{ContextFull, Other};
        validate_request(&payload).map_err(Other)?;
        if !instance.is_ready() {
            return Err(Other(format!("{} has stopped. Load it again.", instance.stem())));
        }
        let session = instance.session.clone();
        if let Some(readout) = session.readout.get() {
            if has_images(&payload) {
                return Err(Other("Images are not supported for this model yet: it is read through the Prompt layout (Winnow), which is text only.".into()));
            }
            let _one_at_a_time = session.readout_lock.lock().await;
            let server = crate::decision::Server { client: &self.local_client, port: session.port, key: &session.key };
            let redact = |e: String| e.replace(&session.key, "[redacted]");
            // The Winnow layout cuts a long state short from the end, which
            // would drop the current state behind the earlier decisions.
            if with_history {
                let text = crate::decision::safe(&payload["state"]);
                if text.len() > crate::decision::STATE_TOKENS
                    && server.tokenize(&text, false, false).await.map_err(|e| Other(redact(e)))?.len() > crate::decision::STATE_TOKENS
                {
                    return Err(ContextFull(format!("The state is longer than {} tokens.", crate::decision::STATE_TOKENS)));
                }
            }
            return server.decide(readout, &payload).await.map_err(|e| {
                let e = redact(e);
                if crate::decision::is_context_overflow(&e) { ContextFull(e) } else { Other(e) }
            });
        }
        payload["model"] = json!(session.alias);
        let response = self
            .local_client
            .post(format!("http://127.0.0.1:{}/v1/systemone", session.port))
            .bearer_auth(&session.key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| Other(format!("Runtime connection failed: {e}")))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| Other(e.to_string()))?
            .replace(&session.key, "[redacted]");
        let mut value: Value = serde_json::from_str(&body).map_err(|_| {
            Other(format!(
                "Runtime returned non-JSON ({status}): {}",
                body.chars().take(500).collect::<String>()
            ))
        })?;
        if !status.is_success() {
            if status == reqwest::StatusCode::NOT_IMPLEMENTED && has_images(&payload) {
                return Err(Other("This model cannot read images as loaded. Turn on Vision for a model with a multimodal projector, then reload it.".into()));
            }
            let detail = value["error"]["message"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| value.to_string());
            let message = format!(
                "Runtime {status}: {}",
                detail.chars().take(500).collect::<String>()
            );
            // llama-server reports a prompt longer than the slot's context this way.
            return Err(if value["error"]["type"] == "exceed_context_size_error" { ContextFull(message) } else { Other(message) });
        }
        if let Some(model) = value.get_mut("model") {
            *model = json!(instance.stem());
        }
        Ok(value)
    }
}

fn has_images(payload: &Value) -> bool {
    ["images", "files"]
        .iter()
        .any(|key| payload[key].as_array().is_some_and(|list| !list.is_empty()))
}

pub fn validate_request(payload: &Value) -> Result<(), String> {
    if !payload.is_object() || payload.get("state").is_none() {
        return Err("Request needs a state field and a questions object.".into());
    }
    // The runtime takes inline data URLs only; reject anything else here so
    // the error names the field instead of surfacing as a runtime 400.
    for key in ["images", "files"] {
        let Some(list) = payload.get(key) else {
            continue;
        };
        let valid = list.as_array().is_some_and(|list| {
            list.iter().all(|item| {
                item.as_str().is_some_and(|url| {
                    (url.starts_with("data:image/") || url.starts_with("data:audio/"))
                        && url.contains(";base64,")
                })
            })
        });
        if !valid {
            return Err(format!(
                "{key} must be an array of base64 data URLs (data:image/...;base64,...)."
            ));
        }
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

#[tauri::command]
pub async fn snapshot(state: tauri::State<'_, AppState>) -> Result<Value, String> {
    state_snapshot(state.inner()).await
}

/// The load task reports its own failures; this catches a runtime that
/// stopped after it was ready. Runs from the UI snapshot and from the
/// supervisor tick, so a crash is noticed while the window is hidden too.
async fn reap_exited(state: &AppState) -> Result<(), String> {
    let _guard = state.lifecycle.lock().await;
    reap_exited_locked(state).await
}

/// `reap_exited` for a caller that already holds the lifecycle lock.
async fn reap_exited_locked(state: &AppState) -> Result<(), String> {
    let list = state.instances.lock().await.clone();
    for instance in list {
        let exited = match instance.child.lock().await.as_mut() {
            Some(child) => child.try_wait().map_err(|e| e.to_string())?,
            None => None,
        };
        if let (Some(status), true) = (exited, instance.is_ready()) {
            let message = {
                let logs = state.logs.lock().await;
                with_runtime_detail(format!("{} stopped unexpectedly ({status}).", instance.stem()), &logs, &instance.label)
            };
            state.record_load_error(&instance, message).await;
            state.unload_locked(&instance.id).await;
        }
    }
    Ok(())
}

pub async fn state_snapshot(state: &AppState) -> Result<Value, String> {
    reap_exited(state).await?;
    let settings = state.settings.lock().await.clone();
    let logs = state.logs.lock().await.clone();
    let downloads = state.downloads.lock().await.clone();
    let instances = state.instances.lock().await.clone();
    let routes = state.routes().await;
    let library = crate::library::snapshot(state).await;
    let mut loaded = Vec::new();
    let mut taken = Vec::new();
    // The fit leaves out one instance of the selected file: the one a reload replaces.
    let replaced = instances.iter().find(|i| i.path() == settings.model_path).map(|i| i.id.clone());
    for instance in &instances {
        let model = instance.model();
        let history = instance.context.lock().await;
        let extras = instance_extras(&instance.settings);
        let required = crate::hardware::estimate_bytes(instance.path(), &extras, instance.settings.context_size);
        if Some(&instance.id) != replaced.as_ref() {
            if let Some(bytes) = required {
                taken.push((bytes, instance.settings.gpu_layers != 0));
            }
        }
        let (name, _) = state.api_names(instance.path()).await;
        let api_names: Vec<&str> = routes.iter().filter(|r| r.instance.as_deref() == Some(&instance.id) && !r.fallthrough).map(|r| r.name.as_str()).collect();
        loaded.push(json!({
            "id": instance.id, "name": name, "label": instance.label,
            "slot": instance.slot, "suffix": instance.suffix(), "default_suffix": default_suffix(instance.slot), "api_names": api_names,
            "instance_aliases": instance.aliases(),
            "path": model.path, "device": model.device, "runtime_path": model.runtime_path,
            "projector": model.projector, "mmproj_path": instance.settings.mmproj_path,
            "mtp": model.mtp, "mtp_path": instance.settings.mtp_path, "layout": model.layout,
            "ready": instance.is_ready(), "context_size": instance.settings.context_size,
            "gpu_layers": instance.settings.gpu_layers, "required_bytes": required,
            "context": {"entries": history.len(), "last_flushed": history.last_flushed}, "keep_context": instance.keeps_context(),
            "settings": instance.settings,
        }));
    }
    let default = instances.iter().find(|i| i.is_ready()).or(instances.first());
    let running = !instances.is_empty();
    let ready = instances.iter().any(|i| i.is_ready());
    let loaded_model = default.map(|i| i.model());
    let api_error = state.api_error.lock().await.clone();
    let api_notice = state.api_notice.lock().await.clone();
    let load_errors = state.load_errors.lock().await.clone();
    let runtime_download = state.runtimes.progress.lock().await.clone();
    // The probe runs in the background from startup; until it finishes the
    // snapshot reports no hardware rather than waiting on nvidia-smi.
    let hardware = crate::hardware::cached();
    // The selected model's fit, beside the memory other loaded models hold.
    let fit = crate::hardware::fit_beside(
        &settings.model_path,
        &instance_extras(&settings).iter().map(String::as_str).collect::<Vec<_>>(),
        settings.context_size,
        settings.gpu_layers,
        &taken,
    );
    Ok(
        json!({"settings": settings, "running": running, "ready": ready,
        "instances": loaded, "default_model": default.map(|i| i.id.clone()), "routes": routes, "library": library,
        "logs": logs, "downloads": downloads, "loaded_model": loaded_model,
        "download_dir": model_directory(&settings, &state.data_dir),
        "api_port": state.listening_port(), "api_error": api_error, "api_notice": api_notice, "load_errors": load_errors, "runtime_download": runtime_download, "startup_warning": state.startup_warning, "quit_confirmation":state.app.quit_confirmation.load(std::sync::atomic::Ordering::SeqCst), "data_dir": state.data_dir, "hardware": hardware, "fit": fit,
        "model_missing": !settings.model_path.is_empty() && !Path::new(&settings.model_path).is_file()}),
    )
}

/// Files loaded beside the model (projector, MTP head), for memory estimates.
fn instance_extras(settings: &Settings) -> Vec<String> {
    let mut extras = vec![settings.mmproj_path.clone()];
    if settings.mtp {
        extras.push(settings.mtp_path.clone());
    }
    extras
}

#[tauri::command]
pub async fn save_settings(
    settings: Settings,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    save_from_editor(state.inner(), settings).await
}

/// The settings form sends every field, but its copies of the alias list and
/// the Settings tab switches can be stale; those have their own commands, so
/// keep the server's values here. `keep_context` (the default for new loads)
/// is a load panel field and is saved from the form.
pub async fn save_from_editor(state: &AppState, mut settings: Settings) -> Result<(), String> {
    {
        let current = state.settings.lock().await;
        settings.model_routes = current.model_routes.clone();
        settings.sha256_check = current.sha256_check.clone();
        settings.allow_force_fit = current.allow_force_fit;
        settings.hide_crowded_load_warning = current.hide_crowded_load_warning;
        settings.fall_through_to_other_instances = current.fall_through_to_other_instances;
        settings.any_available_when_busy = current.any_available_when_busy.clone();
    }
    persist_settings(state, settings).await
}

/// Trims, checks and de-duplicates one model's aliases for the local API.
pub fn normalize_aliases(aliases: &[String]) -> Result<Vec<String>, String> {
    let mut names: Vec<String> = Vec::new();
    for alias in aliases.iter().map(|a| a.trim()).filter(|a| !a.is_empty()) {
        if !crate::library::valid_name(alias) {
            return Err(format!("Model name \"{alias}\" can use letters, digits and . _ : / - only, up to 64 characters."));
        }
        if is_reserved(alias) {
            return Err(reserved_message("alias"));
        }
        if !names.iter().any(|n| n.eq_ignore_ascii_case(alias)) {
            names.push(alias.to_owned());
        }
    }
    if names.len() > 16 {
        return Err("Use at most 16 aliases for one model.".into());
    }
    Ok(names)
}

#[tauri::command]
pub async fn set_model_aliases(path: String, aliases: Vec<String>, state: tauri::State<'_, AppState>) -> Result<Vec<String>, String> {
    set_aliases(state.inner(), path, aliases).await
}

/// Sets the aliases of the model file at `path` in models.yml.
pub async fn set_aliases(state: &AppState, path: String, aliases: Vec<String>) -> Result<Vec<String>, String> {
    if path.trim().is_empty() {
        return Err("Choose a model file first.".into());
    }
    crate::library::set_aliases(state, &path, aliases).await
}

pub async fn persist_settings(state: &AppState, settings: Settings) -> Result<(), String> {
    persist_memory_settings(state, settings, None).await
}

async fn persist_memory_settings(state: &AppState, mut settings: Settings, update: Option<(&str, bool)>) -> Result<(), String> {
    // Loaded models keep the settings they started with, so these can change
    // at any time; they apply to the next load.
    let _guard = state.lifecycle.lock().await;
    if update.is_some() { settings = state.settings.lock().await.clone(); }
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
    if state.downloads.lock().await.iter().any(|d| d.active)
        && state.settings.lock().await.download_dir != settings.download_dir
    {
        return Err(
            "Wait for downloads to finish before changing their directory.".into(),
        );
    }
    if !settings.download_dir.trim().is_empty() {
        tokio::fs::create_dir_all(settings.download_dir.trim())
            .await
            .map_err(|e| format!("Cannot create download directory: {e}"))?;
    }
    {
        let current = state.settings.lock().await;
        settings.allow_force_fit = current.allow_force_fit;
        settings.hide_crowded_load_warning = current.hide_crowded_load_warning;
    }
    match update {
        Some(("allow_force_fit", value)) => settings.allow_force_fit = value,
        Some(("hide_crowded_load_warning", value)) => settings.hide_crowded_load_warning = value,
        Some(_) => return Err("Unknown memory preference".into()),
        None => {},
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

/// Loads the selected model beside any already loaded; returns its id.
#[tauri::command]
pub async fn start_runtime(slot: Option<u32>, state: tauri::State<'_, AppState>) -> Result<String, String> {
    load_model_in(state.inner(), slot).await
}

#[cfg(test)]
pub async fn load_model(state: &AppState) -> Result<String, String> {
    load_model_in(state, None).await
}

/// `slot` is the slot of the instance a reload replaced: the new instance
/// keeps it when it is free, so its suffixed names stay the same. Otherwise
/// (and for fresh loads) it takes the lowest free slot.
pub async fn load_model_in(state: &AppState, slot: Option<u32>) -> Result<String, String> {
    let selected = state.settings.lock().await.model_path.clone();
    if Path::new(&selected).is_file() {
        crate::library::ensure(state, &[selected]).await;
    }
    let _guard = state.lifecycle.lock().await;
    if state.app.exiting.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("Noullama is shutting down.".into());
    }
    let settings = state.settings.lock().await.clone();
    if !Path::new(&settings.model_path).is_file()
        || !settings.model_path.to_lowercase().ends_with(".gguf")
    {
        return Err("Select a downloaded or local .gguf decision model first.".into());
    }
    // Its plain name answers for slot 1 and for fall-through, so no slot may use it.
    if is_reserved(&state.api_names(&settings.model_path).await.0) {
        return Err(format!("This model's API name would be \"{ANY_AVAILABLE}\", which is reserved for sending a request to any free loaded instance. Give it another name in Library, or rename the file."));
    }
    if !settings.mmproj_path.is_empty() && !Path::new(&settings.mmproj_path).is_file() {
        return Err(format!("The multimodal projector {} does not exist. If models.yml lists it for this model, fix the path in Library.", settings.mmproj_path));
    }
    if settings.mtp {
        if !settings.mtp_path.is_empty() {
            if !Path::new(&settings.mtp_path).is_file() {
                return Err(format!("The MTP head {} does not exist. If models.yml lists it for this model, fix the path in Library.", settings.mtp_path));
            }
            // The Hub and the file picker go by name; the metadata decides.
            match crate::gguf::info(&settings.mtp_path) {
                Some(head) if head.is_mtp_head() => {
                    if !crate::gguf::info(&settings.model_path).is_some_and(|model| head.head_fits(&model)) {
                        return Err("The MTP head does not fit this model: its width differs. Choose the head made for this model, or turn MTP off.".into());
                    }
                }
                _ => return Err("The chosen MTP file is not an MTP head. Choose a head file made only of MTP layers, or turn MTP off.".into()),
            }
        } else if !crate::gguf::info(&settings.model_path).is_some_and(|i| i.has_built_in_mtp()) {
            return Err("This model has no built-in MTP layers. Choose an mtp- head file in Local API, or turn MTP off.".into());
        }
    }
    let loaded = state.instances.lock().await.clone();
    if let Some(reason) = load_memory_risk(&settings, 0, 0).await {
        return Err(format!("OOM prevention: {reason} Eject another model, reduce context or GPU offload, or enable Allow force fit in Settings."));
    }
    let port = free_runtime_port(state, &settings, &loaded)?;
    let (slot, label) = slot_and_label(&loaded, &settings.model_path, slot);
    let session = RuntimeSession::new(&state.data_dir, port)?;
    let instance = Arc::new(Instance {
        id: uuid::Uuid::new_v4().simple().to_string()[..12].to_owned(),
        label,
        slot,
        suffix: std::sync::Mutex::new(default_suffix(slot)),
        model: std::sync::Mutex::new(LoadedModel {
            path: settings.model_path.clone(),
            device: String::new(),
            runtime_path: settings.runtime_path.clone(),
            projector: !settings.mmproj_path.is_empty(),
            mtp: settings.mtp,
            layout: None,
        }),
        keep_context: AtomicBool::new(settings.keep_context),
        settings,
        session,
        child: Default::default(),
        ready: AtomicBool::new(false),
        context: Default::default(),
        aliases: Default::default(),
        in_flight: Default::default(),
        last_started: Default::default(),
    });
    state.load_errors.lock().await.retain(|e| !(e.path == instance.path() && e.slot == instance.slot));
    state.instances.lock().await.push(instance.clone());
    state.instances_added.notify_one();
    state.log(instance.load_marker()).await;
    let id = instance.id.clone();
    let state = state.clone();
    tokio::spawn(async move {
        let result = tokio::select! {
            result = tokio::time::timeout(LOAD_LIMIT, prepare_and_poll(&state, &instance)) => result,
            _ = instance.session.cancel.notified() => return,
        };
        let error = match result {
            Ok(Ok(())) => return,
            Ok(Err(error)) => error,
            Err(_) => "Model loading did not finish within 15 minutes.".into(),
        };
        let _guard = state.lifecycle.lock().await;
        if is_registered(&state, &instance).await {
            let error = error.replace(&instance.session.key, "[redacted]");
            state.log(format!("{}{error}", instance.prefix())).await;
            let detail = {
                let logs = state.logs.lock().await;
                with_runtime_detail(error, &logs, &instance.label)
            };
            state.record_load_error(&instance, detail).await;
            state.unload_locked(&instance.id).await;
        }
    });
    Ok(id)
}

/// The slot and log label a new instance of `path` gets beside `loaded`:
/// `preferred` when it is given and free, else the lowest free slot.
fn slot_and_label(loaded: &[Arc<Instance>], path: &str, preferred: Option<u32>) -> (u32, String) {
    let slots: Vec<u32> = loaded.iter().filter(|i| i.path() == path).map(|i| i.slot).collect();
    let slot = pick_slot(&slots, preferred);
    let labels: Vec<String> = loaded.iter().map(|i| i.label.clone()).collect();
    (slot, instance_label(&file_stem(path), slot, &labels))
}

/// The first private port at or above the configured one that no loaded
/// model or the local API holds and that is free right now. Call with the
/// lifecycle lock held, so two loads cannot pick the same port.
fn free_runtime_port(state: &AppState, settings: &Settings, loaded: &[Arc<Instance>]) -> Result<u16, String> {
    let api = state.listening_port();
    (0..64u16)
        .filter_map(|offset| settings.runtime_port.checked_add(offset))
        .filter(|port| *port != api && !loaded.iter().any(|i| i.session.port == *port))
        .find(|port| std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, *port)).is_ok())
        .ok_or_else(|| format!("No free private runtime port at or above {}. Choose another runtime port in Runtime.", settings.runtime_port))
}

const LOAD_MARKER: &str = "Loading model; checking the runtime and waiting for readiness…";

// llama-server explains most failures (bad file, out of memory, unsupported
// architecture) in its own output, so attach its last error lines from this
// load to the summary instead of only pointing at the log.
//
// Every model's lines carry its label, so a load running beside another only
// picks up its own errors.
fn with_runtime_detail(summary: String, logs: &VecDeque<String>, label: &str) -> String {
    let prefix = format!("[{label}] ");
    let marker = format!("{prefix}{LOAD_MARKER}");
    let start = logs
        .iter()
        .rposition(|line| *line == marker)
        .map_or(0, |i| i + 1);
    let mut lines: Vec<&str> = logs
        .iter()
        .skip(start)
        .filter_map(|line| line.strip_prefix(&prefix))
        .map(|line| line.trim())
        .filter(|line| *line != summary.as_str())
        .filter(|line| {
            let lower = line.to_lowercase();
            [
                "error",
                "failed",
                "unable",
                "cannot",
                "invalid",
                "out of memory",
                "unknown model",
                "not supported",
                "abort",
            ]
            .iter()
            .any(|word| lower.contains(word))
        })
        .collect();
    lines.dedup();
    let detail = lines[lines.len().saturating_sub(3)..].join("\n");
    if detail.is_empty() {
        summary
    } else {
        format!("{summary}\n{detail}")
    }
}

// Large models on slow disks or first-run CUDA kernel compilation can take
// minutes, so the hard limit is generous and silence is what fails a load.
const LOAD_LIMIT: Duration = Duration::from_secs(15 * 60);
const LOAD_SILENCE: Duration = Duration::from_secs(5 * 60);

/// Whether this load is still wanted: it has not been ejected or replaced.
async fn is_registered(state: &AppState, instance: &Arc<Instance>) -> bool {
    state.instances.lock().await.iter().any(|i| Arc::ptr_eq(i, instance))
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
    if settings.mtp {
        args.extend(["--spec-type".into(), "draft-mtp".into()]);
        // A separate head is loaded as the draft model; built-in layers run on
        // the main model's weights and need nothing more.
        if !settings.mtp_path.is_empty() {
            args.extend([
                "--spec-draft-model".into(),
                settings.mtp_path.clone(),
                "--spec-draft-ngl".into(),
                settings.gpu_layers.to_string(),
            ]);
        }
    }
    args
}

async fn prepare_and_poll(state: &AppState, instance: &Arc<Instance>) -> Result<(), String> {
    let settings = &instance.settings;
    let session = &instance.session;
    let prefix = instance.prefix();
    crate::library::verify_load(state, settings, &prefix).await?;
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
        if !is_registered(state, instance).await {
            return Ok(());
        }
        let mut live_settings = settings.clone();
        live_settings.allow_force_fit = state.settings.lock().await.allow_force_fit;
        if let Some(reason) = load_memory_risk(&live_settings, 0, 0).await {
            return Err(format!("OOM prevention: {reason}"));
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
            let session = session.clone();
            let prefix = prefix.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(pipe).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    if let Ok(mut last) = session.last_output.lock() {
                        *last = std::time::Instant::now();
                    }
                    state.log(format!("{prefix}{}", line.replace(&session.key, "[redacted]"))).await;
                }
            });
        }
        *instance.child.lock().await = Some(child);
        if let Ok(mut last) = session.last_output.lock() {
            *last = std::time::Instant::now();
        }
        if let Ok(mut model) = instance.model.lock() {
            model.device = device.clone();
        }
        state
            .log(format!("{prefix}Loading {} on {device} (private port {})", settings.model_path, session.port))
            .await;
    }
    loop {
        {
            let _guard = state.lifecycle.lock().await;
            if !is_registered(state, instance).await {
                return Ok(());
            }
            let mut child = instance.child.lock().await;
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
                let native = models["data"].as_array().is_some_and(|models| {
                    models.iter().any(|model| {
                        model["id"] == session.alias
                            && model["architecture"]["output_modalities"]
                                .as_array()
                                .is_some_and(|a| a.iter().any(|x| x == "decisions"))
                    })
                });
                let layout = match crate::decision::resolve(&settings.decision_mode, &settings.model_path) {
                    // Winnow (forced, or detected from a file with no decision key) wins;
                    // otherwise the runtime's native support does.
                    Some(crate::decision::Layout::Winnow) => crate::decision::Layout::Winnow,
                    _ if native => crate::decision::Layout::Native,
                    Some(crate::decision::Layout::Native) => {
                        return Err("Runtime did not advertise built-in decision support. Use a decision GGUF and a compatible llama.cpp build, or set Decision mode to Auto in Local API.".into());
                    }
                    None => {
                        return Err("This model has no built-in decision support, and Noullama has no prompt layout for it. Winnow GGUFs are read through the Prompt layout (Winnow); if this is a Winnow model under another name, set Decision mode to Prompt layout in Local API.".into());
                    }
                };
                if layout == crate::decision::Layout::Winnow && session.readout.get().is_none() {
                    let server = crate::decision::Server { client: &state.local_client, port: session.port, key: &session.key };
                    let profile = crate::decision::profile(&settings.model_path);
                    let thought = match profile.thought {
                        Some(thought) => thought,
                        None => crate::decision::template_has_thought(
                            server.props().await?["chat_template"].as_str().unwrap_or_default(),
                        ),
                    };
                    let labels = server.labels().await?;
                    state
                        .log(format!(
                            "{prefix}Reading decisions through the {} layout ({}, temperature {}, {} labels).",
                            crate::decision::LAYOUT, profile.model, profile.temperature, labels.len()
                        ))
                        .await;
                    let _ = session.readout.set(crate::decision::Readout {
                        profile,
                        thought,
                        labels,
                        context: settings.context_size as usize,
                    });
                }
                let _guard = state.lifecycle.lock().await;
                if !is_registered(state, instance).await {
                    return Ok(());
                }
                let mut child = instance.child.lock().await;
                if !matches!(child.as_mut().map(|c| c.try_wait()), Some(Ok(None))) {
                    return Err("Runtime exited before becoming ready.".into());
                }
                if let Ok(mut model) = instance.model.lock() {
                    model.layout = Some(layout.name().into());
                }
                instance.ready.store(true, Ordering::SeqCst);
                state.instance_freed.notify_one();
                state
                    .log(format!("{prefix}Decision model is ready. Local API is accepting requests."))
                    .await;
                return Ok(());
            }
        }
        if session
            .last_output
            .lock()
            .is_ok_and(|last| last.elapsed() > LOAD_SILENCE)
        {
            return Err(format!(
                "Model loading stalled: the runtime wrote nothing for {} seconds.",
                LOAD_SILENCE.as_secs()
            ));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// Unloads every model.
#[tauri::command]
pub async fn stop_runtime(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let any = !state.instances.lock().await.is_empty();
    state.shutdown().await;
    if any {
        state.log("Runtime stopped; every model unloaded.".into()).await;
    }
    Ok(())
}

/// Ejects one loaded model; the others keep running.
#[tauri::command]
pub async fn unload_model(id: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    unload(state.inner(), &id).await
}

pub async fn unload(state: &AppState, id: &str) -> Result<(), String> {
    crate::library::refresh(state).await;
    let _guard = state.lifecycle.lock().await;
    let instance = state.unload_locked(id).await.ok_or("That model is no longer loaded.")?;
    state.log(format!("{}Unloaded.", instance.prefix())).await;
    Ok(())
}

#[tauri::command]
pub async fn dismiss_load_error(id: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    state.load_errors.lock().await.retain(|e| e.id != id);
    Ok(())
}

/// Changes a loaded instance's suffix (slot 2 and up). It lives only until
/// the instance is unloaded and is never saved.
#[tauri::command]
pub async fn set_instance_suffix(id: String, suffix: String, state: tauri::State<'_, AppState>) -> Result<String, String> {
    change_suffix(state.inner(), &id, &suffix).await
}

pub async fn change_suffix(state: &AppState, id: &str, suffix: &str) -> Result<String, String> {
    crate::library::refresh(state).await;
    // Held so a load or another suffix change cannot slip in between the check and the write.
    let _guard = state.lifecycle.lock().await;
    let instance = state.instance(id).await.ok_or("That model is no longer loaded.")?;
    let routes = state.routes().await;
    let (name, aliases) = state.api_names(instance.path()).await;
    let instances = state.instances.lock().await.clone();
    let taken: Vec<(String, String)> = routes
        .iter()
        // Every other instance's identifiers, and this one's own instance aliases.
        .filter(|r| r.instance.as_deref().is_some_and(|other| other != id || r.kind == "instance_alias"))
        .map(|r| {
            let owner = instances.iter().find(|i| Some(i.id.as_str()) == r.instance.as_deref());
            (r.name.clone(), owner.map_or_else(|| "another loaded model".to_owned(), |o| format!("instance {} of {}", o.slot, o.stem())))
        })
        .collect();
    let library: Vec<(String, String)> = state
        .library
        .lock()
        .await
        .entries
        .iter()
        .flat_map(|e| std::iter::once(&e.name).chain(&e.aliases).map(move |n| (n.clone(), e.name.clone())))
        .collect();
    let suffix = check_suffix(instance.slot, suffix, &name, &aliases, &taken, &library)?;
    instance.set_suffix(suffix.clone());
    state.log(format!("{}API suffix set to {suffix}.", instance.prefix())).await;
    Ok(suffix)
}

/// Sets one loaded instance's own aliases: exact names (no suffix) that
/// reach only it. They live until the instance is unloaded and are never saved.
#[tauri::command]
pub async fn set_instance_aliases(id: String, aliases: Vec<String>, state: tauri::State<'_, AppState>) -> Result<Vec<String>, String> {
    change_instance_aliases(state.inner(), &id, aliases).await
}

pub async fn change_instance_aliases(state: &AppState, id: &str, aliases: Vec<String>) -> Result<Vec<String>, String> {
    crate::library::refresh(state).await;
    // Held so a load, a suffix change or another alias change cannot slip in between the check and the write.
    let _guard = state.lifecycle.lock().await;
    let instance = state.instance(id).await.ok_or("That model is no longer loaded.")?;
    let aliases = normalize_aliases(&aliases)?;
    let library: Vec<(String, String)> = state.library.lock().await.entries.iter()
        .flat_map(|e| std::iter::once(&e.name).chain(&e.aliases).map(move |n| (n.clone(), e.name.clone())))
        .collect();
    let instances = state.instances.lock().await.clone();
    // Every identifier in the route table but this instance's own aliases.
    let taken: Vec<(String, String)> = state.routes().await.into_iter()
        .filter(|r| !(r.kind == "instance_alias" && r.instance.as_deref() == Some(id)))
        .map(|r| {
            let owner = r.instance.as_deref().and_then(|o| instances.iter().find(|i| i.id == o));
            (r.name, owner.map_or_else(|| file_stem(&r.path), |o| format!("instance {} of {}", o.slot, o.stem())))
        })
        .collect();
    let aliases = check_instance_aliases(aliases, &taken, &library)?;
    instance.set_aliases(aliases.clone());
    let text = if aliases.is_empty() { "none".to_owned() } else { aliases.join(", ") };
    state.log(format!("{}Instance aliases set to {text}.", instance.prefix())).await;
    Ok(aliases)
}

/// Checks one instance's aliases (already trimmed, checked and de-duplicated
/// by `normalize_aliases`) against every models.yml name and alias and every
/// other identifier a loaded instance answers to, ignoring case.
pub fn check_instance_aliases(aliases: Vec<String>, taken: &[(String, String)], library: &[(String, String)]) -> Result<Vec<String>, String> {
    for alias in &aliases {
        if let Some((_, entry)) = library.iter().find(|(other, _)| other.eq_ignore_ascii_case(alias)) {
            return Err(format!("\"{alias}\" is already a name or alias of {entry} in models.yml. Choose another alias."));
        }
        if let Some((_, owner)) = taken.iter().find(|(other, _)| other.eq_ignore_ascii_case(alias)) {
            return Err(format!("\"{alias}\" already reaches {owner}. Choose another alias."));
        }
    }
    Ok(aliases)
}

/// Every instance alias of a loaded instance, with who holds it, for checks
/// that must not hand one of them to a models.yml entry.
pub async fn instance_alias_owners(state: &AppState) -> Vec<(String, String)> {
    state.instances.lock().await.iter()
        .flat_map(|i| i.aliases().into_iter().map(move |a| (a, format!("instance {} of {}", i.slot, i.stem()))))
        .collect()
}

/// The Settings tab's "When every instance is busy" for `any-available`.
#[tauri::command]
pub async fn set_any_available_when_busy(mode: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    set_when_busy(state.inner(), &mode).await
}

pub async fn set_when_busy(state: &AppState, mode: &str) -> Result<(), String> {
    if !["reject", "wait"].contains(&mode) {
        return Err("Unknown choice for when every instance is busy.".into());
    }
    let mut settings = state.settings.lock().await.clone();
    settings.any_available_when_busy = mode.to_owned();
    persist_settings(state, settings).await
}

/// The Settings tab's "Fall through to another instance".
#[tauri::command]
pub async fn set_fall_through(on: bool, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut settings = state.settings.lock().await.clone();
    settings.fall_through_to_other_instances = on;
    persist_settings(state.inner(), settings).await
}

/// Graphics memory in use, overall and per loaded model.
#[tauri::command]
pub async fn model_memory(state: tauri::State<'_, AppState>) -> Result<Value, String> {
    let instances = state.instances.lock().await.clone();
    let mut pids = Vec::new();
    for instance in &instances {
        if let Some(pid) = instance.child.lock().await.as_ref().and_then(Child::id) {
            pids.push((instance.id.clone(), pid));
        }
    }
    let ids: Vec<u32> = pids.iter().map(|(_, pid)| *pid).collect();
    let usage = tokio::task::spawn_blocking(move || crate::hardware::memory_use(&ids))
        .await
        .map_err(|e| e.to_string())?;
    let models: serde_json::Map<String, Value> = pids
        .iter()
        .zip(&usage.processes)
        .map(|((id, _), bytes)| (id.clone(), json!(bytes)))
        .collect();
    let mut value = json!(usage);
    value["models"] = Value::Object(models);
    Ok(value)
}

/// Runs a Playground decision on the loaded model `model` (an id), or on the
/// oldest ready model when none is given.
#[tauri::command]
pub async fn decide(payload: Value, model: Option<String>, state: tauri::State<'_, AppState>) -> Result<Value, String> {
    let instance = match model {
        Some(id) => state.instance(&id).await.ok_or("That model was ejected. Choose a loaded model.")?,
        None => state
            .instances
            .lock()
            .await
            .iter()
            .find(|i| i.is_ready())
            .cloned()
            .ok_or("Load a decision model before running inference.")?,
    };
    state.infer_on(&instance, payload).await
}

/// Registers a loaded model with no runtime behind it, for routing tests.
#[tauri::command]
pub async fn set_memory_preference(key: String, value: bool, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let settings = state.settings.lock().await.clone();
    persist_memory_settings(state.inner(), settings, Some((&key, value))).await
}

/// `freed_vram` and `freed_ram`: memory a loaded instance being replaced
/// holds now, counted as available because it is unloaded before the load.
async fn load_memory_risk(settings: &Settings, freed_vram: u64, freed_ram: u64) -> Option<String> {
    if settings.allow_force_fit { return None; }
    let required = crate::hardware::estimate_bytes(&settings.model_path, &instance_extras(settings), settings.context_size)?;
    let gpu_layers = settings.gpu_layers;
    tokio::task::spawn_blocking(move || {
        if gpu_layers != 0 {
            let gpu = crate::hardware::memory_use(&[]);
            if let (Some(total), Some(used)) = (gpu.total_bytes, gpu.used_bytes) {
                let available = total.saturating_sub(used).saturating_add(freed_vram).min(total);
                if (gpu_layers == -1 || gpu_layers >= 99) && crate::oom::clearly_exceeds(required, available) {
                    return Some(format!("Estimated GPU allocation {:.1} GiB clearly exceeds {:.1} GiB available VRAM.", required as f64 / 1073741824., available as f64 / 1073741824.));
                }
            }
        }
        // Partial offload cannot be divided accurately from file size alone.
        // Do not pretend the entire file must be resident in system RAM.
        if gpu_layers == 0 {
            if let Some(m) = crate::oom::memory() {
                let available = m.available.saturating_add(freed_ram);
                if crate::oom::clearly_exceeds(required, available) {
                    return Some(format!("Estimated allocation {:.1} GiB clearly exceeds {:.1} GiB available RAM.", required as f64 / 1073741824., available as f64 / 1073741824.));
                }
            }
        }
        None
    }).await.ok().flatten()
}

#[tauri::command]
pub async fn crowded_load_warning(replace: Option<String>, state: tauri::State<'_, AppState>) -> Result<Option<Value>, String> {
    let settings = state.settings.lock().await.clone();
    if settings.hide_crowded_load_warning || settings.allow_force_fit { return Ok(None); }
    let loaded = state.instances.lock().await.clone();
    // A reload unloads `replace` first, so it is neither another model nor
    // memory in use.
    let replaced = replace.as_deref().and_then(|id| loaded.iter().find(|i| i.id == id));
    let others: Vec<_> = loaded.iter().filter(|i| Some(i.id.as_str()) != replace.as_deref()).collect();
    if others.is_empty() { return Ok(None); }
    let (mut freed_vram, mut freed_ram) = (0, 0);
    if let Some(instance) = replaced {
        let pid = instance.child.lock().await.as_ref().and_then(Child::id);
        let estimate = crate::hardware::estimate_bytes(instance.path(), &instance_extras(&instance.settings), instance.settings.context_size).unwrap_or(0);
        if instance.settings.gpu_layers != 0 {
            // Per-process readings exist on NVIDIA only; otherwise use the estimate.
            let measured = match pid {
                Some(pid) => tokio::task::spawn_blocking(move || crate::hardware::memory_use(&[pid])).await.ok().and_then(|u| u.processes.first().copied().flatten()),
                None => None,
            };
            freed_vram = measured.unwrap_or(estimate);
        } else {
            freed_ram = pid.and_then(crate::oom::resident_bytes).unwrap_or(estimate);
        }
    }
    let mut names = Vec::new();
    for instance in &others {
        let (name, _) = state.api_names(instance.path()).await;
        names.push(json!({"id": instance.id, "name": format!("{name}{}", instance.suffix())}));
    }
    Ok(load_memory_risk(&settings, freed_vram, freed_ram).await.map(|reason| json!({ "reason": reason, "models": names })))
}

/// Supervises loaded models once a second: notices runtimes that exited and
/// runs OOM prevention. With nothing loaded there is nothing to reap or
/// eject, so it sleeps until a load registers an instance.
pub fn start_memory_guard(state: AppState) {
    tauri::async_runtime::spawn(async move {
        let mut pressure = crate::oom::Pressure::default();
        loop {
            if state.instances.lock().await.is_empty() {
                pressure = Default::default();
                // A load that registers between the check and this wait
                // leaves a stored permit, so it is not missed.
                state.instances_added.notified().await;
                continue;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
            if state.app.exiting.load(std::sync::atomic::Ordering::SeqCst) { break; }
            // Never wait here: a load or unload holding the lock must not
            // delay OOM sampling. A busy tick leaves the crash to the next.
            if let Ok(_guard) = state.lifecycle.try_lock() {
                let _ = reap_exited_locked(&state).await;
            }
            if state.settings.lock().await.allow_force_fit {
                pressure = Default::default();
                continue;
            }
            let memory = tokio::task::spawn_blocking(crate::oom::memory).await.ok().flatten();
            if !pressure.sample(memory) { continue; }
            let _guard = state.lifecycle.lock().await;
            // Recheck after waiting for the lock. The setting is global and live.
            if state.settings.lock().await.allow_force_fit || !pressure.sample(crate::oom::memory()) { continue; }
            let instances = state.instances.lock().await.clone();
            let mut candidate = None;
            let mut largest = 128 * 1024 * 1024;
            for instance in instances.iter().rev() {
                // Read only an owned, unreaped child while its handle is locked.
                let child = instance.child.lock().await;
                if let Some(rss) = child.as_ref().and_then(Child::id).and_then(crate::oom::resident_bytes) {
                    if rss > largest { largest = rss; candidate = Some(instance.clone()); }
                }
            }
            if let Some(instance) = candidate {
                let message = "OOM prevention ejected this model after sustained critically low available RAM and memory stalls. Allow force fit in Settings disables this protection.".to_owned();
                state.log(format!("{}{}", instance.prefix(), message)).await;
                state.record_load_error(&instance, message).await;
                state.unload_locked(&instance.id).await;
            }
            // One victim, then a cooldown and fresh evidence before any further action.
            pressure = Default::default();
            drop(_guard);
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    });
}

#[cfg(test)]
pub async fn insert_test_instance(state: &AppState, path: &str, ready: bool) -> Arc<Instance> {
    let port = 1 + state.instances.lock().await.len() as u16;
    insert_test_instance_on(state, path, ready, port).await
}

/// Like `insert_test_instance`, with requests going to a fake runtime on `port`.
#[cfg(test)]
pub async fn insert_test_instance_on(state: &AppState, path: &str, ready: bool, port: u16) -> Arc<Instance> {
    let settings = Settings { model_path: path.into(), ..state.settings.lock().await.clone() };
    let loaded = state.instances.lock().await.clone();
    let (slot, label) = slot_and_label(&loaded, path, None);
    let instance = Arc::new(Instance {
        id: uuid::Uuid::new_v4().simple().to_string()[..12].to_owned(),
        label,
        slot,
        suffix: std::sync::Mutex::new(default_suffix(slot)),
        model: std::sync::Mutex::new(LoadedModel { path: path.into(), device: "CPU".into(), runtime_path: String::new(), projector: false, mtp: false, layout: Some("native".into()) }),
        session: RuntimeSession::new(&state.data_dir, port).unwrap(),
        keep_context: AtomicBool::new(settings.keep_context),
        settings,
        child: Default::default(),
        ready: AtomicBool::new(ready),
        context: Default::default(),
        aliases: Default::default(),
        in_flight: Default::default(),
        last_started: Default::default(),
    });
    state.instances.lock().await.push(instance.clone());
    instance
}

/// Counts a request as running on a test instance until the claim is dropped.
#[cfg(test)]
pub fn hold_test_claim(state: &AppState, instance: &Arc<Instance>) -> Claim {
    Claim::take(state, instance)
}

/// Marks a test instance ready or loading.
#[cfg(test)]
pub fn set_test_ready(instance: &Instance, ready: bool) {
    instance.ready.store(ready, Ordering::SeqCst);
}

/// Waits for a load to finish; panics with the log if it fails.
#[cfg(test)]
pub async fn wait_until_ready(state: &AppState, id: &str, seconds: u64) -> Arc<Instance> {
    for _ in 0..seconds * 10 {
        match state.instance(id).await {
            Some(instance) if instance.is_ready() => return instance,
            Some(_) => tokio::time::sleep(Duration::from_millis(100)).await,
            None => break,
        }
    }
    let logs = state.logs.lock().await.clone();
    let errors: Vec<String> = state.load_errors.lock().await.iter().map(|e| e.message.clone()).collect();
    state.shutdown().await;
    panic!("Model {id} did not become ready: {errors:#?}\n{logs:#?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_instance_takes_the_lowest_free_slot() {
        assert_eq!(free_slot(&[]), 1);
        assert_eq!(free_slot(&[1]), 2);
        assert_eq!(free_slot(&[1, 2, 3]), 4);
        assert_eq!(free_slot(&[2, 3]), 1);
        assert_eq!(free_slot(&[1, 3]), 2);
        // A reload asks for its old slot and keeps it even with a lower one free.
        assert_eq!(pick_slot(&[1], Some(3)), 3);
        assert_eq!(pick_slot(&[1, 3], Some(3)), 2);
        assert_eq!(pick_slot(&[1], Some(0)), 2);
        assert_eq!(pick_slot(&[1], None), 2);
        assert_eq!(default_suffix(1), "");
        assert_eq!(default_suffix(2), "-i-2");
        assert_eq!(default_suffix(12), "-i-12");
        // Labels follow the slot and stay unique across files sharing a stem.
        assert_eq!(instance_label("qwen", 1, &[]), "qwen");
        assert_eq!(instance_label("qwen", 2, &["qwen".into()]), "qwen#2");
        assert_eq!(instance_label("qwen", 1, &["qwen".into()]), "qwen#2");
        assert_eq!(instance_label("qwen", 2, &["qwen#2".into()]), "qwen#3");
    }

    #[test]
    fn identifiers_add_the_suffix_to_every_name_and_alias() {
        let aliases = vec!["triage".to_owned(), "sorter".to_owned()];
        let ids: Vec<(String, &str, String)> = identifiers("qwen", &aliases, "-i-2");
        assert_eq!(ids, [
            ("qwen-i-2".to_owned(), "name", "qwen".to_owned()),
            ("triage-i-2".to_owned(), "alias", "triage".to_owned()),
            ("sorter-i-2".to_owned(), "alias", "sorter".to_owned()),
        ]);
        let plain: Vec<String> = identifiers("qwen", &aliases, "").into_iter().map(|(id, _, _)| id).collect();
        assert_eq!(plain, ["qwen", "triage", "sorter"]);
    }

    #[test]
    fn suffixes_are_checked_like_aliases_and_must_not_collide() {
        let aliases = vec!["triage".to_owned()];
        let library = vec![("qwen".to_owned(), "qwen".to_owned()), ("triage".to_owned(), "qwen".to_owned()), ("qwen-b".to_owned(), "other".to_owned())];
        let taken = vec![("qwen-i-3".to_owned(), "instance 3 of qwen".to_owned()), ("triage-i-3".to_owned(), "instance 3 of qwen".to_owned())];
        let check = |slot, suffix: &str| check_suffix(slot, suffix, "qwen", &aliases, &taken, &library);
        assert_eq!(check(2, " -fast ").unwrap(), "-fast");
        assert_eq!(check(1, "-x").unwrap_err(), "Instance 1 answers to the plain names; its suffix cannot be changed.");
        assert_eq!(check(2, "  ").unwrap_err(), "Enter a suffix for instance 2, such as -i-2.");
        assert_eq!(check(2, " x y").unwrap_err(), "Model name \"qwenx y\" can use letters, digits and . _ : / - only, up to 64 characters.");
        assert_eq!(check(2, &"x".repeat(60)).unwrap_err(), format!("Model name \"triage{}\" can use letters, digits and . _ : / - only, up to 64 characters.", "x".repeat(60)));
        assert_eq!(check(2, "-B").unwrap_err(), "\"qwen-B\" is already a name or alias of other in models.yml. Choose another suffix.");
        assert_eq!(check(2, "-i-3").unwrap_err(), "\"qwen-i-3\" already reaches instance 3 of qwen. Choose another suffix.");
        // The slot's own default is always accepted, even when it would not pass.
        let long = "q".repeat(62);
        assert_eq!(check_suffix(2, "-i-2", &long, &[], &[], &[]).unwrap(), "-i-2");
        assert!(check_suffix(2, "-xyz", &long, &[], &[], &[]).is_err());
    }

    fn table(instances: &[(&str, &str, u32, &str, bool)], entries: &[&str], fall_through: bool) -> Vec<Route> {
        let list: Vec<RouteInstance> = instances.iter().map(|(id, path, slot, suffix, ready)| RouteInstance { id, path, slot: *slot, suffix: suffix.to_string(), ready: *ready, aliases: vec![] }).collect();
        let entries: Vec<String> = entries.iter().map(|e| e.to_string()).collect();
        route_table(&list, &entries, |path| match path {
            "/m/qwen.gguf" => ("qwen".into(), vec!["triage".into()]),
            "/m/other.gguf" => ("other".into(), vec![]),
            _ => (file_stem(path), vec![]),
        }, fall_through)
    }

    #[test]
    fn the_route_table_lists_suffixed_instances_in_load_order() {
        let routes = table(&[("a", "/m/qwen.gguf", 1, "", true), ("b", "/m/other.gguf", 1, "", true), ("c", "/m/qwen.gguf", 2, "-i-2", true)], &["/m/qwen.gguf", "/m/other.gguf", "/m/later.gguf"], false);
        let names: Vec<&str> = routes.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["qwen", "triage", "other", "qwen-i-2", "triage-i-2", "later"]);
        assert!(routes.iter().all(|r| !r.conflict));
        assert_eq!(listed_names(&routes), ["qwen", "triage", "other", "qwen-i-2", "triage-i-2"]);
        // Without slot 1, the bare names reach nothing unless fall-through is on.
        let routes = table(&[("c", "/m/qwen.gguf", 2, "-i-2", true), ("d", "/m/qwen.gguf", 3, "-i-3", true)], &["/m/qwen.gguf"], false);
        let bare: Vec<(&str, Option<&str>)> = routes.iter().filter(|r| r.fallthrough).map(|r| (r.name.as_str(), r.instance.as_deref())).collect();
        assert_eq!(bare, [("qwen", None), ("triage", None)]);
        assert_eq!(listed_names(&routes), ["qwen-i-2", "triage-i-2", "qwen-i-3", "triage-i-3"]);
        let routes = table(&[("d", "/m/qwen.gguf", 3, "-i-3", true), ("c", "/m/qwen.gguf", 2, "-i-2", false)], &["/m/qwen.gguf"], true);
        let bare: Vec<(&str, Option<&str>, bool)> = routes.iter().filter(|r| r.fallthrough).map(|r| (r.name.as_str(), r.instance.as_deref(), r.ready)).collect();
        // The lowest slot is 2, which is still loading, so the bare name is not listed.
        assert_eq!(bare, [("qwen", Some("c"), false), ("triage", Some("c"), false)]);
        assert_eq!(listed_names(&routes), ["qwen-i-3", "triage-i-3"]);
        // A suffix that matches another model's name conflicts on both sides.
        let routes = table(&[("a", "/m/qwen.gguf", 2, "-i-2", true)], &["/m/qwen.gguf", "/m/qwen-i-2.gguf"], false);
        assert!(routes.iter().filter(|r| r.name == "qwen-i-2").all(|r| r.conflict));
        assert_eq!(routes.iter().filter(|r| r.name == "qwen-i-2").count(), 2);
    }
    #[test]
    fn images_must_be_inline_data_urls() {
        let question = json!({"q": {"type": "noul", "instructions": "Is there a table?"}});
        let ok = json!({"state": null, "questions": question, "images": ["data:image/png;base64,iVBORw0KGgo="]});
        assert!(validate_request(&ok).is_ok());
        assert!(has_images(&ok));
        for images in [
            json!(["https://example.com/a.png"]),
            json!("data:image/png;base64,AA"),
            json!([1]),
        ] {
            let request = json!({"state": "x", "questions": question, "images": images});
            assert!(validate_request(&request)
                .unwrap_err()
                .contains("data URLs"));
        }
        assert!(!has_images(&json!({"state": "x", "images": []})));
    }

    #[test]
    fn load_errors_carry_the_runtime_explanation_from_this_load() {
        let marker = format!("[foo] {LOAD_MARKER}");
        let logs: VecDeque<String> = [
            "[foo] error: stale failure from an earlier load",
            &marker,
            "[foo] llama_model_load: loading model",
            "[other] error: a failure from the model loading beside it",
            "[foo] llama_model_load: error loading model: unknown model architecture: 'foo'",
            "[foo] main: exiting due to model loading error",
        ]
        .map(String::from)
        .into();
        let message = with_runtime_detail("Runtime exited during startup.".into(), &logs, "foo");
        assert!(message.starts_with("Runtime exited during startup.\n"));
        assert!(message.contains("unknown model architecture"));
        assert!(!message.contains("stale failure"));
        assert!(!message.contains("beside it"));
        assert_eq!(
            with_runtime_detail("Plain.".into(), &VecDeque::new(), "foo"),
            "Plain."
        );
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn supervisor_wakes_on_load_and_reaps_a_crash_without_any_snapshot() {
        let base = std::env::temp_dir().join(format!("noullama-reap-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base).unwrap();
        // Parked with nothing loaded; the registration below must wake it.
        start_memory_guard(state.clone());
        tokio::time::sleep(Duration::from_millis(50)).await;
        let instance = insert_test_instance(&state, "/m/crashed.gguf", true).await;
        *instance.child.lock().await = Some(Command::new("true").spawn().unwrap());
        state.instances_added.notify_one();
        // The UI is hidden: nothing calls state_snapshot.
        let reaped = tokio::time::timeout(Duration::from_secs(5), async {
            while !state.instances.lock().await.is_empty() {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await;
        assert!(reaped.is_ok(), "the supervisor did not unload the exited runtime");
        let errors = state.load_errors.lock().await.clone();
        assert!(errors.iter().any(|e| e.message.contains("crashed stopped unexpectedly")), "{:?}", errors.iter().map(|e| &e.message).collect::<Vec<_>>());
        state.app.exiting.store(true, std::sync::atomic::Ordering::SeqCst);
    }
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
        let instance = insert_test_instance(&state, "/m/a.gguf", true).await;
        let value = state_snapshot(&state).await.unwrap();
        // Every field the frontend's Instance and Route types read.
        for key in ["id", "name", "label", "path", "device", "runtime_path", "projector", "mmproj_path", "mtp", "mtp_path", "layout", "ready", "context_size", "gpu_layers", "required_bytes", "context", "settings"] {
            assert!(value["instances"][0].get(key).is_some(), "instances[0] lacks {key}");
        }
        assert!(value["instances"][0]["settings"]["gpu_layers"].is_number());
        for key in ["name", "kind", "path", "instance", "ready", "conflict"] {
            assert!(value["routes"][0].get(key).is_some(), "routes[0] lacks {key}");
        }
        for key in ["instances", "default_model", "routes", "load_errors"] {
            assert!(value.get(key).is_some(), "snapshot lacks {key}");
        }
        let snapshot = value.to_string();
        assert!(!snapshot.contains(&session.key));
        assert!(!snapshot.contains(&instance.session.key));
        let args = runtime_args(&Settings::default(), &session);
        assert!(!args.iter().any(|arg| arg.contains(&session.key)));
        let instance_key = instance.session.key_path.clone();
        state.shutdown().await;
        drop(session);
        drop(instance);
        assert!(!key_path.exists());
        assert!(!instance_key.exists());
        std::fs::remove_dir_all(base).unwrap();
    }
    #[test]
    fn mtp_adds_speculative_flags_only_when_on() {
        let base = std::env::temp_dir().join(format!("noullama-mtp-args-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let session = RuntimeSession::new(&base, 18441).unwrap();
        let flag = |args: &[String], name: &str| {
            args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
        };
        let off = runtime_args(&Settings { mtp_path: "/m/mtp-head.gguf".into(), ..Settings::default() }, &session);
        assert!(flag(&off, "--spec-type").is_none());
        assert!(flag(&off, "--spec-draft-model").is_none());

        let built_in = runtime_args(&Settings { mtp: true, ..Settings::default() }, &session);
        assert_eq!(flag(&built_in, "--spec-type").as_deref(), Some("draft-mtp"));
        assert!(flag(&built_in, "--spec-draft-model").is_none());

        let head = Settings { mtp: true, mtp_path: "/m/mtp-head.gguf".into(), gpu_layers: 0, ..Settings::default() };
        let args = runtime_args(&head, &session);
        assert_eq!(flag(&args, "--spec-type").as_deref(), Some("draft-mtp"));
        assert_eq!(flag(&args, "--spec-draft-model").as_deref(), Some("/m/mtp-head.gguf"));
        assert_eq!(flag(&args, "--spec-draft-ngl").as_deref(), Some("0"));
        drop(session);
        std::fs::remove_dir_all(base).ok();
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
        let old = load_model(&state).await.unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        let snapshot = tokio::time::timeout(Duration::from_millis(500), state_snapshot(&state))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(snapshot["running"], true);
        let old = state.instance(&old).await.unwrap();
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
        assert!(!is_registered(&state, &old).await);
        drop(old);
        let replacement = load_model(&state).await.unwrap();
        let replacement = state.instance(&replacement).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(is_registered(&state, &replacement).await);
        assert!(replacement.child.lock().await.is_none());
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
        let id = load_model(&state).await.unwrap();
        let instance = state.instance(&id).await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while instance.child.lock().await.is_none() {
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
            while !state.instances.lock().await.is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(!instance.is_ready());
        assert!(instance.child.lock().await.is_none());
        assert!(state.load_errors.lock().await[0].message.contains("identity did not match"));
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
    /// Loads a real Winnow GGUF through the normal load path and answers a
    /// request in the Winnow layout. Run with
    /// `NOULLAMA_RUNTIME=…/llama-server NOULLAMA_WINNOW=…/Winnow-E4B-Q8_0.gguf cargo test -- --ignored winnow`.
    #[tokio::test]
    #[ignore]
    async fn winnow_loads_and_decides_on_a_real_runtime() {
        let (Ok(runtime), Ok(model)) = (std::env::var("NOULLAMA_RUNTIME"), std::env::var("NOULLAMA_WINNOW")) else {
            panic!("set NOULLAMA_RUNTIME and NOULLAMA_WINNOW");
        };
        let base = std::env::temp_dir().join(format!("noullama-winnow-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let port = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap().local_addr().unwrap().port();
        {
            let mut settings = state.settings.lock().await;
            settings.runtime_path = runtime;
            settings.model_path = model;
            settings.runtime_port = port;
            settings.mtp = std::env::var("NOULLAMA_MTP_HEAD").is_ok();
            settings.mtp_path = std::env::var("NOULLAMA_MTP_HEAD").unwrap_or_default();
        }
        let id = load_model(&state).await.unwrap();
        let instance = wait_until_ready(&state, &id, 300).await;
        assert_eq!(instance.model().layout.as_deref(), Some("winnow-v1"));
        let request = json!({"state": {"ticket": "My card was charged twice for order #4411 and I need the duplicate refunded today."},
            "questions": {
                "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
                "topic": {"type": "choice", "instructions": "What is the ticket about?", "criteria": {"billing": "Charges, refunds, invoices", "shipping": null, "login": null}},
                "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["Can wait", "Today", "Immediately"]}}});
        let started = std::time::Instant::now();
        let result = state.infer_on(&instance, request.clone()).await.unwrap();
        println!("Winnow readout ({:?}): {result}", started.elapsed());
        assert!(result["answers"]["refund"]["noul"].as_f64().unwrap() > 0.9);
        assert_eq!(result["answers"]["topic"]["choice"], "billing");
        let score = result["answers"]["urgency"]["score"].as_f64().unwrap();
        assert!((1.0..2.0).contains(&score), "{score}");
        // The same request gives the same numbers.
        assert_eq!(state.infer_on(&instance, request).await.unwrap()["answers"], result["answers"]);
        state.shutdown().await;
        std::fs::remove_dir_all(base).ok();
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
    async fn memory_preferences_are_independent_persisted_and_survive_stale_forms() {
        let old: Settings = serde_json::from_str("{}").unwrap();
        assert!(!old.allow_force_fit && !old.hide_crowded_load_warning);
        let base = std::env::temp_dir().join(format!("noullama-memory-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        persist_memory_settings(&state, old.clone(), Some(("hide_crowded_load_warning", true))).await.unwrap();
        assert!(!state.settings.lock().await.allow_force_fit);
        persist_memory_settings(&state, old.clone(), Some(("allow_force_fit", true))).await.unwrap();
        save_from_editor(&state, old).await.unwrap();
        let loaded = AppState::new(base.clone()).unwrap();
        let settings = loaded.settings.lock().await;
        assert!(settings.allow_force_fit && settings.hide_crowded_load_warning);
        drop(settings);
        assert!(load_memory_risk(&loaded.settings.lock().await.clone(), 0, 0).await.is_none());
        std::fs::remove_dir_all(base).unwrap();
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

    #[test]
    fn any_available_is_reserved_in_aliases_suffixes_and_instance_aliases() {
        let reserved = "\"any-available\" is reserved: a request that sends it goes to any free loaded instance.";
        // Ignoring case, like every other name collision.
        for alias in ["any-available", "Any-Available"] {
            assert_eq!(normalize_aliases(&[alias.into()]).unwrap_err(), format!("{reserved} Choose another alias."));
        }
        // A suffix that turns "any" into the reserved name.
        assert_eq!(check_suffix(2, "-available", "any", &[], &[], &[]).unwrap_err(), format!("{reserved} Choose another suffix."));
        assert_eq!(check_suffix(2, "-available", "qwen", &["any".into()], &[], &[]).unwrap_err(), format!("{reserved} Choose another suffix."));
        assert_eq!(check_suffix(2, "-available-2", "any", &[], &[], &[]).unwrap(), "-available-2");
        assert!(is_reserved("ANY-available") && !is_reserved("any-available-2"));
    }

    #[test]
    fn instance_aliases_must_not_collide_with_models_yml_or_loaded_identifiers() {
        let library = vec![("qwen".to_owned(), "qwen".to_owned()), ("triage".to_owned(), "qwen".to_owned())];
        let taken = vec![("qwen-i-2".to_owned(), "instance 2 of qwen".to_owned()), ("sorter".to_owned(), "instance 1 of other".to_owned())];
        let check = |aliases: &[&str]| check_instance_aliases(aliases.iter().map(|a| a.to_string()).collect(), &taken, &library);
        assert_eq!(check(&["billing", "refunds"]).unwrap(), ["billing", "refunds"]);
        assert_eq!(check(&["TRIAGE"]).unwrap_err(), "\"TRIAGE\" is already a name or alias of qwen in models.yml. Choose another alias.");
        assert_eq!(check(&["Qwen-I-2"]).unwrap_err(), "\"Qwen-I-2\" already reaches instance 2 of qwen. Choose another alias.");
        assert_eq!(check(&["billing", "sorter"]).unwrap_err(), "\"sorter\" already reaches instance 1 of other. Choose another alias.");
    }

    #[test]
    fn instance_aliases_follow_their_instance_in_the_route_table_unsuffixed() {
        let list = vec![
            RouteInstance { id: "a", path: "/m/qwen.gguf", slot: 1, suffix: String::new(), ready: true, aliases: vec![] },
            RouteInstance { id: "b", path: "/m/qwen.gguf", slot: 2, suffix: "-i-2".into(), ready: true, aliases: vec!["billing".into()] },
            RouteInstance { id: "c", path: "/m/other.gguf", slot: 1, suffix: String::new(), ready: true, aliases: vec!["billing".into(), "spare".into()] },
        ];
        let routes = route_table(&list, &[], |path| (file_stem(path), vec![]), false);
        let names: Vec<(&str, &str, Option<&str>, bool)> = routes.iter().map(|r| (r.name.as_str(), r.kind, r.instance.as_deref(), r.conflict)).collect();
        assert_eq!(names, [
            ("qwen", "name", Some("a"), false),
            ("qwen-i-2", "name", Some("b"), false),
            ("billing", "instance_alias", Some("b"), true),
            ("other", "name", Some("c"), false),
            ("billing", "instance_alias", Some("c"), true),
            ("spare", "instance_alias", Some("c"), false),
        ]);
        assert_eq!(listed_names(&routes), ["qwen", "qwen-i-2", "other", "spare"]);
    }
}
