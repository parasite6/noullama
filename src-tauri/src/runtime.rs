use crate::api_keys::{Access, KeyStore};
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
    // Where a load runs: "" (the first GPU the GPU runtime lists, else the
    // CPU), "CPU", a llama.cpp device name such as "CUDA0", or a split across
    // GPUs: "gpus:all" or a comma list such as "CUDA0,CUDA1". Saved by the
    // Loading tab load panel. See devices.rs.
    pub device: String,
    // For a split only: "layer" or "row" (`--split-mode`), and an optional
    // ratio such as "3,1", one number per GPU in listed order
    // (`--tensor-split`). Empty shares by free memory.
    pub split_mode: String,
    pub tensor_split: String,
    // The llama-server for loads on the CPU; resolved like `runtime_path`.
    // Saved by the Runtime tab.
    pub cpu_runtime_path: String,
    pub api_port: u16,
    pub require_api_key: bool,
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
    // Saved by the Loading tab load panel.
    pub keep_context: bool,
    pub allow_force_fit: bool,
    pub hide_crowded_load_warning: bool,
    // When slot 1 of a file is not loaded, its unsuffixed names reach the
    // loaded instance with the lowest slot instead of answering 503.
    // Changed only by set_fall_through.
    pub fall_through_to_other_instances: bool,
    // What a request for `any-available` does when every ready instance is
    // running a request: "reject" (503) or "wait" (up to
    // `any_available_wait_ms`). Changed only by set_any_available_when_busy.
    pub any_available_when_busy: String,
    // How long "wait" waits, in milliseconds (WAIT_MS_MIN..=WAIT_MS_MAX; a
    // value outside it in settings.json counts as the nearest bound).
    // Changed only by set_any_available_wait_ms.
    pub any_available_wait_ms: u64,
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
            device: String::new(),
            split_mode: "layer".into(),
            tensor_split: String::new(),
            cpu_runtime_path: String::new(),
            api_port: 11435,
            require_api_key: false,
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
            any_available_wait_ms: WAIT_MS_DEFAULT,
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
    pub api_keys: Arc<KeyStore>,
    // models.yml, and the sha256 work around it.
    pub library: Arc<Mutex<crate::library::Library>>,
    pub hashing: Arc<crate::library::HashJobs>,
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
    /// Wakes `any-available` requests when instance availability changes.
    pub instance_freed: Arc<tokio::sync::Notify>,
    /// The GPU runtime's `--list-devices`, until the runtime changes.
    pub devices: Arc<Mutex<crate::devices::DeviceCache>>,
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
    /// Queue requests here so keys can be checked before using the runtime slot.
    dispatch_lock: Mutex<()>,
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
            dispatch_lock: Mutex::new(()),
        }))
    }
}

#[derive(Clone, Serialize)]
pub struct LoadedModel {
    pub path: String,
    pub device: String,
    // The split was chosen as All GPUs, so it is shown as "All GPUs (N)".
    #[serde(skip)]
    pub all_gpus: bool,
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
/// `any_available_when_busy` is "wait", in milliseconds: the default and the
/// range Settings accepts.
pub const WAIT_MS_DEFAULT: u64 = 120_000;
pub const WAIT_MS_MIN: u64 = 1;
pub const WAIT_MS_MAX: u64 = 600_000;

/// A wait as the Settings tab and the 503 message state it: whole seconds as
/// seconds, anything else in milliseconds.
pub fn wait_text(ms: u64) -> String {
    match ms {
        1000 => "1 second".into(),
        _ if ms % 1000 == 0 => format!("{} seconds", ms / 1000),
        _ => format!("{ms} ms"),
    }
}

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
        Some(end) if name.get(end..).is_some_and(|suffix| suffix.eq_ignore_ascii_case(".gguf")) => name[..end].to_owned(),
        _ => name,
    }
}

/// One loaded model: its own llama-server on its own private port.
pub struct Instance {
    pub(crate) loaded_sha256: std::sync::OnceLock<String>,
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
    /// Set while the load checks its files' sha256 values, before launch.
    verifying: AtomicBool,
}

impl Instance {
    fn matches_loaded_sum(&self, access: &Access) -> bool {
        access.allows_loaded_sum(self.path(), self.loaded_sha256.get().map(String::as_str))
    }

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
/// goes away mid-request, and wakes waiting `any-available` requests when the instance becomes idle.
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
        if self.instance.in_flight.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.freed.notify_waiters();
        }
    }
}

/// Why `any-available` found no instance.
pub enum AnyError {
    LoadedChecksum,
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
    LoadedChecksum,
    /// The request named no model.
    Missing(String),
    /// The named model is not loaded or still loading.
    Unavailable(String),
    Unknown(String),
    /// An identifier that reaches more than one model.
    Ambiguous(String),
    Forbidden,
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
    #[cfg(test)]
    pub fn new(data_dir: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_config_dir(data_dir.clone(), data_dir)
    }

    pub fn with_config_dir(data_dir: PathBuf, config_dir: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
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
            api_keys: Arc::new(KeyStore::new(config_dir)),
            library: Default::default(),
            hashing: Default::default(),
            mismatches: Default::default(),
            hash_worker: Default::default(),
            hash_cache_lock: Default::default(),
            instances_added: Default::default(),
            request_seq: Default::default(),
            instance_freed: Default::default(),
            devices: Default::default(),
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
        self.routes_with_access(&Access::default()).await.0
    }

    async fn routes_with_access(&self, access: &Access) -> (Vec<Route>, Option<std::collections::HashSet<String>>, Vec<Arc<Instance>>) {
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
        let routes = route_table(&table, &entries, names, fall_through);
        (routes, access.allowed_paths(self, &library).await, instances)
    }
    /// Names that reach a ready model right now, as `/v1/models` lists them:
    /// the route table's, then `any-available` when any instance is ready.
    pub async fn model_names(&self, access: &Access) -> Vec<String> {
        let (mut routes, allowed, instances) = self.routes_with_access(access).await;
        let any_ready = if let Some(allowed) = allowed {
            let eligible = |instance: &Instance| allowed.contains(instance.path()) && instance.matches_loaded_sum(access);
            let ready = instances.iter().any(|instance| instance.is_ready() && eligible(instance));
            routes.retain(|route| access.allows_alias_owner(&route.base, &route.path)
                && instances.iter().any(|instance| route.instance.as_deref() == Some(instance.id.as_str()) && eligible(instance)));
            ready
        } else {
            self.any_ready().await
        };
        let mut names = listed_names(&routes);
        if any_ready {
            names.push(ANY_AVAILABLE.to_owned());
        }
        names
    }
    /// A request for `any-available`: claims the ready instance that has been
    /// idle the longest. When every ready instance is busy, answers at once
    /// (setting "reject") or waits for the first to free up (setting "wait",
    /// up to `any_available_wait_ms`). The claim is held until the request ends.
    pub async fn claim_any(&self, access: &Access) -> Result<Claim, AnyError> {
        let (wait, wait_ms) = {
            let settings = self.settings.lock().await;
            (settings.any_available_when_busy == "wait", settings.any_available_wait_ms.clamp(WAIT_MS_MIN, WAIT_MS_MAX))
        };
        let deadline = tokio::time::Instant::now() + Duration::from_millis(wait_ms);
        loop {
            // Registered before looking, so a request that finishes in between still wakes it.
            let freed = self.instance_freed.notified();
            tokio::pin!(freed);
            freed.as_mut().enable();
            let mut instances = self.instances.lock().await.clone();
            if access.0.is_some() {
                crate::library::refresh(self).await;
                let library = self.library.lock().await;
                let allowed = access.allowed_paths(self, &library).await.unwrap_or_default();
                let had_ready = instances.iter().any(|instance| instance.is_ready() && allowed.contains(instance.path()));
                instances.retain(|instance| allowed.contains(instance.path()) && instance.matches_loaded_sum(access));
                if had_ready && !instances.iter().any(|instance| instance.is_ready()) {
                    return Err(AnyError::LoadedChecksum);
                }
            }
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
                return Err(AnyError::Busy(format!("Every loaded instance stayed busy for {}, so \"{ANY_AVAILABLE}\" gave up on this request. Try again.", wait_text(wait_ms))));
            }
        }
    }
    /// The loaded model a request's `model` field names: one of its file's
    /// names or aliases in models.yml plus the instance's suffix (a loaded
    /// file with no entry answers to its file name). Every request must name
    /// one, exactly as written.
    pub async fn route(&self, requested: Option<&str>, access: &Access) -> Result<Arc<Instance>, RouteError> {
        use RouteError::*;
        let Some(name) = requested.filter(|n| !n.is_empty()) else {
            let names = self.model_names(access).await;
            return Err(Missing(if names.is_empty() {
                "Set \"model\" to a loaded model's name or alias. No model is loaded yet; load one in noullama.".into()
            } else {
                format!("Set \"model\" to a loaded model's name or alias: {}.", names.join(", "))
            }));
        };
        let (routes, allowed, instances) = self.routes_with_access(access).await;
        let matches: Vec<&Route> = routes.iter().filter(|r| r.name == name).collect();
        if let Some(allowed) = allowed {
            if matches.is_empty() || matches.iter().any(|route| !allowed.contains(&route.path) || !access.allows_alias_owner(&route.base, &route.path)) {
                return Err(Forbidden);
            }
        }
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
                let mut names = listed_names(&routes);
                if self.any_ready().await {
                    names.push(ANY_AVAILABLE.to_owned());
                }
                Err(Unknown(if names.is_empty() {
                    format!("Unknown model '{name}'.{hint} No model is ready; load one in noullama.")
                } else {
                    format!("Unknown model '{name}'.{hint} Loaded models answer to: {}.", names.join(", "))
                }))
            }
            [one] => match one.instance.as_deref().and_then(|id| instances.iter().find(|i| i.id == id)) {
                Some(instance) if instance.is_ready() && !instance.matches_loaded_sum(access) => Err(LoadedChecksum),
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
                    if instance_alias { "Remove the instance alias in Loading, or give them different names in Library." } else if suffixed { "Give them different names in Library, or change the instance suffix in Loading." } else { "Give them different names in Library." }
                )))
            }
        }
    }
    /// Runs a decision on one loaded model, with earlier decisions added when
    /// Keep context is on.
    /// The request counts as running on it (see `Claim`) until it ends.
    pub async fn infer_on(&self, instance: &Arc<Instance>, payload: Value) -> Result<Value, String> {
        let claim = self.claim_named(instance);
        let _slot = claim.instance.session.dispatch_lock.lock().await;
        crate::context::infer(self, &claim.instance, payload).await
    }

    pub fn claim_named(&self, instance: &Arc<Instance>) -> Claim {
        Claim::take(self, instance)
    }

    pub async fn infer_claimed(
        &self,
        claim: &Claim,
        payload: Value,
        access: &Access,
    ) -> Result<Result<Value, String>, crate::api_keys::Rejection> {
        let _slot = claim.instance.session.dispatch_lock.lock().await;
        self.api_keys.revalidate(access)?;
        Ok(crate::context::infer(self, &claim.instance, payload).await)
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

const SERVER_NAME: &str = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };
// Bounds for looking inside a selected folder, so a broad or hostile folder
// cannot make the search slow or lead it outside that folder.
const SERVER_SEARCH_DEPTH: usize = 3;
const SERVER_SEARCH_ENTRIES: usize = 2_000;

/// The executable to run for `path`, or `path` itself when it cannot be
/// resolved (the launch then reports the problem).
pub fn server_binary(path: &str) -> PathBuf {
    resolve_server(path).unwrap_or_else(|_| PathBuf::from(path.trim()))
}

/// Finds llama-server for a runtime path: the file itself, `llama-server`
/// in the folder, or the one llama-server inside the folder or inside the
/// managed build the folder belongs to (archives may unpack into several
/// package folders, such as a separate CUDA runtime package).
pub fn resolve_server(path: &str) -> Result<PathBuf, String> {
    let path = path.trim();
    if path.is_empty() {
        return Err("No llama.cpp build is selected. Choose one in Runtime.".into());
    }
    let given = PathBuf::from(path);
    let meta = std::fs::metadata(&given)
        .map_err(|_| format!("The selected llama.cpp build is missing: {path}. Choose an installed build in Runtime."))?;
    if meta.is_file() {
        return Ok(given);
    }
    if !meta.is_dir() {
        return Err(format!("{path} is not a file or folder. Choose llama-server or its folder in Runtime."));
    }
    let direct = given.join(SERVER_NAME);
    if direct.is_file() {
        return Ok(direct);
    }
    let root = managed_build_root(&given).unwrap_or(given);
    let found = find_servers(&root);
    match found.as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(format!("No {SERVER_NAME} was found in {}. Choose llama-server or its folder in Runtime.", root.display())),
        many => Err(format!(
            "{} contains several {SERVER_NAME} files ({}). Choose one of them in Runtime.",
            root.display(),
            many.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// The managed build folder (holding `noullama-install.json`) that `dir` is
/// in, checking `dir` and up to three parents.
fn managed_build_root(dir: &Path) -> Option<PathBuf> {
    dir.ancestors().take(4).find(|d| d.join("noullama-install.json").is_file()).map(Path::to_path_buf)
}

/// Every regular `llama-server` file under `root`, at most
/// `SERVER_SEARCH_DEPTH` levels down and `SERVER_SEARCH_ENTRIES` entries,
/// without following symlinked folders or entering hidden ones.
fn find_servers(root: &Path) -> Vec<PathBuf> {
    let mut pending = vec![(root.to_path_buf(), 0)];
    let mut found = Vec::new();
    let mut seen = 0;
    while let Some((dir, depth)) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            seen += 1;
            if seen > SERVER_SEARCH_ENTRIES {
                found.sort();
                return found;
            }
            let name = entry.file_name();
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_dir() {
                if depth < SERVER_SEARCH_DEPTH {
                    pending.push((entry.path(), depth + 1));
                }
            } else if kind.is_file() && name == SERVER_NAME {
                found.push(entry.path());
            }
        }
    }
    found.sort();
    found
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

pub(crate) fn runtime_command(settings: &Settings) -> Command {
    let binary = server_binary(&settings.runtime_path);
    let mut command = Command::new(&binary);
    if let Ok(paths) = std::env::join_paths(runtime_library_dirs(settings, &binary)) {
        command.env("LD_LIBRARY_PATH", paths);
    }
    command
}

#[tauri::command]
pub async fn snapshot(state: tauri::State<'_, AppState>) -> Result<String, String> {
    let snapshot = state_snapshot(state.inner()).await?;
    serde_json::to_string(&snapshot).map_err(|error| error.to_string())
}

/// The load task reports its own failures; this catches a runtime that
/// stopped after it was ready. Runs from the UI snapshot and from the
/// supervisor tick, so a crash is noticed while the window is hidden too.
async fn reap_exited(state: &AppState) -> Result<(), String> {
    let Ok(_guard) = state.lifecycle.try_lock() else { return Ok(()) };
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
    let jobs = state.hashing.lock().unwrap_or_else(|error| error.into_inner()).clone();
    let mut loaded = Vec::new();
    for instance in &instances {
        let model = instance.model();
        let history = instance.context.lock().await;
        let extras = instance_extras(&instance.settings);
        let required = crate::hardware::estimate_bytes(instance.path(), &extras, instance.settings.context_size);
        let (name, _) = state.api_names(instance.path()).await;
        let api_names: Vec<&str> = routes.iter().filter(|r| r.instance.as_deref() == Some(&instance.id) && !r.fallthrough).map(|r| r.name.as_str()).collect();
        loaded.push(json!({
            "id": instance.id, "name": name, "label": instance.label,
            "slot": instance.slot, "suffix": instance.suffix(), "default_suffix": default_suffix(instance.slot), "api_names": api_names,
            "instance_aliases": instance.aliases(),
            "path": model.path, "device": model.device, "runtime_path": model.runtime_path,
            "projector": model.projector, "mmproj_path": instance.settings.mmproj_path,
            "mtp": model.mtp, "mtp_path": instance.settings.mtp_path, "layout": model.layout,
            "ready": instance.is_ready(), "verifying": verifying_status(instance, &jobs), "context_size": instance.settings.context_size,
            "gpu_layers": crate::devices::gpu_layers(&instance.settings), "required_bytes": required,
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
    let cpu_runtime_download = state.runtimes.cpu_progress.lock().await.clone();
    // The probe runs in the background from startup; until it finishes the
    // snapshot reports no hardware rather than waiting on nvidia-smi.
    // The load panel's fit comes from `load_fit`, for its unsaved values.
    let hardware = crate::hardware::cached();
    Ok(
        json!({"settings": settings, "running": running, "ready": ready,
        "instances": loaded, "default_model": default.map(|i| i.id.clone()), "routes": routes, "library": library,
        "logs": logs, "downloads": downloads, "loaded_model": loaded_model,
        "download_dir": model_directory(&settings, &state.data_dir),
        "api_port": state.listening_port(), "api_error": api_error, "api_notice": api_notice, "load_errors": load_errors, "runtime_download": runtime_download, "cpu_runtime_download": cpu_runtime_download, "startup_warning": state.startup_warning, "quit_confirmation":state.app.quit_confirmation.load(std::sync::atomic::Ordering::SeqCst), "data_dir": state.data_dir, "hardware": hardware,
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

/// Saves the Loading tab load panel, the Runtime tab or the download
/// directory. `settings` holds only the fields that form changes; the rest
/// stay as they are saved now.
#[tauri::command]
pub async fn save_settings(
    settings: serde_json::Map<String, Value>,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    save_from_editor(state.inner(), settings).await
}

/// Fields the settings forms never save: the alias list and the Settings tab
/// switches have their own commands, so a form's copy of them (which can be
/// stale) is ignored. `keep_context` (the default for new loads) is a load
/// panel field and is saved from the form.
const SERVER_OWNED: [&str; 8] = [
    "model_routes",
    "require_api_key",
    "sha256_check",
    "allow_force_fit",
    "hide_crowded_load_warning",
    "fall_through_to_other_instances",
    "any_available_when_busy",
    "any_available_wait_ms",
];

/// Applies a form's changed fields over the saved settings.
pub async fn save_from_editor(state: &AppState, changes: serde_json::Map<String, Value>) -> Result<(), String> {
    update_settings(state, |settings| {
        *settings = with_changes(settings, changes)?;
        Ok(())
    })
    .await
}

/// `settings` with a form's changed fields applied, as a save would store them.
fn with_changes(settings: &Settings, changes: serde_json::Map<String, Value>) -> Result<Settings, String> {
    let mut base = settings.clone();
    let routes = std::mem::take(&mut base.model_routes);
    let Value::Object(mut merged) = serde_json::to_value(&base).map_err(|e| e.to_string())? else {
        return Err("Settings are not an object.".into());
    };
    let known: Vec<String> = merged.keys().cloned().collect();
    for (key, value) in changes {
        if SERVER_OWNED.contains(&key.as_str()) {
            continue;
        }
        if !known.contains(&key) {
            return Err(format!("Unknown setting \"{key}\"."));
        }
        merged.insert(key, value);
    }
    let mut next: Settings = serde_json::from_value(Value::Object(merged)).map_err(|e| format!("Invalid settings: {e}"))?;
    next.model_routes = routes;
    Ok(next)
}

/// The load panel's fit badge for its current, possibly unsaved, values
/// (`settings`, load panel fields only; the build and shared-library folders
/// are the saved ones, as a load uses). `replace` is the instance a reload
/// would replace; a fresh load passes none.
#[tauri::command]
pub async fn load_fit(settings: serde_json::Map<String, Value>, replace: Option<String>, state: tauri::State<'_, AppState>) -> Result<Value, String> {
    load_fit_in(state.inner(), settings, replace.as_deref(), crate::hardware::cached()).await
}

/// `load_fit` with the hardware passed in. The device comes from the same
/// listing and rule a load uses (`listed_for`, `choose_device`).
pub async fn load_fit_in(state: &AppState, changes: serde_json::Map<String, Value>, replace: Option<&str>, hardware: Option<&crate::hardware::Hardware>) -> Result<Value, String> {
    let candidate = with_changes(&state.settings.lock().await.clone(), changes)?;
    let listed = crate::devices::listed_for(state, &candidate).await;
    let instances = state.instances.lock().await.clone();
    let (fit, counted) = fit_for_load(&candidate, listed.as_deref(), &instances, replace, hardware)?;
    Ok(json!({ "fit": fit, "others": counted }))
}

/// How a load with `candidate` would fit, beside the loaded `instances`
/// other than `replace`. Returns the fit and how many instances it set
/// memory aside for: those on the CPU for a CPU load; on a GPU load, those on
/// a GPU it uses plus those on the CPU (graphics memory of other GPUs does not
/// matter to it). A split is rated against its GPUs' summed memory, and with
/// a ratio each GPU's share against that GPU. The reason is returned when
/// there is no estimate.
fn fit_for_load(
    candidate: &Settings,
    listed: Option<&[crate::devices::RuntimeDevice]>,
    instances: &[Arc<Instance>],
    replace: Option<&str>,
    hardware: Option<&crate::hardware::Hardware>,
) -> Result<(crate::hardware::Fit, usize), String> {
    if candidate.model_path.trim().is_empty() {
        return Err("Select a model to see how it would run on this machine.".into());
    }
    if crate::hardware::estimate_bytes(&candidate.model_path, &[] as &[&str], 0).is_none() {
        return Err(crate::hardware::MODEL_UNREADABLE.into());
    }
    let mut settings = candidate.clone();
    settings.device = crate::devices::choose_device(&settings, listed)?;
    let hardware = hardware.ok_or("Memory detection is still running. The estimate appears when it finishes.")?;
    let gpu_layers = crate::devices::gpu_layers(&settings);
    let on_gpu = gpu_layers != 0;
    let mut hardware = hardware.clone();
    // A split's GPUs with their own total memory as the runtime reports it.
    let mut split_gpus: Vec<(String, u64)> = Vec::new();
    if on_gpu && crate::devices::is_split(&settings.device) {
        for name in crate::devices::split_names(&settings.device) {
            let total = listed.and_then(|l| l.iter().find(|d| d.name == name)).and_then(|d| d.total_mib);
            let mib = total.ok_or_else(|| format!("{name} did not report its memory, so there is no estimate for this split."))?;
            split_gpus.push((name.to_owned(), mib * 1024 * 1024));
        }
        hardware.vram_bytes = Some(split_gpus.iter().map(|(_, bytes)| bytes).sum());
    } else if on_gpu {
        // That GPU's own memory as the runtime reports it; the machine-wide probe otherwise.
        let total = listed.and_then(|l| l.iter().find(|d| d.name == settings.device)).and_then(|d| d.total_mib);
        if let Some(mib) = total {
            hardware.vram_bytes = Some(mib * 1024 * 1024);
        }
    } else {
        hardware.vram_bytes = None;
    }
    let mut reserved = std::collections::HashMap::<String, u64>::new();
    let taken: Vec<(u64, bool)> = instances
        .iter()
        .filter(|i| Some(i.id.as_str()) != replace)
        .filter_map(|i| {
            let other_gpu = crate::devices::gpu_layers(&i.settings) != 0;
            if other_gpu && !(on_gpu && crate::devices::shares_gpu(&i.settings.device, &settings.device)) {
                return None;
            }
            let bytes = crate::hardware::estimate_bytes(i.path(), &instance_extras(&i.settings), i.settings.context_size)?;
            if !other_gpu { return Some((bytes, false)); }
            let overlap = gpu_overlap_bytes(&i.settings, &settings.device, bytes, &mut reserved);
            (overlap > 0).then_some((overlap, true))
        })
        .collect();
    let extras = instance_extras(&settings);
    let extras: Vec<&str> = extras.iter().map(String::as_str).collect();
    let mut fit = crate::hardware::fit_beside(&settings.model_path, &extras, settings.context_size, gpu_layers, &hardware, &taken)?;
    if let Ok(Some(weights)) = crate::devices::parse_ratio(&settings.tensor_split, Some(split_gpus.len())) {
        for (name, total) in &mut split_gpus {
            *total = total.saturating_sub(reserved.get(name).copied().unwrap_or_default());
        }
        fit = crate::hardware::check_shares(fit, &split_gpus, &weights);
    }
    Ok((fit, taken.len()))
}

/// Aggregate VRAM on the candidate's GPUs, recording known per-device shares.
fn gpu_overlap_bytes(existing: &Settings, candidate: &str, bytes: u64, by_device: &mut std::collections::HashMap<String, u64>) -> u64 {
    let names = crate::devices::split_names(&existing.device);
    let target = crate::devices::split_names(candidate);
    let weights = if names.len() == 1 {
        Some(vec![1.0])
    } else {
        crate::devices::parse_ratio(&existing.tensor_split, Some(names.len())).ok().flatten()
    };
    let Some(weights) = weights else {
        // Without a ratio, only the full device set has a known estimate.
        return if names.iter().all(|name| target.contains(name)) { bytes } else { 0 };
    };
    let sum: f64 = weights.iter().sum();
    let mut overlapping = 0u64;
    for (name, weight) in names.into_iter().zip(weights).filter(|(name, _)| target.contains(name)) {
        let share = (bytes as f64 * (weight / sum)) as u64;
        let entry = by_device.entry(name.to_owned()).or_default();
        *entry = entry.saturating_add(share);
        overlapping = overlapping.saturating_add(share);
    }
    overlapping
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

/// The one way settings change. It holds the lifecycle lock from reading the
/// saved settings, through `change` and the write of settings.json, to
/// replacing them in memory, so two changes close together both apply
/// instead of one writing back the other's old values. Loaded models keep
/// the settings they started with, so settings can change at any time; they
/// apply to the next load.
///
/// Lock order: the lifecycle lock first, then the short `settings` and
/// `downloads` locks. Never call this while holding the lifecycle lock: tokio's
/// Mutex is not reentrant, so that would hang.
pub async fn update_settings(state: &AppState, change: impl FnOnce(&mut Settings) -> Result<(), String>) -> Result<(), String> {
    let _guard = state.lifecycle.lock().await;
    let current = state.settings.lock().await.clone();
    let mut settings = current.clone();
    change(&mut settings)?;
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
    if crate::devices::is_split(&settings.device)
        && (&settings.device, &settings.split_mode, &settings.tensor_split) != (&current.device, &current.split_mode, &current.tensor_split)
    {
        crate::devices::check_split_settings(&settings)?;
    }
    if !settings.download_dir.trim().is_empty()
        && !Path::new(settings.download_dir.trim()).is_absolute()
    {
        return Err("Choose an absolute path for the download directory.".into());
    }
    if current.download_dir != settings.download_dir
        && state.downloads.lock().await.iter().any(|d| d.active)
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

pub(crate) async fn probe_output(
    command: &mut Command,
    limit: Duration,
) -> Result<std::io::Result<std::process::Output>, tokio::time::error::Elapsed> {
    tokio::time::timeout(limit, command.kill_on_drop(true).output()).await
}

#[tauri::command]
pub async fn probe_runtime(path: String, library_path: String) -> Result<String, String> {
    let settings = Settings {
        runtime_path: path,
        library_path,
        ..Settings::default()
    };
    let server = resolve_server(&settings.runtime_path)?;
    let output = probe_output(runtime_command(&settings).arg("--version"), Duration::from_secs(10))
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
    let devices = probe_output(runtime_command(&settings).arg("--list-devices"), Duration::from_secs(15))
    .await
    .map_err(|_| "Device detection timed out")?
    .map_err(|e| e.to_string())?;
    Ok(format!(
        "Using {}\n{text}\n{}{}",
        server.display(),
        String::from_utf8_lossy(&devices.stdout),
        String::from_utf8_lossy(&devices.stderr)
    ))
}

/// Loads the selected model beside any already loaded; returns its id.
#[tauri::command]
pub async fn start_runtime(slot: Option<u32>, replace: Option<String>, state: tauri::State<'_, AppState>) -> Result<String, String> {
    load_model_replacing(state.inner(), slot, replace.as_deref()).await
}

/// `slot` is the slot of the instance a reload replaced: the new instance
/// keeps it when it is free, so its suffixed names stay the same. Otherwise
/// (and for fresh loads) it takes the lowest free slot.
#[cfg(test)]
pub async fn load_model_in(state: &AppState, slot: Option<u32>) -> Result<String, String> {
    load_model_replacing(state, slot, None).await
}

async fn load_model_replacing(state: &AppState, slot: Option<u32>, replace: Option<&str>) -> Result<String, String> {
    let selected = state.settings.lock().await.model_path.clone();
    if Path::new(&selected).is_file() {
        crate::library::ensure(state, &[selected]).await;
    }
    // Listing the GPU runtime's devices can take seconds, so it runs before
    // the lifecycle lock (usually from the cache the load panel filled).
    let listed_with = state.settings.lock().await.clone();
    let mut listed = crate::devices::listed_for(state, &listed_with).await;
    let _guard = state.lifecycle.lock().await;
    if state.app.exiting.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("Noullama is shutting down.".into());
    }
    let mut settings = state.settings.lock().await.clone();
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
            return Err("This model has no built-in MTP layers. Choose an mtp- head file in Loading, or turn MTP off.".into());
        }
    }
    let unchanged = |a: &Settings, b: &Settings| (&a.runtime_path, &a.library_path, &a.device) == (&b.runtime_path, &b.library_path, &b.device);
    if !unchanged(&settings, &listed_with) {
        listed = crate::devices::listed_for(state, &settings).await;
    }
    // The instance keeps the device it was loaded on, never the empty default
    // (a split keeps its GPUs as a comma list).
    let all_gpus = settings.device.trim() == crate::devices::ALL_GPUS;
    settings.device = crate::devices::choose_device(&settings, listed.as_deref())?;
    let devices = listed.unwrap_or_default();
    let device = crate::devices::display(&settings.device, all_gpus, &devices).unwrap_or_else(|| settings.device.clone());
    let loaded = state.instances.lock().await.clone();
    let replacement = replace.and_then(|id| loaded.iter().find(|instance| instance.id == id));
    let checked_devices = if replacement.is_some() {
        Some(check_launch_devices(&settings, all_gpus).await?)
    } else {
        if let Some(reason) = load_memory_risk(state, &settings, loaded.iter(), (None, 0), &devices, false).await {
            return Err(format!("OOM prevention: {reason} Eject another model, reduce context or GPU offload, or enable Allow force fit in Settings."));
        }
        None
    };
    let port = free_runtime_port(state, &settings, &loaded)?;
    let others: Vec<_> = loaded.iter().filter(|instance| Some(instance.id.as_str()) != replace).cloned().collect();
    let preferred_slot = replacement.map(|instance| instance.slot).or(slot);
    let (slot, label) = slot_and_label(&others, &settings.model_path, preferred_slot);
    let session = RuntimeSession::new(&state.data_dir, port)?;
    let instance = Arc::new(new_instance(settings, slot, label, session, device, all_gpus));
    state.load_errors.lock().await.retain(|e| !(e.path == instance.path() && e.slot == instance.slot));
    if let Some(previous) = replacement {
        state.unload_locked(&previous.id).await;
    }
    state.instances.lock().await.push(instance.clone());
    state.instances_added.notify_one();
    state.log(instance.load_marker()).await;
    let id = instance.id.clone();
    let state = state.clone();
    tokio::spawn(async move {
        let result = tokio::select! {
            result = tokio::time::timeout(LOAD_LIMIT, prepare_and_poll(&state, &instance, checked_devices)) => result,
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

/// A new, still loading instance in `slot`: the slot's default suffix, no
/// instance aliases, and Keep context from the `keep_context` setting.
/// `settings.device` is the chosen device; `device` is how it is shown, and
/// `all_gpus` whether it was chosen as All GPUs.
fn new_instance(settings: Settings, slot: u32, label: String, session: Arc<RuntimeSession>, device: String, all_gpus: bool) -> Instance {
    Instance {
        loaded_sha256: std::sync::OnceLock::new(),
        id: uuid::Uuid::new_v4().simple().to_string()[..12].to_owned(),
        label,
        slot,
        suffix: std::sync::Mutex::new(default_suffix(slot)),
        model: std::sync::Mutex::new(LoadedModel {
            path: settings.model_path.clone(),
            device,
            all_gpus,
            runtime_path: crate::devices::launch_settings(&settings).runtime_path,
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
        verifying: Default::default(),
    }
}

/// Clears `Instance::verifying` however the check ends, cancelled included.
struct Verifying<'a>(&'a AtomicBool);

impl Drop for Verifying<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// The load's sha256 check, marked on the instance while it runs.
async fn verify_instance(state: &AppState, instance: &Instance) -> Result<Option<String>, String> {
    instance.verifying.store(true, Ordering::SeqCst);
    let _verifying = Verifying(&instance.verifying);
    crate::library::verify_load(state, &instance.settings, &instance.prefix()).await
}

/// The file a loading instance is hashing right now, with its progress, or
/// null. Only hashes the load runs count: a sum the cache already had, or
/// a background hash after the check, shows nothing.
fn verifying_status(instance: &Instance, jobs: &BTreeMap<String, crate::library::HashJob>) -> Value {
    if instance.is_ready() || !instance.verifying.load(Ordering::SeqCst) {
        return Value::Null;
    }
    let settings = &instance.settings;
    let mtp = if settings.mtp { settings.mtp_path.as_str() } else { "" };
    [("model", settings.model_path.as_str()), ("mtp", mtp), ("mmproj", settings.mmproj_path.as_str())]
        .into_iter()
        .find_map(|(slot, file)| {
            let job = jobs.get(file).filter(|_| !file.is_empty())?;
            Some(json!({"file": file, "slot": slot, "done_bytes": job.done_bytes, "total_bytes": job.total_bytes}))
        })
        .unwrap_or(Value::Null)
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

/// The process a load starts: the runtime for its device (the CPU runtime
/// with no GPU layers for "CPU"), its library folders, the launch arguments
/// and the flags that keep it on its device or split.
fn launch_command(settings: &Settings, session: &RuntimeSession) -> Command {
    let launch = crate::devices::launch_settings(settings);
    let mut command = runtime_command(&launch);
    command.args(runtime_args(&launch, session)).args(crate::devices::device_args(settings));
    command
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

async fn check_launch_devices(settings: &Settings, all_gpus: bool) -> Result<(Vec<crate::devices::RuntimeDevice>, String), String> {
    // Detect devices with the runtime this load launches: the GPU runtime
    // must still list each chosen GPU; the CPU runtime only has to run.
    let devices = crate::devices::list_devices(&crate::devices::launch_settings(settings)).await?;
    let display = match crate::devices::display(&settings.device, all_gpus, &devices) {
        Some(display) => display,
        None => {
            let names = crate::devices::split_names(&settings.device);
            let missing = names.iter().find(|name| !devices.iter().any(|d| d.name == **name)).copied().unwrap_or(&settings.device);
            return Err(crate::devices::not_found(missing));
        }
    };
    Ok((devices, display))
}

async fn prepare_and_poll(state: &AppState, instance: &Arc<Instance>, checked_devices: Option<(Vec<crate::devices::RuntimeDevice>, String)>) -> Result<(), String> {
    let settings = &instance.settings;
    let session = &instance.session;
    let prefix = instance.prefix();
    let fingerprint = crate::library::fingerprint(instance.path());
    let verified = verify_instance(state, instance).await?;
    // Keep the port reserved during preflight, then verify the launch's random
    // alias over an authenticated endpoint before ever accepting inference.
    let reservation = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, session.port))
        .await
        .map_err(|e| format!("Private runtime port is unavailable: {e}"))?;
    let fresh_devices = checked_devices.is_none();
    let (devices, device) = match checked_devices {
        Some(checked) => checked,
        None => check_launch_devices(settings, instance.model().all_gpus).await?,
    };
    let (loaded_sum, launch_fingerprint) = {
        let _guard = state.lifecycle.lock().await;
        if !is_registered(state, instance).await {
            return Ok(());
        }
        let mut live_settings = settings.clone();
        live_settings.allow_force_fit = state.settings.lock().await.allow_force_fit;
        let loaded = state.instances.lock().await.clone();
        let others = loaded.iter().filter(|other| other.id != instance.id);
        if let Some(reason) = load_memory_risk(state, &live_settings, others, (None, 0), &devices, fresh_devices).await {
            return Err(format!("OOM prevention: {reason}"));
        }
        let launch_fingerprint = crate::library::fingerprint(instance.path());
        let loaded_sum = match verified {
            Some(sum) if fingerprint == launch_fingerprint => Some(sum),
            Some(_) => None,
            None => crate::library::cached_hashes(state, &[instance.path()]).await.remove(instance.path()),
        };
        drop(reservation);
        let mut child = launch_command(settings, session)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                format!(
                    "Could not start {}: {e}",
                    server_binary(&crate::devices::launch_settings(settings).runtime_path).display()
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
        (loaded_sum, launch_fingerprint)
    };
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
                        return Err("Runtime did not advertise built-in decision support. Use a decision GGUF and a compatible llama.cpp build, or set Decision mode to Auto in Loading.".into());
                    }
                    None => {
                        return Err("This model has no built-in decision support, and Noullama has no prompt layout for it. Winnow GGUFs are read through the Prompt layout (Winnow); if this is a Winnow model under another name, set Decision mode to Prompt layout in Loading.".into());
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
                if let Some(sum) = loaded_sum.filter(|_| launch_fingerprint == crate::library::fingerprint(instance.path())) {
                    let _ = instance.loaded_sha256.set(sum);
                }
                instance.ready.store(true, Ordering::SeqCst);
                state.instance_freed.notify_waiters();
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
    update_settings(state, |s| { s.any_available_when_busy = mode.to_owned(); Ok(()) }).await
}

/// The Settings tab's "Wait up to (ms)" for `any-available`.
#[tauri::command]
pub async fn set_any_available_wait_ms(ms: u64, state: tauri::State<'_, AppState>) -> Result<(), String> {
    set_wait_ms(state.inner(), ms).await
}

pub async fn set_wait_ms(state: &AppState, ms: u64) -> Result<(), String> {
    if !(WAIT_MS_MIN..=WAIT_MS_MAX).contains(&ms) {
        return Err(format!("Use a wait of {WAIT_MS_MIN}–{WAIT_MS_MAX} ms."));
    }
    update_settings(state, |s| { s.any_available_wait_ms = ms; Ok(()) }).await
}

/// The Settings tab's "Fall through to another instance".
#[tauri::command]
pub async fn set_fall_through(on: bool, state: tauri::State<'_, AppState>) -> Result<(), String> {
    change_fall_through(state.inner(), on).await
}

pub async fn change_fall_through(state: &AppState, on: bool) -> Result<(), String> {
    update_settings(state, |s| { s.fall_through_to_other_instances = on; Ok(()) }).await
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

/// Settings' "Allow force fit" and the crowded-load dialog's "Don't show again".
#[tauri::command]
pub async fn set_memory_preference(key: String, value: bool, state: tauri::State<'_, AppState>) -> Result<(), String> {
    set_memory_pref(state.inner(), &key, value).await
}

pub async fn set_memory_pref(state: &AppState, key: &str, value: bool) -> Result<(), String> {
    update_settings(state, |s| {
        match key {
            "allow_force_fit" => s.allow_force_fit = value,
            "hide_crowded_load_warning" => s.hide_crowded_load_warning = value,
            _ => return Err("Unknown memory preference".into()),
        }
        Ok(())
    })
    .await
}

/// `freed_gpu` and `freed_ram`: memory a loaded instance being replaced
/// holds now, counted as available because it is unloaded before the load.
/// `settings.device` is the chosen device (one GPU or a split's GPUs);
/// `devices` is the GPU runtime's list, used to find each GPU's own
/// nvidia-smi reading. A split with any GPU unread skips the check and logs
/// which one, rather than guessing from another GPU.
async fn load_memory_risk<'a>(
    state: &AppState,
    settings: &Settings,
    loaded: impl Iterator<Item = &'a Arc<Instance>>,
    freed: (Option<(&Settings, u64)>, u64),
    devices: &[crate::devices::RuntimeDevice],
    fresh: bool,
) -> Option<String> {
    let layers = crate::devices::gpu_layers(settings);
    let vram = if !settings.allow_force_fit && (layers == -1 || layers >= 99) {
        let names: Vec<String> = crate::devices::split_names(&settings.device).into_iter().map(str::to_owned).collect();
        let listed = devices.to_vec();
        let wanted = names.clone();
        let mut readings: Vec<Option<(u64, u64)>> = tokio::task::spawn_blocking(move || {
            wanted.iter().map(|name| crate::hardware::nvidia_device_memory(name, &listed)).collect()
        }).await.unwrap_or_else(|_| vec![None; names.len()]);
        if readings.iter().any(Option::is_none) {
            let relisted = if fresh { None } else { crate::devices::list_devices(settings).await.ok() };
            let list = if fresh { Some(devices) } else { relisted.as_deref() };
            for (reading, name) in readings.iter_mut().zip(&names) {
                if reading.is_none() {
                    *reading = list.and_then(|list| crate::hardware::listed_device_memory(name, list));
                }
            }
        }
        let missing: Vec<&str> = names.iter().zip(&readings).filter(|(_, r)| r.is_none()).map(|(n, _)| n.as_str()).collect();
        if missing.is_empty() {
            Some(names.into_iter().zip(readings.into_iter().flatten()).map(|(name, (total, used))| (name, total, used)).collect::<Vec<_>>())
        } else {
            state.log(format!("VRAM preflight unavailable for device {}.", missing.join(", "))).await;
            None
        }
    } else {
        None
    };
    load_memory_risk_with(settings, loaded, freed, || crate::oom::memory().map(|m| m.available), move || vram).await
}

/// `load_memory_risk` with the readings passed in: `available_ram`, and
/// `vram` as `(name, total, used)` of each GPU this load is on. Each is read
/// only when the check needs it.
async fn load_memory_risk_with<'a>(
    settings: &Settings,
    loaded: impl Iterator<Item = &'a Arc<Instance>>,
    (freed_gpu, freed_ram): (Option<(&Settings, u64)>, u64),
    available_ram: impl FnOnce() -> Option<u64> + Send + 'static,
    vram: impl FnOnce() -> Option<Vec<(String, u64, u64)>> + Send + 'static,
) -> Option<String> {
    if settings.allow_force_fit { return None; }
    let required = crate::hardware::estimate_bytes(&settings.model_path, &instance_extras(settings), settings.context_size)?;
    let gpu_layers = crate::devices::gpu_layers(settings);
    // Still-loading instances are not in the live readings yet. Graphics
    // memory is per GPU, so only those sharing a GPU with this load count
    // there; system memory is shared by every CPU-side load.
    let mut gpu_reserved = std::collections::HashMap::new();
    let reserved = loaded
        .filter(|other| !other.is_ready())
        .filter(|other| {
            let other_gpu = crate::devices::gpu_layers(&other.settings) != 0;
            other_gpu == (gpu_layers != 0) && (gpu_layers == 0 || crate::devices::shares_gpu(&other.settings.device, &settings.device))
        })
        .filter_map(|other| {
            let bytes = crate::hardware::estimate_bytes(other.path(), &instance_extras(&other.settings), other.settings.context_size)?;
            Some(if gpu_layers == 0 { bytes } else { gpu_overlap_bytes(&other.settings, &settings.device, bytes, &mut gpu_reserved) })
        })
        .fold(0u64, u64::saturating_add);
    let mut gpu_freed = std::collections::HashMap::new();
    let freed_vram = freed_gpu.map_or(0, |(other, bytes)| gpu_overlap_bytes(other, &settings.device, bytes, &mut gpu_freed));
    let weights = if crate::devices::is_split(&settings.device) {
        crate::devices::parse_ratio(&settings.tensor_split, Some(crate::devices::split_names(&settings.device).len())).ok().flatten()
    } else {
        None
    };
    tokio::task::spawn_blocking(move || {
        if gpu_layers == -1 || gpu_layers >= 99 {
            if let Some(gpus) = vram() {
                if let Some(reason) = crate::hardware::gpu_shortfall(required, &gpus, freed_vram, reserved, weights.as_deref(), &gpu_freed, &gpu_reserved) {
                    return Some(reason);
                }
            }
        }
        // Partial offload cannot be divided accurately from file size alone.
        // Do not pretend the entire file must be resident in system RAM.
        if gpu_layers == 0 {
            if let Some(available) = available_ram() {
                let available = available.saturating_add(freed_ram).saturating_sub(reserved);
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
    crowded_warning_in(state.inner(), replace).await
}

/// `crowded_load_warning`. A device the load would refuse (no CPU runtime, a
/// device that is not listed) is an error here, even with the warning hidden
/// or force fit on, so a reload stops before it ejects the instance it replaces.
pub async fn crowded_warning_in(state: &AppState, replace: Option<String>) -> Result<Option<Value>, String> {
    let mut settings = state.settings.lock().await.clone();
    let listed = crate::devices::listed_for(state, &settings).await;
    settings.device = crate::devices::choose_device(&settings, listed.as_deref())?;
    if settings.hide_crowded_load_warning || settings.allow_force_fit { return Ok(None); }
    let on_gpu = crate::devices::gpu_layers(&settings) != 0;
    let loaded = state.instances.lock().await.clone();
    // A reload unloads `replace` first, so it is neither another model nor
    // memory in use.
    let replaced = replace.as_deref().and_then(|id| loaded.iter().find(|i| i.id == id));
    let others: Vec<_> = loaded.iter().filter(|i| Some(i.id.as_str()) != replace.as_deref()).collect();
    if others.is_empty() { return Ok(None); }
    let (mut freed_gpu, mut freed_ram) = (None, 0);
    if let Some(instance) = replaced.filter(|instance| instance.is_ready()) {
        let pid = instance.child.lock().await.as_ref().and_then(Child::id);
        let estimate = crate::hardware::estimate_bytes(instance.path(), &instance_extras(&instance.settings), instance.settings.context_size).unwrap_or(0);
        if crate::devices::gpu_layers(&instance.settings) == 0 {
            freed_ram = pid.and_then(crate::oom::resident_bytes).unwrap_or(estimate);
        } else if crate::devices::shares_gpu(&instance.settings.device, &settings.device) {
            // Its graphics memory only helps a load on a GPU it is on.
            // Per-process readings exist on NVIDIA only; otherwise use the estimate.
            let measured = match pid {
                Some(pid) => tokio::task::spawn_blocking(move || crate::hardware::memory_use(&[pid])).await.ok().and_then(|u| u.processes.first().copied().flatten()),
                None => None,
            };
            freed_gpu = Some((&instance.settings, measured.unwrap_or(estimate)));
        }
    }
    // Ejecting a model on another GPU frees nothing on this one, so a GPU
    // load offers only the models sharing a GPU with it.
    let mut names = Vec::new();
    for instance in others.iter().filter(|i| !on_gpu || crate::devices::shares_gpu(&i.settings.device, &settings.device)) {
        let (name, _) = state.api_names(instance.path()).await;
        names.push(json!({"id": instance.id, "name": format!("{name}{}", instance.suffix())}));
    }
    let devices = listed.unwrap_or_default();
    Ok(load_memory_risk(state, &settings, others.into_iter(), (freed_gpu, freed_ram), &devices, false).await.map(|reason| json!({ "reason": reason, "models": names })))
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
/// It skips the file and runtime checks of `load_model_in` but builds the
/// instance the same way (`slot_and_label`, `new_instance`), then marks it as a
/// built-in decision model on the CPU, as a finished load would.
#[cfg(test)]
pub async fn insert_test_instance_on(state: &AppState, path: &str, ready: bool, port: u16) -> Arc<Instance> {
    let settings = Settings { model_path: path.into(), ..state.settings.lock().await.clone() };
    let loaded = state.instances.lock().await.clone();
    let (slot, label) = slot_and_label(&loaded, path, None);
    let device = if settings.device.is_empty() { crate::devices::CPU.to_owned() } else { settings.device.clone() };
    let instance = Arc::new(new_instance(settings, slot, label, RuntimeSession::new(&state.data_dir, port).unwrap(), device, false));
    instance.model.lock().unwrap().layout = Some("native".into());
    instance.ready.store(ready, Ordering::SeqCst);
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
        // Labels follow the slot and stay unique across different files
        // sharing a stem (the load test covers one file's instances).
        assert_eq!(instance_label("qwen", 1, &[]), "qwen");
        assert_eq!(instance_label("qwen", 2, &["qwen".into()]), "qwen#2");
        assert_eq!(instance_label("qwen", 1, &["qwen".into()]), "qwen#2");
        assert_eq!(instance_label("qwen", 2, &["qwen#2".into()]), "qwen#3");
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
        assert!(check_suffix(2, "-xyz", &long, &[], &[], &[]).unwrap_err().contains("up to 64 characters"));
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
    #[tokio::test]
    async fn loading_cpu_instance_reserves_ram_until_ready() {
        let base = std::env::temp_dir().join(format!("noullama-admission-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let model = base.join("model.gguf");
        std::fs::write(&model, b"model").unwrap();
        let settings = Settings {
            model_path: model.to_str().unwrap().into(),
            gpu_layers: 0,
            context_size: 65_536,
            ..Settings::default()
        };
        *state.settings.lock().await = settings.clone();
        let available_ram = || Some(2 * 1024 * 1024 * 1024);
        let no_gpu = || None;
        let _guard = state.lifecycle.lock().await;
        assert!(load_memory_risk_with(&settings, std::iter::empty(), (None, 0), available_ram, no_gpu).await.is_none());
        let first = insert_test_instance(&state, &settings.model_path, false).await;
        let loaded = state.instances.lock().await.clone();
        assert!(load_memory_risk_with(&settings, loaded.iter(), (None, 0), available_ram, no_gpu).await.is_some());
        let force = Settings { allow_force_fit: true, ..settings.clone() };
        assert!(load_memory_risk_with(&force, loaded.iter(), (None, 0), || panic!("force fit must skip RAM sampling"), || panic!("force fit must skip VRAM sampling")).await.is_none());
        first.ready.store(true, Ordering::SeqCst);
        assert!(load_memory_risk_with(&settings, loaded.iter(), (None, 0), available_ram, no_gpu).await.is_none());
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn file_stem_handles_utf8_and_case_insensitive_suffixes() {
        for (path, stem) in [("éabcd", "éabcd"), ("x.GGUF", "x"), ("é.gguf", "é"), (".gguf", "")] {
            assert_eq!(file_stem(path), stem);
        }
    }

    #[cfg(unix)]
    #[test]
    fn runtime_paths_resolve_to_the_one_llama_server_they_mean() {
        let base = std::env::temp_dir().join(format!("noullama-resolve-{}", uuid::Uuid::new_v4()));
        let touch = |path: &Path| {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"").unwrap();
        };
        // The managed CUDA layout: the server in package-0, the CUDA
        // runtime libraries in a separate package-1 folder.
        let build = base.join("runtimes/b1-cuda13");
        let server = build.join("package-0/llama-b1/llama-server");
        let cudart = build.join("package-1/cudart-llama-b1-bin-ubuntu-cuda-13.5-x64");
        touch(&server);
        touch(&cudart.join("libcudart.so.13"));
        touch(&build.join("noullama-install.json"));
        for path in [&server, &server.parent().unwrap().to_path_buf(), &cudart, &build.join("package-1"), &build.join("package-0"), &build] {
            assert_eq!(resolve_server(path.to_str().unwrap()).unwrap(), *server, "{}", path.display());
        }

        assert_eq!(resolve_server("  ").unwrap_err(), "No llama.cpp build is selected. Choose one in Runtime.");
        let gone = base.join("runtimes/b0-cuda13/package-0/llama-b0/llama-server");
        assert_eq!(
            resolve_server(gone.to_str().unwrap()).unwrap_err(),
            format!("The selected llama.cpp build is missing: {}. Choose an installed build in Runtime.", gone.display())
        );

        // Two servers in one folder are never guessed between.
        let two = base.join("two");
        touch(&two.join("a/llama-server"));
        touch(&two.join("b/llama-server"));
        assert!(resolve_server(two.to_str().unwrap()).unwrap_err().contains("contains several llama-server files"));

        // The search stays inside the folder: no symlinked folders, no hidden
        // folders, nothing deeper than the limit.
        let outside = base.join("outside");
        touch(&outside.join("llama-server"));
        let linked = base.join("linked");
        std::fs::create_dir_all(&linked).unwrap();
        std::os::unix::fs::symlink(&outside, linked.join("bin")).unwrap();
        touch(&linked.join(".hidden/llama-server"));
        touch(&linked.join("1/2/3/4/llama-server"));
        assert_eq!(
            resolve_server(linked.to_str().unwrap()).unwrap_err(),
            format!("No llama-server was found in {}. Choose llama-server or its folder in Runtime.", linked.display())
        );

        // The entry limit ends a search of a huge folder.
        let wide = base.join("wide");
        for n in 0..SERVER_SEARCH_ENTRIES + 10 {
            std::fs::create_dir_all(wide.join(format!("d{n:05}"))).unwrap();
        }
        touch(&wide.join(format!("d{:05}/llama-server", SERVER_SEARCH_ENTRIES + 5)));
        let started = std::time::Instant::now();
        let _ = resolve_server(wide.to_str().unwrap());
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(find_servers(&wide).len() <= 1);
        std::fs::remove_dir_all(base).unwrap();
    }

    #[tokio::test]
    async fn snapshot_shows_the_pre_load_sha256_check_and_clears_it() {
        let base = std::env::temp_dir().join(format!("noullama-verifying-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let model = base.join("m.gguf");
        // Three 4 MiB reads, each slowed down for the test.
        std::fs::write(&model, vec![0u8; 12 << 20]).unwrap();
        let path = model.to_str().unwrap().to_owned();
        crate::library::SLOW_HASH.lock().unwrap().insert(path.clone());
        state.settings.lock().await.sha256_check = "every_load".into();
        let instance = insert_test_instance(&state, &path, false).await;
        let check = tokio::spawn({
            let (state, instance) = (state.clone(), instance.clone());
            async move { verify_instance(&state, &instance).await }
        });
        let seen = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let snapshot = state_snapshot(&state).await.unwrap();
                let verifying = snapshot["instances"][0]["verifying"].clone();
                if !verifying.is_null() {
                    return verifying;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the snapshot never showed the check");
        assert_eq!(seen["slot"], "model");
        assert_eq!(seen["file"], path.as_str());
        assert_eq!(seen["total_bytes"], 12u64 << 20);
        check.await.unwrap().unwrap();
        let snapshot = state_snapshot(&state).await.unwrap();
        assert!(snapshot["instances"][0]["verifying"].is_null());
        assert!(!instance.verifying.load(Ordering::SeqCst));
        crate::library::SLOW_HASH.lock().unwrap().remove(&path);
        state.shutdown().await;
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn snapshot_returns_while_lifecycle_is_locked() {
        let base = std::env::temp_dir().join(format!("noullama-snapshot-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let _guard = state.lifecycle.lock().await;
        tokio::time::timeout(Duration::from_secs(2), state_snapshot(&state))
            .await.unwrap().unwrap();
        std::fs::remove_dir_all(base).unwrap();
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
        // The script is written by a child process. Had this process held it
        // open for writing, a child another test forks at that moment would
        // inherit the descriptor, and running the script would fail with
        // "Text file busy" until that child execs.
        std::fs::create_dir_all(base).unwrap();
        let draft = base.join(format!(".{name}.draft"));
        std::fs::write(&draft, script).unwrap();
        assert!(std::process::Command::new("cp").arg(&draft).arg(&path).status().unwrap().success());
        std::fs::remove_file(&draft).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn timed_out_probe_kills_the_child() {
        let base = std::env::temp_dir().join(format!("noullama-probe-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).unwrap();
        let script = fake_runtime(&base, "slow-probe", "#!/bin/sh\nprintf '%s' \"$$\" > \"$0.pid\"\nexec sleep 5\n");
        assert!(probe_output(&mut Command::new(script), Duration::from_millis(500)).await.is_err());
        let pid: u32 = std::fs::read_to_string(base.join("slow-probe.pid")).unwrap().parse().unwrap();
        let proc_dir = PathBuf::from(format!("/proc/{pid}"));
        tokio::time::timeout(Duration::from_secs(1), async {
            while proc_dir.exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }).await.expect("timed-out probe child still exists");
        std::fs::remove_dir_all(base).unwrap();
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
            // On the CPU, the fake runtime is the CPU runtime.
            settings.device = "CPU".into();
            settings.cpu_runtime_path = settings.runtime_path.clone();
        }
        let old = load_model_in(&state, None).await.unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        let snapshot = tokio::time::timeout(Duration::from_millis(500), state_snapshot(&state))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(snapshot["running"], true);
        let old = state.instance(&old).await.unwrap();
        let downloads = base.join("downloads").to_string_lossy().into_owned();
        tokio::time::timeout(
            Duration::from_millis(500),
            update_settings(&state, |s| { s.download_dir = downloads; Ok(()) }),
        )
        .await
        .unwrap()
        .unwrap();
        tokio::time::timeout(Duration::from_millis(500), state.shutdown())
            .await
            .unwrap();
        assert!(!is_registered(&state, &old).await);
        drop(old);
        let replacement = load_model_in(&state, None).await.unwrap();
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
        // Outside the ephemeral range, where parallel tests' outgoing
        // connections could take the port; random so other test processes
        // start elsewhere (as in `slow_loading_state`).
        let port = 26000 + (uuid::Uuid::new_v4().as_u128() % 6000) as u16;
        {
            let mut settings = state.settings.lock().await;
            settings.runtime_path = runtime.to_string_lossy().into_owned();
            settings.model_path = model.to_string_lossy().into_owned();
            settings.runtime_port = port;
            settings.gpu_layers = 0;
            // On the CPU, the fake runtime is the CPU runtime.
            settings.device = "CPU".into();
            settings.cpu_runtime_path = settings.runtime_path.clone();
        }
        let id = load_model_in(&state, None).await.unwrap();
        let instance = state.instance(&id).await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while instance.child.lock().await.is_none() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        // The load may have skipped a taken port. Its reservation is released
        // when the runtime starts, but a child a parallel test forks can hold a
        // copy of it until that child execs, so retry for a moment.
        let port = instance.session.port;
        let listener = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)).await {
                    Ok(listener) => break listener,
                    Err(_) => tokio::time::sleep(Duration::from_millis(20)).await,
                }
            }
        })
        .await
        .expect("the runtime port is free again");
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
    /// An app whose loads go through `load_model_in` with a runtime that never
    /// becomes ready, for `qwen.gguf` in its data directory.
    #[cfg(unix)]
    async fn slow_loading_state(name: &str) -> (AppState, PathBuf) {
        let base = std::env::temp_dir().join(format!("noullama-{name}-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let runtime = fake_runtime(&base, "slow-runtime", "#!/bin/sh\nexec sleep 60\n");
        let model = base.join("qwen.gguf");
        std::fs::write(&model, "test").unwrap();
        // Each load reserves its private port again after `free_runtime_port`
        // checked it. Outside the ephemeral range (32768 and up), parallel
        // tests' outgoing connections cannot take it in between; 26000–31999
        // is kept apart from the API port tests' 20000–25999.
        let port = 26000 + (uuid::Uuid::new_v4().as_u128() % 6000) as u16;
        {
            let mut settings = state.settings.lock().await;
            settings.runtime_path = runtime.to_string_lossy().into_owned();
            settings.model_path = model.to_string_lossy().into_owned();
            settings.runtime_port = port;
            settings.gpu_layers = 0;
            // On the CPU, the fake runtime is the CPU runtime.
            settings.device = "CPU".into();
            settings.cpu_runtime_path = settings.runtime_path.clone();
        }
        (state, base)
    }

    /// The Load button (`slot` none) and Reload (unload, then `slot` of the
    /// replaced instance) through the real load path.
    #[cfg(unix)]
    #[tokio::test]
    async fn loads_take_the_lowest_free_slot_and_a_reload_keeps_its_own() {
        let (state, base) = slow_loading_state("slots").await;
        let load = |slot: Option<u32>| {
            let state = state.clone();
            async move {
                let id = load_model_in(&state, slot).await.unwrap();
                state.instance(&id).await.unwrap()
            }
        };
        // A load that fails is unloaded at once and changes the free slots; say why.
        let load = |slot: Option<u32>| {
            let load = load(slot);
            let state = state.clone();
            async move {
                let instance = load.await;
                tokio::time::sleep(Duration::from_millis(50)).await;
                let errors: Vec<String> = state.load_errors.lock().await.iter().map(|e| e.message.clone()).collect();
                assert!(is_registered(&state, &instance).await, "the load failed: {errors:?}");
                instance
            }
        };
        let reaches = |name: &'static str| {
            let state = state.clone();
            async move { state.routes().await.into_iter().find(|r| r.name == name).and_then(|r| r.instance) }
        };
        let first = load(None).await;
        let second = load(None).await;
        let third = load(None).await;
        assert_eq!([first.slot, second.slot, third.slot], [1, 2, 3]);
        assert_eq!([first.suffix(), second.suffix(), third.suffix()], ["", "-i-2", "-i-3"]);
        assert_eq!([first.label.as_str(), second.label.as_str(), third.label.as_str()], ["qwen", "qwen#2", "qwen#3"]);
        assert_eq!(reaches("qwen-i-3").await, Some(third.id.clone()));
        // Instance 2 is ejected, then instance 3 reloaded: it stays instance 3.
        unload(&state, &second.id).await.unwrap();
        unload(&state, &third.id).await.unwrap();
        let reloaded = load(Some(3)).await;
        assert_eq!((reloaded.slot, reloaded.suffix().as_str()), (3, "-i-3"));
        assert_eq!(reaches("qwen-i-3").await, Some(reloaded.id.clone()));
        assert!(matches!(state.route(Some("qwen-i-3"), &Access::default()).await, Err(RouteError::Unavailable(m)) if m == "qwen instance 3 is still loading."));
        // A fresh load fills the gap.
        let fresh = load(None).await;
        assert_eq!((fresh.slot, fresh.suffix().as_str()), (2, "-i-2"));
        // A reload whose slot was taken meanwhile gets the lowest free one.
        let moved = load(Some(1)).await;
        assert_eq!((moved.slot, moved.suffix().as_str(), moved.label.as_str()), (4, "-i-4", "qwen#4"));
        state.shutdown().await;
        remove_once_released(&base).await;
    }

    /// A load starts its Keep context switch from the setting at that moment;
    /// changing the setting later leaves loaded instances alone.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_load_starts_keep_context_from_the_setting_and_keeps_its_own_switch() {
        let (state, base) = slow_loading_state("keep").await;
        let off = state.instance(&load_model_in(&state, None).await.unwrap()).await.unwrap();
        state.settings.lock().await.keep_context = true;
        let on = state.instance(&load_model_in(&state, None).await.unwrap()).await.unwrap();
        assert!(!off.keeps_context(), "the setting changed after this load");
        assert!(on.keeps_context());
        state.settings.lock().await.keep_context = false;
        let again = state.instance(&load_model_in(&state, None).await.unwrap()).await.unwrap();
        assert!(!again.keeps_context());
        assert!(on.keeps_context());
        state.shutdown().await;
        remove_once_released(&base).await;
    }

    /// Removes a test's data directory. Stopped loads delete their runtime key
    /// files from their own tasks, so the directory can still be emptying.
    async fn remove_once_released(base: &Path) {
        for _ in 0..100 {
            if std::fs::remove_dir_all(base).is_ok() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
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
            settings.cpu_runtime_path = settings.runtime_path.clone();
            settings.model_path = model;
            settings.runtime_port = port;
            settings.mtp = std::env::var("NOULLAMA_MTP_HEAD").is_ok();
            settings.mtp_path = std::env::var("NOULLAMA_MTP_HEAD").unwrap_or_default();
        }
        let id = load_model_in(&state, None).await.unwrap();
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

    /// An instance alias change holds the lifecycle lock from its collision
    /// check to its write; a models.yml save that adds the same name while it
    /// runs must see it and refuse, not slip past with an older list.
    #[tokio::test]
    async fn models_yml_saves_wait_for_an_instance_alias_change_in_progress() {
        let base = std::env::temp_dir().join(format!("noullama-alias-race-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let file = "/models/acme/qwen.gguf";
        let instance = insert_test_instance(&state, file, true).await;
        set_aliases(&state, file.into(), vec![]).await.unwrap();
        let wait = || tokio::time::sleep(Duration::from_millis(50));
        // The Library tab's save.
        let revision = state.library.lock().await.revision;
        let mut entries = state.library.lock().await.entries.clone();
        entries[0].aliases = vec!["billing".into()];
        let guard = state.lifecycle.lock().await;
        let save = tokio::spawn({
            let state = state.clone();
            async move { crate::library::save(&state, entries, revision).await }
        });
        wait().await;
        instance.set_aliases(vec!["billing".into()]);
        drop(guard);
        assert!(save.await.unwrap().unwrap_err().starts_with("\"billing\" is an instance alias of instance 1 of qwen"));
        // The API names editor's save.
        let guard = state.lifecycle.lock().await;
        let save = tokio::spawn({
            let state = state.clone();
            async move { set_aliases(&state, file.into(), vec!["refunds".into()]).await }
        });
        wait().await;
        instance.set_aliases(vec!["billing".into(), "refunds".into()]);
        drop(guard);
        assert!(save.await.unwrap().unwrap_err().starts_with("\"refunds\" is an instance alias of instance 1 of qwen"));
        assert!(state.library.lock().await.entries[0].aliases.is_empty(), "neither save was written");
        std::fs::remove_dir_all(base).unwrap();
    }

    /// A settings form's changed fields, as `save_settings` receives them.
    fn form(value: Value) -> serde_json::Map<String, Value> {
        value.as_object().expect("an object").clone()
    }

    /// Every command that changes settings, at once and each on its own
    /// field: none may write back another's old value.
    #[tokio::test]
    async fn concurrent_settings_changes_all_survive() {
        let base = std::env::temp_dir().join(format!("noullama-concurrent-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let downloads = base.join("models").to_string_lossy().into_owned();
        let results = tokio::join!(
            set_when_busy(&state, "wait"),
            set_wait_ms(&state, 5_000),
            change_fall_through(&state, true),
            set_memory_pref(&state, "allow_force_fit", true),
            set_memory_pref(&state, "hide_crowded_load_warning", true),
            crate::library::set_check(&state, "every_load".into()),
            // The load panel, the Runtime tab and the download directory.
            save_from_editor(&state, form(json!({"context_size": 4096, "keep_context": true}))),
            save_from_editor(&state, form(json!({"hf_path": "/opt/hf", "runtime_port": 12346}))),
            save_from_editor(&state, form(json!({"download_dir": downloads}))),
            crate::api::save_api_port(&state, 12345),
        );
        let (a, b, c, d, e, f, g, h, i, j) = results;
        for result in [a, b, c, d, e, f, g, h, i, j] {
            result.unwrap();
        }
        let check = |s: &Settings| {
            assert_eq!(s.any_available_when_busy, "wait");
            assert_eq!(s.any_available_wait_ms, 5_000);
            assert!(s.fall_through_to_other_instances && s.allow_force_fit && s.hide_crowded_load_warning && s.keep_context);
            assert_eq!(s.sha256_check, "every_load");
            assert_eq!((s.context_size, s.hf_path.as_str(), s.runtime_port, s.api_port), (4096, "/opt/hf", 12346, 12345));
            assert_eq!(s.download_dir, downloads);
        };
        check(&state.settings.lock().await.clone());
        check(&AppState::new(base.clone()).unwrap().settings.lock().await.clone());
        assert_eq!(save_from_editor(&state, form(json!({"nope": 1}))).await.unwrap_err(), "Unknown setting \"nope\".");
        assert!(save_from_editor(&state, form(json!({"context_size": "big"}))).await.unwrap_err().starts_with("Invalid settings"));
        std::fs::remove_dir_all(base).unwrap();
    }

    #[tokio::test]
    async fn memory_preferences_are_independent_persisted_and_survive_stale_forms() {
        let old: Settings = serde_json::from_str("{}").unwrap();
        assert!(!old.allow_force_fit && !old.hide_crowded_load_warning);
        let base = std::env::temp_dir().join(format!("noullama-memory-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        set_memory_pref(&state, "hide_crowded_load_warning", true).await.unwrap();
        assert!(!state.settings.lock().await.allow_force_fit);
        set_memory_pref(&state, "allow_force_fit", true).await.unwrap();
        assert_eq!(set_memory_pref(&state, "other", true).await.unwrap_err(), "Unknown memory preference");
        // A form that sends every field, with older copies of these two.
        save_from_editor(&state, form(serde_json::to_value(old).unwrap())).await.unwrap();
        let loaded = AppState::new(base.clone()).unwrap();
        let settings = loaded.settings.lock().await;
        assert!(settings.allow_force_fit && settings.hide_crowded_load_warning);
        drop(settings);
        assert!(load_memory_risk(&loaded, &loaded.settings.lock().await.clone(), std::iter::empty(), (None, 0), &[], false).await.is_none());
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
        update_settings(&state, |s| { *s = saved.clone(); Ok(()) }).await.unwrap();
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
        assert_eq!(save_from_editor(&state, form(json!({"download_dir": settings.download_dir}))).await.unwrap_err(), "Choose an absolute path for the download directory.");
        assert!(!base.join("settings.json").exists(), "nothing is saved");
        settings.download_dir = base.join("custom models").to_string_lossy().into_owned();
        save_from_editor(&state, form(json!({"download_dir": settings.download_dir}))).await.unwrap();
        assert!(Path::new(&settings.download_dir).is_dir());
        let stored: Settings =
            serde_json::from_slice(&std::fs::read(base.join("settings.json")).unwrap()).unwrap();
        assert_eq!(stored.download_dir, settings.download_dir);
        std::fs::remove_dir_all(base).unwrap();
    }

    #[tokio::test]
    async fn routing_settings_are_checked_saved_and_kept_from_stale_forms() {
        let base = std::env::temp_dir().join(format!("noullama-routing-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let stale = state.settings.lock().await.clone();
        assert_eq!(stale.any_available_wait_ms, 120_000);
        for out_of_range in [0, 600_001] {
            assert_eq!(set_wait_ms(&state, out_of_range).await.unwrap_err(), "Use a wait of 1–600000 ms.");
        }
        assert_eq!(state.settings.lock().await.any_available_wait_ms, 120_000);
        set_wait_ms(&state, 600_000).await.unwrap();
        set_when_busy(&state, "wait").await.unwrap();
        change_fall_through(&state, true).await.unwrap();
        // The load panel saves its form, whose copies of these are older.
        save_from_editor(&state, form(serde_json::to_value(Settings { context_size: 4096, ..stale }).unwrap())).await.unwrap();
        let reread = |base: PathBuf| async move { AppState::new(base).unwrap().settings.lock().await.clone() };
        let saved = reread(base.clone()).await;
        assert_eq!((saved.context_size, saved.any_available_wait_ms, saved.any_available_when_busy.as_str(), saved.fall_through_to_other_instances), (4096, 600_000, "wait", true));
        set_wait_ms(&state, 1).await.unwrap();
        assert_eq!(reread(base.clone()).await.any_available_wait_ms, 1);
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn waits_read_as_seconds_when_whole() {
        let texts: Vec<String> = [1, 999, 1000, 1500, 60_000, 120_000, 600_000].map(wait_text).into();
        assert_eq!(texts, ["1 ms", "999 ms", "1 second", "1500 ms", "60 seconds", "120 seconds", "600 seconds"]);
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

    /// `--list-devices` stdout of a CUDA build with one GPU, as a fake GPU
    /// runtime prints it.
    #[cfg(unix)]
    const GPU_LIST: &str = "Available devices:\n  CUDA0: Example GPU (8192 MiB, 7168 MiB free)\n";

    /// A fake runtime that lists `devices` and, when launched, writes its
    /// arguments beside itself and waits.
    #[cfg(unix)]
    fn listing_runtime(base: &Path, name: &str, devices: &str) -> PathBuf {
        fake_runtime(base, name, &format!("#!/bin/sh\nif [ \"$1\" = \"--list-devices\" ]; then printf '{}'; exit 0; fi\necho \"$@\" > \"$0.args\"\nexec sleep 60\n", devices.replace('\n', "\\n")))
    }

    /// What the fake runtime was launched with; fails with the load's errors
    /// and log if it never was.
    #[cfg(unix)]
    async fn launched_args(state: &AppState, runtime: &Path) -> Vec<String> {
        let file = PathBuf::from(format!("{}.args", runtime.display()));
        let found = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let Ok(text) = std::fs::read_to_string(&file) {
                    if text.ends_with('\n') {
                        return text.split_whitespace().map(str::to_owned).collect::<Vec<_>>();
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        match found {
            Ok(args) => args,
            Err(_) => {
                let errors: Vec<String> = state.load_errors.lock().await.iter().map(|e| e.message.clone()).collect();
                let logs = state.logs.lock().await.clone();
                panic!("{} was not launched: {errors:#?}\n{logs:#?}", runtime.display());
            }
        }
    }

    #[cfg(unix)]
    fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
        args.iter().position(|a| a == name).map(|i| args[i + 1].as_str())
    }

    /// The command a load starts, built for a GPU device and for the CPU.
    #[cfg(unix)]
    #[test]
    fn launches_run_on_their_one_device_with_its_runtime() {
        let base = std::env::temp_dir().join(format!("noullama-launch-{}", uuid::Uuid::new_v4()));
        for dir in ["gpu", "cpu"] {
            std::fs::create_dir_all(base.join(dir)).unwrap();
            std::fs::write(base.join(dir).join("llama-server"), "").unwrap();
        }
        let session = RuntimeSession::new(&base, 18442).unwrap();
        let settings = Settings {
            model_path: "/m/qwen.gguf".into(),
            runtime_path: base.join("gpu").to_string_lossy().into_owned(),
            cpu_runtime_path: base.join("cpu").to_string_lossy().into_owned(),
            library_path: "/opt/extra".into(),
            gpu_layers: 99,
            mtp: true,
            mtp_path: "/m/mtp-head.gguf".into(),
            ..Settings::default()
        };
        let args = |command: &Command| command.as_std().get_args().map(|a| a.to_string_lossy().into_owned()).collect::<Vec<_>>();
        let gpu = launch_command(&Settings { device: "CUDA1".into(), ..settings.clone() }, &session);
        assert_eq!(gpu.as_std().get_program(), base.join("gpu/llama-server").as_os_str());
        let gpu_args = args(&gpu);
        assert_eq!(gpu_args[gpu_args.len() - 6..], ["--device", "CUDA1", "--split-mode", "none", "--main-gpu", "0"]);
        assert_eq!(flag(&gpu_args, "--n-gpu-layers"), Some("99"));
        assert_eq!(flag(&gpu_args, "--spec-draft-ngl"), Some("99"));

        let cpu = launch_command(&Settings { device: "CPU".into(), ..settings.clone() }, &session);
        assert_eq!(cpu.as_std().get_program(), base.join("cpu/llama-server").as_os_str());
        let cpu_args = args(&cpu);
        assert_eq!(flag(&cpu_args, "--n-gpu-layers"), Some("0"));
        assert_eq!(flag(&cpu_args, "--spec-draft-ngl"), Some("0"));
        assert_eq!(cpu_args[cpu_args.len() - 2..], ["--device", "none"]);
        assert!(!cpu_args.contains(&"--split-mode".to_owned()));
        let library = cpu.as_std().get_envs().find(|(k, _)| *k == "LD_LIBRARY_PATH").and_then(|(_, v)| v).unwrap();
        let library: Vec<PathBuf> = std::env::split_paths(library).collect();
        assert_eq!(library[..2], [base.join("cpu"), PathBuf::from("/opt/extra")]);
        assert!(!library.contains(&base.join("gpu")));
        drop(session);
        std::fs::remove_dir_all(base).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_cpu_load_needs_a_cpu_runtime_and_a_saved_gpu_must_be_listed() {
        let base = std::env::temp_dir().join(format!("noullama-device-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let gpu = listing_runtime(&base, "gpu-runtime", GPU_LIST);
        let model = base.join("model.gguf");
        std::fs::write(&model, "test").unwrap();
        {
            let mut settings = state.settings.lock().await;
            settings.runtime_path = gpu.to_string_lossy().into_owned();
            settings.model_path = model.to_string_lossy().into_owned();
            settings.device = "CPU".into();
        }
        assert_eq!(load_model_in(&state, None).await.unwrap_err(), "Device CPU needs a CPU runtime. Choose or download one under CPU runtime in Runtime.");
        state.settings.lock().await.device = "CUDA1".into();
        assert_eq!(load_model_in(&state, None).await.unwrap_err(), "Device CUDA1 was not found: the llama.cpp build in Runtime does not report it. Choose another device in Loading.");
        // A GPU runtime that cannot list its devices finds none of them either.
        let broken = fake_runtime(&base, "broken-runtime", "#!/bin/sh\nexit 1\n");
        {
            let mut settings = state.settings.lock().await;
            settings.runtime_path = broken.to_string_lossy().into_owned();
            settings.device = "CUDA0".into();
        }
        assert_eq!(load_model_in(&state, None).await.unwrap_err(), crate::devices::not_found("CUDA0"));
        assert!(state.instances.lock().await.is_empty());
        state.shutdown().await;
        remove_once_released(&base).await;
    }

    /// An empty device loads on the first GPU listed, on the GPU runtime;
    /// "CPU" loads on the CPU runtime with no GPU layers. Each instance keeps its device.
    #[cfg(unix)]
    #[tokio::test]
    async fn loads_launch_on_the_chosen_device_and_keep_it() {
        let base = std::env::temp_dir().join(format!("noullama-placed-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let gpu = listing_runtime(&base, "gpu-runtime", GPU_LIST);
        let cpu = listing_runtime(&base, "cpu-runtime", "Available devices:\n  (none)\n");
        let model = base.join("qwen.gguf");
        std::fs::write(&model, "test").unwrap();
        {
            let mut settings = state.settings.lock().await;
            settings.runtime_path = gpu.to_string_lossy().into_owned();
            settings.cpu_runtime_path = cpu.to_string_lossy().into_owned();
            settings.model_path = model.to_string_lossy().into_owned();
            settings.runtime_port = 26000 + (uuid::Uuid::new_v4().as_u128() % 6000) as u16;
            // Memory readings are not part of this test.
            settings.allow_force_fit = true;
        }
        let first = state.instance(&load_model_in(&state, None).await.unwrap()).await.unwrap();
        assert_eq!(first.settings.device, "CUDA0");
        assert_eq!(first.model().device, "CUDA0 · Example GPU");
        let args = launched_args(&state, &gpu).await;
        assert_eq!(flag(&args, "--device"), Some("CUDA0"));
        assert_eq!(flag(&args, "--split-mode"), Some("none"));
        assert_eq!(flag(&args, "--n-gpu-layers"), Some("99"));

        state.settings.lock().await.device = "CPU".into();
        let second = state.instance(&load_model_in(&state, None).await.unwrap()).await.unwrap();
        assert_eq!((second.settings.device.as_str(), second.model().device.as_str()), ("CPU", "CPU"));
        assert_eq!(second.model().runtime_path, cpu.to_string_lossy());
        let args = launched_args(&state, &cpu).await;
        assert_eq!(flag(&args, "--device"), Some("none"));
        assert_eq!(flag(&args, "--n-gpu-layers"), Some("0"));
        // The snapshot reports what each launched with.
        let snapshot = state_snapshot(&state).await.unwrap();
        let shown: Vec<(String, i64)> = snapshot["instances"].as_array().unwrap().iter()
            .map(|i| (i["settings"]["device"].as_str().unwrap().to_owned(), i["gpu_layers"].as_i64().unwrap())).collect();
        assert_eq!(shown, [("CUDA0".to_owned(), 99), ("CPU".to_owned(), 0)]);
        assert_eq!(first.settings.device, "CUDA0", "a later load does not move an instance");
        state.shutdown().await;
        drop((first, second));
        remove_once_released(&base).await;
    }

    /// A still-loading instance holds graphics memory on its own GPU only.
    #[tokio::test]
    async fn loading_instances_reserve_vram_only_on_their_own_gpu() {
        let base = std::env::temp_dir().join(format!("noullama-per-gpu-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let model = base.join("model.gguf");
        std::fs::write(&model, b"model").unwrap();
        // About 1.5 GiB estimated, against a 2 GiB card with nothing on it.
        let on = |device: &str| Settings { model_path: model.to_string_lossy().into_owned(), gpu_layers: 99, context_size: 65_536, device: device.into(), ..Settings::default() };
        let card = || Some(vec![("CUDA0".to_owned(), 2u64 << 30, 0u64)]);
        let no_ram = || panic!("a full GPU load reads no RAM");
        *state.settings.lock().await = on("CUDA0");
        let loading = insert_test_instance(&state, &on("CUDA0").model_path, false).await;
        let loaded = state.instances.lock().await.clone();
        assert!(load_memory_risk_with(&on("CUDA0"), std::iter::empty(), (None, 0), no_ram, card).await.is_none());
        let same = load_memory_risk_with(&on("CUDA0"), loaded.iter(), (None, 0), no_ram, card).await;
        assert!(same.is_some_and(|reason| reason.contains("available VRAM")));
        assert!(load_memory_risk_with(&on("CUDA1"), loaded.iter(), (None, 0), no_ram, card).await.is_none());
        // Once ready it is in the live reading instead.
        set_test_ready(&loading, true);
        assert!(load_memory_risk_with(&on("CUDA0"), loaded.iter(), (None, 0), no_ram, card).await.is_none());
        drop((loading, loaded));
        state.shutdown().await;
        remove_once_released(&base).await;
    }

    #[test]
    fn settings_from_before_devices_still_read() {
        let base = std::env::temp_dir().join(format!("noullama-old-settings-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("settings.json"), r#"{"runtime_path":"/opt/llama/bin","gpu_layers":0,"library_path":"/usr/local/cuda/lib64"}"#).unwrap();
        let state = AppState::new(base.clone()).unwrap();
        assert!(state.startup_warning.is_none());
        let settings = state.settings.try_lock().unwrap().clone();
        assert_eq!((settings.device.as_str(), settings.cpu_runtime_path.as_str()), ("", ""));
        assert_eq!((settings.runtime_path.as_str(), settings.gpu_layers), ("/opt/llama/bin", 0));
        // Settings from before splitting get llama.cpp's own split defaults.
        assert_eq!((settings.split_mode.as_str(), settings.tensor_split.as_str()), ("layer", ""));
        std::fs::remove_dir_all(base).unwrap();
    }

    /// A reload asks for the crowded-load warning before it ejects the
    /// instance it replaces; a device the load would refuse stops it there.
    #[tokio::test]
    async fn reload_with_a_missing_runtime_keeps_the_ready_instance() {
        let dir = std::env::temp_dir().join(format!("noullama-test-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(dir.clone()).unwrap();
        let path = dir.join("weights.gguf").to_str().unwrap().to_owned();
        std::fs::write(&path, "example weights").unwrap();
        let previous = insert_test_instance(&state, &path, true).await;
        {
            let mut settings = state.settings.lock().await;
            settings.model_path = path;
            settings.device = crate::devices::CPU.into();
            settings.cpu_runtime_path = dir.join("missing").to_string_lossy().into_owned();
        }
        let error = load_model_replacing(&state, None, Some(&previous.id)).await.unwrap_err();
        assert!(error.contains("selected llama.cpp build is missing"));
        assert!(state.instance(&previous.id).await.unwrap().is_ready());
        assert_eq!(state.instances.lock().await.len(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn the_crowded_check_refuses_a_device_before_a_reload_ejects_anything() {
        let base = std::env::temp_dir().join(format!("noullama-crowded-device-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let model = base.join("model.gguf");
        std::fs::write(&model, b"model").unwrap();
        let replaced = insert_test_instance(&state, model.to_str().unwrap(), true).await;
        {
            let mut settings = state.settings.lock().await;
            settings.model_path = model.to_string_lossy().into_owned();
            settings.device = "CPU".into();
            settings.hide_crowded_load_warning = true;
        }
        assert_eq!(crowded_warning_in(&state, Some(replaced.id.clone())).await.unwrap_err(), crate::devices::CPU_RUNTIME_MISSING);
        state.settings.lock().await.device = "CUDA3".into();
        assert_eq!(crowded_warning_in(&state, Some(replaced.id.clone())).await.unwrap_err(), crate::devices::not_found("CUDA3"));
        {
            let mut settings = state.settings.lock().await;
            settings.device = "CPU".into();
            settings.cpu_runtime_path = "/cpu/llama-server".into();
        }
        assert!(crowded_warning_in(&state, Some(replaced.id.clone())).await.unwrap().is_none());
        drop(replaced);
        state.shutdown().await;
        remove_once_released(&base).await;
    }

    const MIB: u64 = 1024 * 1024;

    fn machine(ram_mib: u64, vram_mib: Option<u64>) -> crate::hardware::Hardware {
        crate::hardware::Hardware { ram_bytes: ram_mib * MIB, vram_bytes: vram_mib.map(|v| v * MIB), gpu: None, cpu_cores: 8 }
    }

    /// A 1 MiB model file and a state whose loads run on the CPU runtime.
    async fn fit_state(name: &str) -> (PathBuf, AppState, String) {
        let base = std::env::temp_dir().join(format!("noullama-fit-{name}-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let model = base.join("model.gguf");
        std::fs::write(&model, vec![0u8; MIB as usize]).unwrap();
        let path = model.to_string_lossy().into_owned();
        {
            let mut settings = state.settings.lock().await;
            settings.model_path = path.clone();
            settings.device = "CPU".into();
            settings.cpu_runtime_path = "/cpu/llama-server".into();
        }
        (base, state, path)
    }

    /// Reload replaces the instance picked in the load panel, so the fit
    /// leaves out that one, not the first instance of the file; a fresh load
    /// leaves out none.
    #[tokio::test]
    async fn the_fit_leaves_out_the_instance_the_reload_replaces() {
        let (base, state, path) = fit_state("replace").await;
        state.settings.lock().await.context_size = 131072;
        let first = insert_test_instance(&state, &path, true).await;
        state.settings.lock().await.context_size = 512;
        let second = insert_test_instance(&state, &path, true).await;
        assert_eq!((first.slot, second.slot), (1, 2));
        state.settings.lock().await.context_size = 4096;
        let candidate = state.settings.lock().await.clone();
        let instances = state.instances.lock().await.clone();
        // About 3 GiB for the first instance, a few MiB for the second.
        let hw = machine(3300, None);
        let fit = |replace: Option<&str>| fit_for_load(&candidate, None, &instances, replace, Some(&hw)).unwrap();
        let (reload_second, counted) = fit(Some(&second.id));
        assert_eq!(counted, 1);
        let (reload_first, _) = fit(Some(&first.id));
        assert_eq!(reload_first.level, 1, "{}", reload_first.detail);
        assert!(reload_second.level > reload_first.level, "the first instance's memory stays taken: {}", reload_second.detail);
        let (fresh, counted) = fit(None);
        assert_eq!(counted, 2);
        assert!(fresh.level >= reload_second.level);
        drop((first, second, instances));
        state.shutdown().await;
        remove_once_released(&base).await;
    }

    /// An empty device loads on the first GPU, else on the CPU, and the fit
    /// is rated for that device: system memory and no GPU layers on the CPU,
    /// that GPU's own memory on a GPU.
    #[tokio::test]
    async fn the_fit_rates_the_device_the_load_would_use() {
        let (base, state, _) = fit_state("device").await;
        {
            let mut settings = state.settings.lock().await;
            settings.device = String::new();
            settings.gpu_layers = 99;
            settings.context_size = 4096;
        }
        let candidate = state.settings.lock().await.clone();
        // ~97 MiB needed. A 100 MiB probe reading would rate it from system memory.
        let hw = machine(64 * 1024, Some(100));
        let (cpu, _) = fit_for_load(&candidate, Some(&[]), &[], None, Some(&hw)).unwrap();
        assert_eq!(cpu.level, 1, "{}", cpu.detail);
        assert!(cpu.detail.contains("system memory available"), "{}", cpu.detail);
        let gpus = crate::devices::parse_list_devices("Available devices:\n  CUDA0: Test GPU (200 MiB, 190 MiB free)\n  CUDA1: Other GPU (200 MiB, 190 MiB free)\n");
        let (gpu, _) = fit_for_load(&candidate, Some(&gpus), &[], None, Some(&hw)).unwrap();
        assert_eq!(gpu.level, 1, "{}", gpu.detail);
        assert!(gpu.detail.contains("of graphics memory available"), "{}", gpu.detail);
        // A model on the other GPU takes nothing from this one.
        state.settings.lock().await.device = "CUDA1".into();
        let elsewhere = insert_test_instance(&state, &candidate.model_path, true).await;
        let instances = state.instances.lock().await.clone();
        let (beside, counted) = fit_for_load(&candidate, Some(&gpus), &instances, None, Some(&hw)).unwrap();
        assert_eq!((counted, beside.detail), (0, gpu.detail));
        drop((elsewhere, instances));
        state.shutdown().await;
        remove_once_released(&base).await;
    }

    /// The badge follows the load panel's unsaved values, and says why when
    /// there is no estimate.
    #[tokio::test]
    async fn the_fit_uses_unsaved_load_panel_values_and_names_the_reason() {
        let (base, state, path) = fit_state("unsaved").await;
        state.settings.lock().await.context_size = 512;
        let hw = machine(64 * 1024, None);
        let changes = |pairs: Value| pairs.as_object().unwrap().clone();
        let saved = load_fit_in(&state, changes(json!({})), None, Some(&hw)).await.unwrap();
        let edited = load_fit_in(&state, changes(json!({"context_size": 65536})), None, Some(&hw)).await.unwrap();
        assert!(edited["fit"]["required_bytes"].as_u64() > saved["fit"]["required_bytes"].as_u64());
        assert_eq!(state.settings.lock().await.context_size, 512, "nothing is saved");
        let missing = base.join("missing.gguf").to_string_lossy().into_owned();
        assert_eq!(load_fit_in(&state, changes(json!({"model_path": missing})), None, Some(&hw)).await.unwrap_err(), crate::hardware::MODEL_UNREADABLE);
        assert_eq!(load_fit_in(&state, changes(json!({"device": "CUDA3"})), None, Some(&hw)).await.unwrap_err(), crate::devices::not_found("CUDA3"));
        state.settings.lock().await.cpu_runtime_path = String::new();
        assert_eq!(load_fit_in(&state, changes(json!({})), None, Some(&hw)).await.unwrap_err(), crate::devices::CPU_RUNTIME_MISSING);
        state.settings.lock().await.cpu_runtime_path = "/cpu/llama-server".into();
        assert!(load_fit_in(&state, changes(json!({})), None, None).await.unwrap_err().contains("Memory detection is still running"));
        assert!(load_fit_in(&state, changes(json!({"model_path": path, "bogus": 1})), None, Some(&hw)).await.unwrap_err().contains("Unknown setting"));
        state.shutdown().await;
        remove_once_released(&base).await;
    }

    /// `--list-devices` stdout of a CUDA build with two made-up GPUs.
    const TWO_GPU_LIST: &str = "Available devices:\n  CUDA0: Example GPU A (10240 MiB, 9000 MiB free)\n  CUDA1: Example GPU B (24576 MiB, 23000 MiB free)\n";

    /// A split loads on the GPUs listed at load time: All GPUs on each of
    /// them, a chosen set in listed order with its mode and ratio. The
    /// instance keeps the GPUs and shows them with their summed memory.
    #[cfg(unix)]
    #[tokio::test]
    async fn splits_launch_across_the_chosen_gpus() {
        let base = std::env::temp_dir().join(format!("noullama-split-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let gpu = listing_runtime(&base, "gpu-runtime", TWO_GPU_LIST);
        let model = base.join("qwen.gguf");
        std::fs::write(&model, "test").unwrap();
        {
            let mut settings = state.settings.lock().await;
            settings.runtime_path = gpu.to_string_lossy().into_owned();
            settings.model_path = model.to_string_lossy().into_owned();
            settings.runtime_port = 26000 + (uuid::Uuid::new_v4().as_u128() % 6000) as u16;
            settings.allow_force_fit = true;
            settings.device = crate::devices::ALL_GPUS.into();
        }
        let all = state.instance(&load_model_in(&state, None).await.unwrap()).await.unwrap();
        assert_eq!(all.settings.device, "CUDA0,CUDA1");
        assert_eq!(all.model().device, "All GPUs (2) · 34 GB");
        let args = launched_args(&state, &gpu).await;
        assert_eq!(flag(&args, "--device"), Some("CUDA0,CUDA1"));
        assert_eq!(flag(&args, "--split-mode"), Some("layer"));
        assert!(!args.contains(&"--main-gpu".to_owned()) && !args.contains(&"--tensor-split".to_owned()), "{args:?}");
        std::fs::remove_file(format!("{}.args", gpu.display())).unwrap();

        {
            let mut settings = state.settings.lock().await;
            settings.device = "CUDA1,CUDA0".into();
            settings.split_mode = "row".into();
            settings.tensor_split = "3,1".into();
        }
        let chosen = state.instance(&load_model_in(&state, None).await.unwrap()).await.unwrap();
        assert_eq!(chosen.settings.device, "CUDA0,CUDA1");
        assert_eq!(chosen.model().device, "CUDA0 + CUDA1 · 34 GB");
        let args = launched_args(&state, &gpu).await;
        assert_eq!(args[args.len() - 6..], ["--device", "CUDA0,CUDA1", "--split-mode", "row", "--tensor-split", "3,1"]);
        state.shutdown().await;
        drop((all, chosen));
        remove_once_released(&base).await;
    }

    /// A split the load cannot run as saved is refused before anything starts.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_split_that_does_not_match_the_listed_gpus_is_refused() {
        let base = std::env::temp_dir().join(format!("noullama-split-refused-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let gpu = listing_runtime(&base, "gpu-runtime", TWO_GPU_LIST);
        let model = base.join("model.gguf");
        std::fs::write(&model, "test").unwrap();
        {
            let mut settings = state.settings.lock().await;
            settings.runtime_path = gpu.to_string_lossy().into_owned();
            settings.model_path = model.to_string_lossy().into_owned();
            settings.device = "CUDA0,CUDA2".into();
        }
        assert_eq!(load_model_in(&state, None).await.unwrap_err(), crate::devices::not_found("CUDA2"));
        state.settings.lock().await.device = "CUDA1,".into();
        assert_eq!(load_model_in(&state, None).await.unwrap_err(), crate::devices::TOO_FEW_GPUS);
        {
            let mut settings = state.settings.lock().await;
            settings.device = crate::devices::ALL_GPUS.into();
            settings.tensor_split = "1,1,1".into();
        }
        assert_eq!(load_model_in(&state, None).await.unwrap_err(), crate::devices::ratio_count(2, 3));
        assert!(state.instances.lock().await.is_empty());
        // A save checks the mode and the numbers.
        assert_eq!(update_settings(&state, |s| { s.split_mode = "rows".into(); Ok(()) }).await.unwrap_err(), crate::devices::BAD_SPLIT_MODE);
        assert_eq!(update_settings(&state, |s| { s.tensor_split = "3;1".into(); Ok(()) }).await.unwrap_err(), crate::devices::BAD_RATIO);
        update_settings(&state, |s| { s.split_mode = "row".into(); s.tensor_split = "3, 1".into(); Ok(()) }).await.unwrap();
        update_settings(&state, |s| { s.device = "CUDA0".into(); s.tensor_split = "unfinished,".into(); Ok(()) }).await.unwrap();
        state.shutdown().await;
        remove_once_released(&base).await;
    }

    /// A split's fit counts the GPUs' summed memory, and a ratio that gives
    /// one GPU more than it holds rates it no better than Strained.
    #[tokio::test]
    async fn the_fit_of_a_split_sums_its_gpus_and_checks_each_share() {
        let (base, state, _) = fit_state("split").await;
        {
            let mut settings = state.settings.lock().await;
            settings.gpu_layers = 99;
            settings.context_size = 4096;
        }
        // ~97 MiB needed; each GPU offers 54 MiB of its 60.
        let gpus = crate::devices::parse_list_devices("Available devices:\n  CUDA0: Example GPU A (60 MiB, 50 MiB free)\n  CUDA1: Example GPU B (60 MiB, 50 MiB free)\n");
        let hw = machine(64 * 1024, Some(100));
        let on = |device: &str, ratio: &str| Settings { device: device.into(), tensor_split: ratio.into(), ..state.settings.try_lock().unwrap().clone() };
        let (one, _) = fit_for_load(&on("CUDA0", ""), Some(&gpus), &[], None, Some(&hw)).unwrap();
        assert_eq!(one.level, 3, "{}", one.detail);
        let (both, _) = fit_for_load(&on(crate::devices::ALL_GPUS, ""), Some(&gpus), &[], None, Some(&hw)).unwrap();
        assert_eq!(both.level, 2, "{}", both.detail);
        let (even, _) = fit_for_load(&on("CUDA0,CUDA1", "1,1"), Some(&gpus), &[], None, Some(&hw)).unwrap();
        assert_eq!(even.level, 2, "{}", even.detail);
        let (lopsided, _) = fit_for_load(&on("CUDA0,CUDA1", "1,0"), Some(&gpus), &[], None, Some(&hw)).unwrap();
        assert_eq!(lopsided.level, 3, "{}", lopsided.detail);
        assert!(lopsided.detail.contains("gives CUDA0 about"), "{}", lopsided.detail);
        // A GPU without a memory reading is named, not guessed.
        let unread = crate::devices::parse_list_devices("Available devices:\n  CUDA0: A (60 MiB, 50 MiB free)\n  CUDA1: B\n");
        assert_eq!(fit_for_load(&on(crate::devices::ALL_GPUS, ""), Some(&unread), &[], None, Some(&hw)).err().unwrap(), "CUDA1 did not report its memory, so there is no estimate for this split.");
        state.shutdown().await;
        remove_once_released(&base).await;
    }

    /// The VRAM guard sums a split's free memory and, with a ratio, checks
    /// each GPU's share. A loading split holds memory on each of its GPUs.
    #[tokio::test]
    async fn the_vram_guard_sums_a_split_and_checks_each_share() {
        let base = std::env::temp_dir().join(format!("noullama-split-guard-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(base.clone()).unwrap();
        let model = base.join("model.gguf");
        std::fs::write(&model, b"model").unwrap();
        // About 1.5 GiB estimated, against two cards with 0.75 GiB free each.
        let on = |device: &str, ratio: &str| Settings { model_path: model.to_string_lossy().into_owned(), gpu_layers: 99, context_size: 65_536, device: device.into(), tensor_split: ratio.into(), ..Settings::default() };
        let cards = || Some(vec![("CUDA0".to_owned(), 1u64 << 30, 1u64 << 28), ("CUDA1".to_owned(), 1u64 << 30, 1u64 << 28)]);
        let card = || Some(vec![("CUDA0".to_owned(), 1u64 << 30, 1u64 << 28)]);
        let no_ram = || panic!("a full GPU load reads no RAM");
        let none = std::iter::empty();
        assert!(load_memory_risk_with(&on("CUDA0", ""), none, (None, 0), no_ram, card).await.is_some_and(|r| r.contains("available VRAM.")));
        assert!(load_memory_risk_with(&on("CUDA0,CUDA1", ""), std::iter::empty(), (None, 0), no_ram, cards).await.is_none());
        assert!(load_memory_risk_with(&on("CUDA0,CUDA1", "1,1"), std::iter::empty(), (None, 0), no_ram, cards).await.is_none());
        let lopsided = load_memory_risk_with(&on("CUDA0,CUDA1", "1,0"), std::iter::empty(), (None, 0), no_ram, cards).await;
        assert!(lopsided.as_deref().is_some_and(|r| r.contains("CUDA0's share")), "{lopsided:?}");
        // An automatic split counts in aggregate when both of its GPUs are used.
        *state.settings.lock().await = on("CUDA0,CUDA1", "");
        let loading = insert_test_instance(&state, &on("CUDA0", "").model_path, false).await;
        let loaded = state.instances.lock().await.clone();
        let crowded = load_memory_risk_with(&on("CUDA0,CUDA1", ""), loaded.iter(), (None, 0), no_ram, cards).await;
        assert!(crowded.as_deref().is_some_and(|r| r.contains("across CUDA0 + CUDA1")), "{crowded:?}");
        let card1 = || Some(vec![("CUDA1".to_owned(), 2u64 << 30, 0u64)]);
        assert!(load_memory_risk_with(&on("CUDA1", ""), loaded.iter(), (None, 0), no_ram, card1).await.is_none());
        assert!(load_memory_risk_with(&on("CUDA2", ""), loaded.iter(), (None, 0), no_ram, card1).await.is_none());
        let previous = on("CUDA0,CUDA1", "1,3");
        let mut shares = std::collections::HashMap::new();
        assert_eq!(gpu_overlap_bytes(&previous, "CUDA0", 1024, &mut shares), 256);
        assert_eq!(shares, std::collections::HashMap::from([("CUDA0".into(), 256)]));
        let reloaded = load_memory_risk_with(&on("CUDA0,CUDA1", "0,1"), std::iter::empty(), (Some((&on("CUDA0", ""), 1 << 30)), 0), no_ram, cards).await;
        assert!(reloaded.as_deref().is_some_and(|reason| reason.contains("CUDA1's share")), "{reloaded:?}");
        drop((loading, loaded));
        // A GPU of the split without a reading skips the check and says which.
        let unread = crate::devices::parse_list_devices("Available devices:\n  CUDA0: Example GPU A (1024 MiB, 768 MiB free)\n  CUDA1: Example GPU B\n");
        state.instances.lock().await.clear();
        assert!(load_memory_risk(&state, &on("CUDA0,CUDA1", ""), std::iter::empty(), (None, 0), &unread, true).await.is_none());
        assert!(state.logs.lock().await.iter().any(|line| line.contains("VRAM preflight unavailable for device CUDA1.")));
        state.shutdown().await;
        remove_once_released(&base).await;
    }
}
