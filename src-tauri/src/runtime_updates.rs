//! Automatic llama.cpp updates.
//!
//! Only builds Noullama downloaded itself are updated: the folders in
//! `<data>/runtimes/<tag>-<backend>` with the `noullama-install.json` that
//! `install_build` writes, selected or not. A build outside that folder is
//! never touched.
//!
//! Each backend has two lines of builds, nightly and stable, and each moves
//! to its newest release on its own. A folder starts on the line its install
//! record names; when the other line's newest release has the same tag, the
//! folder is shared and belongs to both (kept in `runtime-updates.json`,
//! since the install record keeps the label of the first install). When a
//! line moves, its older folders leave it; a folder that is on no line any
//! more is marked superseded and removed once no loaded model runs from it
//! and no setting names it.
//!
//! The selected builds follow their own line: a setting moves to the newest
//! folder of the line it follows when its folder leaves that line. Every
//! check finishes such moves, so one that could not be saved is made later.

use crate::runtime::AppState;
use crate::runtimes::{Download, InstallProgress, ReleaseLines, RuntimeBuild};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
    sync::atomic::Ordering,
    time::{Duration, SystemTime},
};

/// How often the builds are checked: llama.cpp publishes several nightlies
/// a day, and each CUDA build is hundreds of megabytes.
const CHECK_EVERY: Duration = Duration::from_secs(24 * 60 * 60);
/// The first check waits for startup to settle.
const FIRST_CHECK: Duration = Duration::from_secs(10);
/// How often a due check and superseded folders are looked at.
const TICK: Duration = Duration::from_secs(60 * 60);
/// How soon a check that found the newest nightly still uploading runs again.
const UPLOAD_RETRY: Duration = Duration::from_secs(10 * 60);
const STATE_FILE: &str = "runtime-updates.json";
const INSTALL_FILE: &str = "noullama-install.json";
/// Written into a replaced build's folder; the folder is removed when unused.
const SUPERSEDED: &str = "noullama-superseded";
const SETTING_FIELDS: [&str; 2] = ["runtime_path", "cpu_runtime_path"];

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Channel {
    Nightly,
    Stable,
}

impl Channel {
    fn of(build: &RuntimeBuild) -> Self {
        if build.release == "Nightly" { Channel::Nightly } else { Channel::Stable }
    }
    fn name(self) -> &'static str {
        match self {
            Channel::Nightly => "nightly",
            Channel::Stable => "stable",
        }
    }
}

/// A setting the updater moved, and the line it follows while it still
/// names that path.
#[derive(Clone, Serialize, Deserialize)]
struct Follow {
    field: String,
    path: String,
    channel: Channel,
}

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Record {
    /// When the last complete check finished, in seconds since the Unix epoch.
    checked_at: Option<u64>,
    /// The lines of folders whose lines differ from their install record,
    /// by folder name. An empty list: the folder is on no line.
    lines: BTreeMap<String, Vec<Channel>>,
    follows: Vec<Follow>,
}

fn now() -> u64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or_default()
}

fn read_record(data_dir: &Path) -> Record {
    std::fs::read(data_dir.join(STATE_FILE)).ok().and_then(|bytes| serde_json::from_slice(&bytes).ok()).unwrap_or_default()
}

fn write_record(data_dir: &Path, record: &Record) {
    let temp = data_dir.join(format!("{STATE_FILE}.tmp"));
    if serde_json::to_vec_pretty(record).is_ok_and(|bytes| std::fs::write(&temp, bytes).is_ok()) {
        let _ = std::fs::rename(temp, data_dir.join(STATE_FILE));
    }
}

/// The update's progress and when the builds were last checked, for the snapshot.
pub async fn view(state: &AppState) -> Value {
    let progress = state.runtimes.update_progress.lock().await.clone();
    json!({ "progress": progress, "checked_at": read_record(&state.data_dir).checked_at })
}

/// A managed build: its folder under `<data>/runtimes` and what
/// `install_build` recorded about it.
struct Managed {
    folder: PathBuf,
    build: RuntimeBuild,
}

impl Managed {
    fn name(&self) -> String {
        folder_name(&self.folder)
    }
}

fn folder_name(folder: &Path) -> String {
    folder.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

fn runtimes_root(data_dir: &Path) -> PathBuf {
    data_dir.join("runtimes")
}

/// The managed build `selected` (a llama-server path or a folder) lies in, if any.
fn managed(root: &Path, selected: &str) -> Option<Managed> {
    let selected = selected.trim();
    if selected.is_empty() {
        return None;
    }
    let relative = Path::new(selected).strip_prefix(root).ok()?;
    let Some(Component::Normal(name)) = relative.components().next() else {
        return None;
    };
    if name.to_string_lossy().starts_with('.') || relative.components().any(|c| matches!(c, Component::ParentDir)) {
        return None;
    }
    let folder = root.join(name);
    let build = serde_json::from_slice(&std::fs::read(folder.join(INSTALL_FILE)).ok()?).ok()?;
    Some(Managed { folder, build })
}

/// Whether `path` lies in a managed build that an update replaced, so the
/// Runtime tab does not offer it.
pub(crate) fn is_superseded(root: &Path, path: &str) -> bool {
    managed(root, path).is_some_and(|m| m.folder.join(SUPERSEDED).exists())
}

/// Every build Noullama installed that no update has replaced yet.
fn installed_builds(root: &Path) -> Vec<Managed> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut builds: Vec<Managed> = entries
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()) && !e.path().join(SUPERSEDED).exists())
        .filter_map(|e| managed(root, &e.path().to_string_lossy()))
        .collect();
    builds.sort_by(|a, b| a.folder.cmp(&b.folder));
    builds
}

/// The lines a build is on: as recorded, else the one its install record names.
fn lines_of(record: &Record, build: &Managed) -> Vec<Channel> {
    record.lines.get(&build.name()).cloned().unwrap_or_else(|| vec![Channel::of(&build.build)])
}

/// A `b<number>` tag's number, for ordering builds; other tags sort first.
fn build_number(tag: &str) -> u64 {
    tag.strip_prefix('b').and_then(|n| n.parse().ok()).unwrap_or(0)
}

/// Whether `candidate` replaces `current`: a higher build number, or for
/// tags that are not build numbers (stable `vX.Y.Z`), a different tag.
fn is_newer(candidate: &str, current: &str) -> bool {
    let number = |tag: &str| tag.strip_prefix('b').and_then(|n| n.parse::<u64>().ok());
    match (number(candidate), number(current)) {
        (Some(new), Some(old)) => new > old,
        _ => candidate != current,
    }
}

/// The release a line whose newest build is `current` moves to, if it is
/// newer and for the same backend (a stable release without this backend
/// falls back to CPU; that is not an update).
fn successor<'a>(current: &RuntimeBuild, next: &'a RuntimeBuild) -> Option<&'a RuntimeBuild> {
    (next.backend == current.backend && is_newer(&next.tag, &current.tag)).then_some(next)
}

/// Starts the background checks while automatic updates are on: 10 seconds
/// after launch and then hourly, a check runs when the last complete one is
/// a day old. Superseded builds are removed at the same ticks once nothing
/// uses them, whether or not updates are on.
pub fn start(state: AppState) {
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(FIRST_CHECK).await;
        loop {
            if state.app.exiting.load(Ordering::SeqCst) {
                return;
            }
            let due = read_record(&state.data_dir).checked_at.is_none_or(|at| now().saturating_sub(at) >= CHECK_EVERY.as_secs());
            let mut waiting = false;
            if due && state.settings.lock().await.auto_update_runtime {
                // Busy with another runtime download: try again next tick.
                waiting = update(&state, true, |backend| {
                    let state = state.clone();
                    async move { crate::runtimes::release_lines(&state, &backend).await }
                })
                .await
                .unwrap_or(false);
            }
            sweep(&state).await;
            tokio::time::sleep(if waiting { UPLOAD_RETRY } else { TICK }).await;
        }
    });
}

#[tauri::command]
pub async fn set_auto_update_runtime(value: bool, state: tauri::State<'_, AppState>) -> Result<(), String> {
    crate::runtime::update_settings(state.inner(), |s| {
        s.auto_update_runtime = value;
        Ok(())
    })
    .await
}

/// "Check now": checks and updates the builds at once, also when automatic
/// updates are off. Returns when the check has finished.
#[tauri::command]
pub async fn update_runtime_now(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let state = state.inner().clone();
    let worker = state.clone();
    update(&state, false, move |backend| {
        let state = worker.clone();
        async move { crate::runtimes::release_lines(&state, &backend).await }
    })
    .await?;
    sweep(&state).await;
    Ok(())
}

/// What one check did.
#[derive(Default)]
struct Outcome {
    updated: Vec<String>,
    /// Lines whose newest nightly is still uploading, as its message.
    waiting: Vec<String>,
    errors: Vec<String>,
}

/// Checks the installed managed builds against `lines_for(backend)` and
/// installs any newer one. Fails only when another runtime download is
/// running; problems are reported in the update's progress. Returns true
/// when a newest nightly is still uploading, so the check is waiting.
/// `automatic`: run by the background checks, which retry such a wait.
async fn update<F, Fut>(state: &AppState, automatic: bool, lines_for: F) -> Result<bool, String>
where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = Result<ReleaseLines, String>>,
{
    // The update runs in this task, so the lifecycle lock is not held for it.
    drop(crate::runtimes::claim(state, Download::Update, "Checking for llama.cpp updates…").await?);
    let progress = &state.runtimes.update_progress;
    let outcome = check(state, &lines_for, progress).await;
    let waiting = !outcome.waiting.is_empty();
    // An incomplete check (offline, rate limited, still uploading) runs again
    // at the next tick instead of a day later.
    if outcome.errors.is_empty() && !waiting {
        let mut record = read_record(&state.data_dir);
        record.checked_at = Some(now());
        write_record(&state.data_dir, &record);
    }
    let mut parts = Vec::new();
    if !outcome.updated.is_empty() {
        parts.push(format!("Updated llama.cpp: {}.", outcome.updated.join(", ")));
    }
    for message in &outcome.waiting {
        let retry = if automatic { format!("Noullama checks again in {} minutes.", UPLOAD_RETRY.as_secs() / 60) } else { "Try again in a few minutes.".into() };
        parts.push(format!("{} {retry}", message.trim_end_matches(" Try again in a few minutes.")));
    }
    if !outcome.errors.is_empty() {
        parts.push("Some llama.cpp updates did not finish.".into());
    } else if parts.is_empty() {
        parts.push("llama.cpp is up to date.".into());
    }
    for error in &outcome.errors {
        state.log(format!("llama.cpp update failed: {error}")).await;
    }
    let mut progress = progress.lock().await;
    progress.active = false;
    progress.message = parts.join(" ");
    progress.error = (!outcome.errors.is_empty()).then(|| outcome.errors.join(" "));
    Ok(waiting)
}

async fn check<F, Fut>(state: &AppState, lines_for: &F, progress: &tokio::sync::Mutex<InstallProgress>) -> Outcome
where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = Result<ReleaseLines, String>>,
{
    let root = runtimes_root(&state.data_dir);
    let mut outcome = Outcome::default();
    let mut record = read_record(&state.data_dir);
    let installed = installed_builds(&root);
    let mut backends: Vec<String> = installed.iter().map(|m| m.build.backend.clone()).collect();
    backends.sort();
    backends.dedup();
    // Each backend, and each line of it, is checked on its own: a failure or
    // a wait on one does not hold up the others.
    for backend in backends {
        progress.lock().await.message = format!("Checking for newer {backend} builds…");
        let releases = match lines_for(backend.clone()).await {
            Ok(releases) => releases,
            Err(error) => {
                outcome.errors.push(format!("{backend}: {error}"));
                continue;
            }
        };
        for channel in [Channel::Nightly, Channel::Stable] {
            let installed = installed_builds(&root);
            let line: Vec<&Managed> = installed.iter().filter(|m| m.build.backend == backend && lines_of(&record, m).contains(&channel)).collect();
            let Some(current) = line.iter().copied().max_by_key(|m| build_number(&m.build.tag)) else {
                continue;
            };
            let next = match if channel == Channel::Nightly { &releases.nightly } else { &releases.stable } {
                Ok(next) => next,
                Err(error) if error.contains(crate::runtimes::STILL_UPLOADING) => {
                    outcome.waiting.push(error.clone());
                    continue;
                }
                Err(error) => {
                    outcome.errors.push(format!("{} {backend}: {error}", channel.name()));
                    continue;
                }
            };
            let Some(next) = successor(&current.build, next) else {
                continue;
            };
            progress.lock().await.total += next.assets.iter().map(|a| a.size).sum::<u64>();
            let new_build = match crate::runtimes::install_build(state, next, progress).await {
                Ok(build) => build,
                Err(error) => {
                    outcome.errors.push(format!("{} {backend} {}: {error}", channel.name(), next.tag));
                    continue;
                }
            };
            // The new folder joins the line (it can already be on the other
            // one, sharing a tag); the line's older folders leave it.
            let new_name = format!("{}-{}", next.tag, next.backend);
            let mut joined = managed(&root, &root.join(&new_name).to_string_lossy()).map(|m| lines_of(&record, &m)).unwrap_or_default();
            if !joined.contains(&channel) {
                joined.push(channel);
            }
            joined.sort();
            for old in line.iter().filter(|m| m.name() != new_name) {
                let rest: Vec<Channel> = lines_of(&record, old).into_iter().filter(|c| *c != channel).collect();
                record.lines.insert(old.name(), rest);
            }
            record.lines.insert(new_name, joined);
            write_record(&state.data_dir, &record);
            state.log(format!("Updated the {} {backend} llama.cpp build from {} to {}.", channel.name(), current.build.tag, next.tag)).await;
            progress.lock().await.installed.push(new_build);
            outcome.updated.push(format!("{} {} → {}", channel.name(), current.build.tag, next.tag));
        }
    }
    if let Err(error) = settle(state, &root, &mut record).await {
        outcome.errors.push(error);
    }
    outcome
}

/// Moves each setting whose folder has left the line it follows to that
/// line's newest folder, then marks the folders that are on no line as
/// superseded. Runs after every check, so a move that could not be saved
/// before is made now.
async fn settle(state: &AppState, root: &Path, record: &mut Record) -> Result<(), String> {
    let installed = installed_builds(root);
    let settings = state.settings.lock().await.clone();
    let mut moves: Vec<(&'static str, String, Channel)> = Vec::new();
    for (field, path) in SETTING_FIELDS.into_iter().zip([&settings.runtime_path, &settings.cpu_runtime_path]) {
        let Some(build) = managed(root, path) else {
            continue;
        };
        let lines = lines_of(record, &build);
        let follows = record.follows.iter().find(|f| f.field == field && f.path == *path).map(|f| f.channel);
        // Unknown: the line it is on, or the one its install record names.
        let channel = follows.unwrap_or_else(|| if lines.len() == 1 { lines[0] } else { Channel::of(&build.build) });
        if lines.contains(&channel) {
            continue;
        }
        let newest = installed
            .iter()
            .filter(|m| m.build.backend == build.build.backend && lines_of(record, m).contains(&channel))
            .max_by_key(|m| build_number(&m.build.tag));
        if let Some(server) = newest.and_then(|m| crate::runtimes::find_server(&m.folder)) {
            moves.push((field, server.to_string_lossy().into_owned(), channel));
        }
    }
    if !moves.is_empty() {
        let changes = moves.clone();
        crate::runtime::update_settings(state, move |s| {
            for (field, path, _) in &changes {
                match *field {
                    "runtime_path" => s.runtime_path = path.clone(),
                    _ => s.cpu_runtime_path = path.clone(),
                }
            }
            Ok(())
        })
        .await
        .map_err(|error| format!("The selected llama.cpp build could not be switched to the new one: {error}"))?;
        for (field, path, channel) in moves {
            record.follows.retain(|f| f.field != field);
            record.follows.push(Follow { field: field.into(), path, channel });
        }
    }
    // Forget folders that are gone and follows that no longer match.
    let settings = state.settings.lock().await.clone();
    record.follows.retain(|f| if f.field == "runtime_path" { settings.runtime_path == f.path } else { settings.cpu_runtime_path == f.path });
    record.lines.retain(|name, _| root.join(name).join(INSTALL_FILE).is_file());
    write_record(&state.data_dir, record);
    for build in installed.iter().filter(|m| lines_of(record, m).is_empty()) {
        let _ = tokio::fs::write(build.folder.join(SUPERSEDED), b"").await;
    }
    Ok(())
}

/// The real location of `path`, or `path` itself when it cannot be resolved.
fn resolved(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Removes superseded build folders that no loaded model runs from and no
/// setting names. Paths are compared as written and resolved, so a symbolic
/// link into a folder keeps it. Holds the lifecycle lock, so a load cannot
/// start from a folder while it is removed.
pub async fn sweep(state: &AppState) {
    let root = runtimes_root(&state.data_dir);
    let Ok(entries) = std::fs::read_dir(&root) else {
        return;
    };
    let candidates: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.join(SUPERSEDED).is_file() && p.join(INSTALL_FILE).is_file()).collect();
    if candidates.is_empty() {
        return;
    }
    let _guard = state.lifecycle.lock().await;
    let settings = state.settings.lock().await.clone();
    let mut used: Vec<PathBuf> = state.instances.lock().await.iter().map(|i| PathBuf::from(i.model().runtime_path)).collect();
    for path in [&settings.runtime_path, &settings.cpu_runtime_path].into_iter().filter(|p| !p.trim().is_empty()) {
        used.push(PathBuf::from(path.trim()));
        used.push(crate::runtime::server_binary(path));
    }
    let used: Vec<PathBuf> = used.iter().flat_map(|p| [p.clone(), resolved(p)]).collect();
    for folder in candidates {
        let real = resolved(&folder);
        if used.iter().any(|p| p.starts_with(&folder) || p.starts_with(&real)) {
            continue;
        }
        let name = folder_name(&folder);
        let target = folder.clone();
        match tokio::task::spawn_blocking(move || std::fs::remove_dir_all(target)).await {
            Ok(Ok(())) => state.log(format!("Removed the replaced llama.cpp build {name}.")).await,
            Ok(Err(error)) => state.log(format!("Could not remove the replaced llama.cpp build {name}: {error}")).await,
            Err(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtimes::ReleaseAsset;

    fn build(release: &str, tag: &str, backend: &str) -> RuntimeBuild {
        RuntimeBuild { release: release.into(), tag: tag.into(), backend: backend.into(), assets: vec![] }
    }

    #[test]
    fn lines_move_within_their_backend_and_never_back() {
        assert_eq!(successor(&build("Nightly", "b100", "cuda13"), &build("Nightly", "b200", "cuda13")).unwrap().tag, "b200");
        assert!(successor(&build("Nightly", "b200", "cuda13"), &build("Nightly", "b200", "cuda13")).is_none(), "up to date");
        assert!(successor(&build("Nightly", "b300", "cuda13"), &build("Nightly", "b200", "cuda13")).is_none(), "never downgrades");
        // A stable release without this backend falls back to CPU; that is not an update.
        assert!(successor(&build("b120", "b120", "cuda13"), &build("b150", "b150", "cpu")).is_none());
        assert!(is_newer("v0.9.0", "v0.8.0") && !is_newer("v0.8.0", "v0.8.0"));
    }

    #[test]
    fn only_managed_builds_are_recognised() {
        let base = std::env::temp_dir().join(format!("noullama-updates-{}", uuid::Uuid::new_v4()));
        let root = base.join("runtimes");
        let folder = root.join("b100-cpu");
        std::fs::create_dir_all(folder.join("package-0/bin")).unwrap();
        std::fs::write(folder.join(INSTALL_FILE), serde_json::to_vec(&build("Nightly", "b100", "cpu")).unwrap()).unwrap();
        std::fs::create_dir_all(root.join("custom")).unwrap();
        let server = folder.join("package-0/bin/llama-server");
        assert_eq!(managed(&root, &server.to_string_lossy()).unwrap().folder, folder);
        assert_eq!(managed(&root, &folder.to_string_lossy()).unwrap().build.tag, "b100");
        assert!(managed(&root, &root.join("custom/llama-server").to_string_lossy()).is_none(), "no install record");
        assert!(managed(&root, "/usr/bin/llama-server").is_none());
        assert!(managed(&root, "").is_none());
        assert!(!is_superseded(&root, &server.to_string_lossy()));
        std::fs::write(folder.join(SUPERSEDED), b"").unwrap();
        assert!(is_superseded(&root, &server.to_string_lossy()), "the Runtime tab does not offer a replaced build");
        std::fs::remove_dir_all(base).unwrap();
    }

    /// A state whose downloads come from a local fixture archive, and a
    /// maker of CPU builds that download it.
    async fn fixture() -> (PathBuf, AppState, impl Fn(&str, &str) -> RuntimeBuild) {
        use axum::{body::Body, response::Response, routing::get, Router};
        use sha2::{Digest, Sha256};
        let base = std::env::temp_dir().join(format!("noullama-updates-{}", uuid::Uuid::new_v4()));
        let mut state = AppState::new(base.join("data")).unwrap();
        state.client = state.local_client.clone();
        // A llama-server that is never executed.
        let mut bytes = Vec::new();
        {
            let gzip = flate2::write::GzEncoder::new(&mut bytes, flate2::Compression::default());
            let mut tar = tar::Builder::new(gzip);
            let data = b"fixture only; never executed";
            let mut header = tar::Header::new_gnu();
            header.set_mode(0o755);
            header.set_size(data.len() as u64);
            header.set_cksum();
            tar.append_data(&mut header, "build/llama-server", &data[..]).unwrap();
            tar.into_inner().unwrap().finish().unwrap();
        }
        let digest = format!("sha256:{:x}", Sha256::digest(&bytes));
        let size = bytes.len() as u64;
        let router = Router::new().route("/archive", get(move || { let data = bytes.clone(); async move { Response::new(Body::from(data)) } }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/archive", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let remote = move |release: &str, tag: &str| RuntimeBuild {
            release: release.into(),
            tag: tag.into(),
            backend: "cpu".into(),
            assets: vec![ReleaseAsset { name: format!("llama-{tag}-bin-ubuntu-x64.tar.gz"), size, digest: digest.clone(), url: url.clone() }],
        };
        (base, state, remote)
    }

    async fn install(state: &AppState, build: &RuntimeBuild) -> String {
        let progress = tokio::sync::Mutex::new(InstallProgress::default());
        crate::runtimes::install_build(state, build, &progress).await.unwrap().path
    }

    async fn select(state: &AppState, path: &str) {
        let path = path.to_owned();
        crate::runtime::update_settings(state, move |s| {
            s.runtime_path = path;
            Ok(())
        })
        .await
        .unwrap();
    }

    /// One check against fixed releases; true when it is waiting.
    async fn run(state: &AppState, automatic: bool, nightly: Result<RuntimeBuild, String>, stable: Result<RuntimeBuild, String>) -> bool {
        update(state, automatic, |_| {
            let lines = ReleaseLines { nightly: nightly.clone(), stable: stable.clone() };
            async move { Ok(lines) }
        })
        .await
        .unwrap()
    }

    fn uploading(tag: &str) -> Result<RuntimeBuild, String> {
        Err(format!("llama.cpp {tag} {}: its linux/x64 cpu build is not there yet. Try again in a few minutes.", crate::runtimes::STILL_UPLOADING))
    }

    #[tokio::test]
    async fn both_lines_update_and_replaced_builds_go_once_unused() {
        let (base, state, remote) = fixture().await;
        // As after "Download nightly + fallback": a nightly and a stable
        // build, with the stable one selected and the nightly not.
        install(&state, &remote("Nightly", "b100")).await;
        let old = install(&state, &remote("b090", "b090")).await;
        let root = runtimes_root(&state.data_dir);
        let (old_nightly, old_stable) = (root.join("b100-cpu"), root.join("b090-cpu"));
        select(&state, &old).await;
        assert!(!run(&state, true, Ok(remote("Nightly", "b200")), Ok(remote("b150", "b150"))).await);
        let progress = state.runtimes.update_progress.lock().await.clone();
        assert!(!progress.active && progress.error.is_none(), "{:?}", progress.error);
        assert_eq!(progress.installed.len(), 2, "both the nightly and the stable build are updated");
        let settings = state.settings.lock().await.clone();
        assert!(settings.runtime_path.contains("b150-cpu"), "the selection follows its own line: {}", settings.runtime_path);
        assert!(settings.cpu_runtime_path.is_empty(), "an empty setting stays empty");
        assert!(root.join("b200-cpu").join(INSTALL_FILE).is_file(), "the unselected nightly is updated too");
        assert!(old_nightly.join(SUPERSEDED).is_file() && old_stable.join(SUPERSEDED).is_file());
        let checked = read_record(&state.data_dir).checked_at;
        assert!(checked.is_some());
        // A model still running from the old build keeps it until it is unloaded.
        let instance = crate::runtime::insert_test_instance(&state, "/models/a.gguf", true).await;
        instance.set_test_runtime_path(&old);
        sweep(&state).await;
        assert!(old_stable.exists(), "a build a loaded model runs from is kept");
        assert!(!old_nightly.exists(), "an unused replaced build is removed");
        state.instances.lock().await.clear();
        sweep(&state).await;
        assert!(!old_stable.exists(), "the replaced build is removed once nothing uses it");
        // Up to date: nothing changes and nothing is marked.
        run(&state, true, Ok(remote("Nightly", "b200")), Ok(remote("b150", "b150"))).await;
        assert_eq!(state.runtimes.update_progress.lock().await.message, "llama.cpp is up to date.");
        assert!(!root.join("b200-cpu").join(SUPERSEDED).exists() && !root.join("b150-cpu").join(SUPERSEDED).exists());
        // GitHub unreadable: reported, and not counted as a check.
        let mut record = read_record(&state.data_dir);
        record.checked_at = checked;
        write_record(&state.data_dir, &record);
        update(&state, true, |_| async { Err::<ReleaseLines, String>("offline".into()) }).await.unwrap();
        assert_eq!(state.runtimes.update_progress.lock().await.error.as_deref(), Some("cpu: offline"));
        assert_eq!(read_record(&state.data_dir).checked_at, checked);
        // The newest nightly still uploading does not hold up the stable line,
        // and the check waits instead of failing.
        assert!(run(&state, true, uploading("b300"), Ok(remote("b160", "b160"))).await, "waiting");
        let waiting = state.runtimes.update_progress.lock().await.clone();
        assert!(waiting.error.is_none() && waiting.message.ends_with("Noullama checks again in 10 minutes."), "{}", waiting.message);
        assert!(state.settings.lock().await.runtime_path.contains("b160-cpu"), "the stable line moved meanwhile");
        assert_eq!(read_record(&state.data_dir).checked_at, checked);
        // Check now does not promise a retry it does not schedule.
        assert!(run(&state, false, uploading("b300"), Ok(remote("b160", "b160"))).await);
        assert!(state.runtimes.update_progress.lock().await.message.ends_with("Try again in a few minutes."));
        // Another runtime download blocks a check.
        state.runtimes.progress.lock().await.active = true;
        assert!(update(&state, true, |_| async { Err::<ReleaseLines, String>("unused".into()) }).await.is_err());
        std::fs::remove_dir_all(base).unwrap();
    }

    #[tokio::test]
    async fn a_stable_selection_stays_stable_when_it_shares_a_nightly_folder() {
        let (base, state, remote) = fixture().await;
        let root = runtimes_root(&state.data_dir);
        install(&state, &remote("Nightly", "b100")).await;
        select(&state, &install(&state, &remote("b090", "b090")).await).await;
        // The new stable release is the nightly already installed: its folder
        // is reused (its install record still says Nightly) and now on both lines.
        run(&state, true, Ok(remote("Nightly", "b100")), Ok(remote("Official v2", "b100"))).await;
        assert!(state.settings.lock().await.runtime_path.contains("b100-cpu"));
        assert!(root.join("b090-cpu").join(SUPERSEDED).is_file());
        // The nightly line moves on; the stable selection stays.
        run(&state, true, Ok(remote("Nightly", "b120")), Ok(remote("Official v2", "b100"))).await;
        assert!(state.settings.lock().await.runtime_path.contains("b100-cpu"), "a stable selection does not drift to a nightly");
        assert!(!root.join("b100-cpu").join(SUPERSEDED).exists(), "still the stable line's build");
        // The next stable release moves it, and the shared folder leaves both lines.
        run(&state, true, Ok(remote("Nightly", "b120")), Ok(remote("Official v3", "b110"))).await;
        assert!(state.settings.lock().await.runtime_path.contains("b110-cpu"));
        assert!(root.join("b100-cpu").join(SUPERSEDED).is_file());
        assert!(!root.join("b120-cpu").join(SUPERSEDED).exists());
        std::fs::remove_dir_all(base).unwrap();
    }

    #[tokio::test]
    async fn a_switch_that_could_not_be_saved_is_made_by_the_next_check() {
        let (base, state, remote) = fixture().await;
        let root = runtimes_root(&state.data_dir);
        select(&state, &install(&state, &remote("Nightly", "b100")).await).await;
        // Settings that fail validation make the save fail.
        state.settings.lock().await.api_port = 11436;
        run(&state, true, Ok(remote("Nightly", "b200")), Err("no stable".into())).await;
        assert!(state.runtimes.update_progress.lock().await.error.as_deref().unwrap_or_default().contains("could not be switched"));
        assert!(state.settings.lock().await.runtime_path.contains("b100-cpu"));
        assert!(!root.join("b100-cpu").join(SUPERSEDED).exists(), "kept while selected");
        state.settings.lock().await.api_port = 11435;
        run(&state, true, Ok(remote("Nightly", "b200")), Err("no stable".into())).await;
        assert!(state.settings.lock().await.runtime_path.contains("b200-cpu"), "the next check finishes the switch");
        assert!(root.join("b100-cpu").join(SUPERSEDED).is_file());
        std::fs::remove_dir_all(base).unwrap();
    }

    #[tokio::test]
    async fn a_symbolic_link_into_a_replaced_build_keeps_it() {
        let (base, state, remote) = fixture().await;
        let root = runtimes_root(&state.data_dir);
        let server = install(&state, &remote("Nightly", "b100")).await;
        std::fs::write(root.join("b100-cpu").join(SUPERSEDED), b"").unwrap();
        let link = base.join("my-llama-server");
        std::os::unix::fs::symlink(&server, &link).unwrap();
        select(&state, &link.to_string_lossy()).await;
        sweep(&state).await;
        assert!(root.join("b100-cpu").exists(), "the selected link resolves into it");
        select(&state, "").await;
        sweep(&state).await;
        assert!(!root.join("b100-cpu").exists());
        std::fs::remove_dir_all(base).unwrap();
    }
}
