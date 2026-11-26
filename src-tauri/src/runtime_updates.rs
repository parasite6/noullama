//! Automatic llama.cpp updates.
//!
//! Only builds Noullama downloaded itself are updated: the folders in
//! `<data>/runtimes/<tag>-<backend>` with the `noullama-install.json` that
//! `install_build` writes, selected or not. A custom build is never touched.
//! Each channel of each backend is kept current on its own: the nightlies
//! move to the newest nightly, the stable builds to the newest stable release.
//!
//! An update installs the new build, points the settings that named a build
//! it replaces at it, and marks the replaced folders superseded. A superseded folder is
//! removed once no loaded model runs from it and no setting names it, so a
//! model that is loaded keeps its build until it is unloaded.

use crate::runtime::AppState;
use crate::runtimes::{Download, InstallProgress, RuntimeBuild, RuntimePlan};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    path::{Component, Path, PathBuf},
    sync::atomic::Ordering,
    time::{Duration, SystemTime},
};

/// How often the selected builds are checked: llama.cpp publishes several
/// nightlies a day, and each CUDA build is hundreds of megabytes.
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

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Record {
    /// When the last check finished, in seconds since the Unix epoch.
    checked_at: Option<u64>,
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

/// A selected managed build: its folder under `<data>/runtimes` and what
/// `install_build` recorded about it.
struct Managed {
    folder: PathBuf,
    build: RuntimeBuild,
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

/// A `b<number>` tag's number, for ordering builds; other tags sort first.
fn build_number(tag: &str) -> u64 {
    tag.strip_prefix('b').and_then(|n| n.parse().ok()).unwrap_or(0)
}

fn is_nightly(build: &RuntimeBuild) -> bool {
    build.release == "Nightly"
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

/// The build `current` should move to from `plan`, if there is a newer one
/// for its channel and backend.
fn successor<'a>(current: &RuntimeBuild, plan: &'a RuntimePlan) -> Option<&'a RuntimeBuild> {
    let next = if is_nightly(current) { &plan.nightly } else { &plan.fallback };
    (next.backend == current.backend && is_newer(&next.tag, &current.tag)).then_some(next)
}

/// Starts the background checks while automatic updates are on: 10 seconds
/// after launch and then hourly, a check runs when the last successful one
/// is a day old. Superseded builds are removed at the same ticks once
/// nothing uses them, whether or not updates are on.
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
                waiting = update(&state, |backend| {
                    let state = state.clone();
                    async move { crate::runtimes::release_plan(&state, &backend).await }
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

/// "Check now": checks and updates the selected builds at once, also when
/// automatic updates are off. Returns when the check has finished.
#[tauri::command]
pub async fn update_runtime_now(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let state = state.inner().clone();
    let worker = state.clone();
    update(&state, move |backend| {
        let state = worker.clone();
        async move { crate::runtimes::release_plan(&state, &backend).await }
    })
    .await?;
    sweep(&state).await;
    Ok(())
}

/// Checks the installed managed builds against `plan_for(backend)` and
/// installs any newer one. Fails only when another runtime download is
/// running; a failed check is reported in the update's progress. Returns
/// true when the newest nightly is still uploading, so the check is waiting.
async fn update<F, Fut>(state: &AppState, plan_for: F) -> Result<bool, String>
where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = Result<RuntimePlan, String>>,
{
    // The update runs in this task, so the lifecycle lock is not held for it.
    drop(crate::runtimes::claim(state, Download::Update, "Checking for llama.cpp updates…").await?);
    let progress = &state.runtimes.update_progress;
    let result = check(state, &plan_for, progress).await;
    // A failed check (offline, rate limited) is tried again at the next tick.
    if result.is_ok() {
        write_record(&state.data_dir, &Record { checked_at: Some(now()) });
    }
    let mut progress = progress.lock().await;
    progress.active = false;
    match result {
        Ok(message) => progress.message = message,
        // Not a failure: the release is there, its archives are not yet.
        Err(error) if error.contains(crate::runtimes::STILL_UPLOADING) => {
            progress.message = format!("{} Noullama checks again in {} minutes.", error.trim_end_matches(" Try again in a few minutes."), UPLOAD_RETRY.as_secs() / 60);
            return Ok(true);
        }
        Err(error) => {
            progress.message = "The llama.cpp update did not finish.".into();
            state.log(format!("llama.cpp update failed: {error}")).await;
            progress.error = Some(error);
        }
    }
    Ok(false)
}

async fn check<F, Fut>(state: &AppState, plan_for: &F, progress: &tokio::sync::Mutex<InstallProgress>) -> Result<String, String>
where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = Result<RuntimePlan, String>>,
{
    let root = runtimes_root(&state.data_dir);
    let installed = installed_builds(&root);
    if installed.is_empty() {
        return Ok("No llama.cpp build that Noullama downloaded is installed, so there is nothing to update.".into());
    }
    // One channel (nightly or stable) of one backend is one line of builds;
    // each line moves to its newest release.
    let mut lines: Vec<(String, bool)> = installed.iter().map(|m| (m.build.backend.clone(), is_nightly(&m.build))).collect();
    lines.sort();
    lines.dedup();
    let mut plans: Vec<(String, RuntimePlan)> = Vec::new();
    let mut updated = Vec::new();
    for (backend, nightly) in lines {
        let line: Vec<&Managed> = installed.iter().filter(|m| m.build.backend == backend && is_nightly(&m.build) == nightly).collect();
        let Some(current) = line.iter().copied().max_by_key(|m| build_number(&m.build.tag)) else {
            continue;
        };
        if !plans.iter().any(|(b, _)| *b == backend) {
            progress.lock().await.message = format!("Checking for newer {backend} builds…");
            plans.push((backend.clone(), plan_for(backend.clone()).await?));
        }
        let plan = &plans.iter().find(|(b, _)| *b == backend).unwrap().1;
        let Some(next) = successor(&current.build, plan) else {
            continue;
        };
        progress.lock().await.total += next.assets.iter().map(|a| a.size).sum::<u64>();
        let new_build = crate::runtimes::install_build(state, next, progress).await?;
        let new_folder = root.join(format!("{}-{}", next.tag, next.backend));
        // The newest build of the other channel can share a folder with an
        // older build of this one (same tag); it is kept.
        let other = if nightly { &plan.fallback } else { &plan.nightly };
        let keep = root.join(format!("{}-{}", other.tag, other.backend));
        let replaced: Vec<PathBuf> = line.iter().map(|m| m.folder.clone()).filter(|f| *f != new_folder && *f != keep).collect();
        // Point each setting that names a replaced build at the new one.
        let new_path = new_build.path.clone();
        crate::runtime::update_settings(state, |s| {
            for field in [&mut s.runtime_path, &mut s.cpu_runtime_path] {
                if managed(&root, field).is_some_and(|m| replaced.contains(&m.folder)) {
                    *field = new_path.clone();
                }
            }
            Ok(())
        })
        .await?;
        for folder in &replaced {
            let _ = tokio::fs::write(folder.join(SUPERSEDED), b"").await;
        }
        let channel = if nightly { "nightly" } else { "stable" };
        state.log(format!("Updated the {channel} {backend} llama.cpp build from {} to {}.", current.build.tag, next.tag)).await;
        progress.lock().await.installed.push(new_build);
        updated.push(format!("{channel} {} → {}", current.build.tag, next.tag));
    }
    Ok(if updated.is_empty() { "llama.cpp is up to date.".into() } else { format!("Updated llama.cpp: {}.", updated.join(", ")) })
}

/// Removes superseded build folders that no loaded model runs from and no
/// setting names. Holds the lifecycle lock, so a load cannot start from a
/// folder while it is removed.
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
    let running: Vec<String> = state.instances.lock().await.iter().map(|i| i.model().runtime_path).collect();
    for folder in candidates {
        let named = [&settings.runtime_path, &settings.cpu_runtime_path].into_iter().any(|p| managed(&root, p).is_some_and(|m| m.folder == folder));
        let in_use = running.iter().any(|p| Path::new(p).starts_with(&folder));
        if named || in_use {
            continue;
        }
        let name = folder.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
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

    fn plan(nightly: RuntimeBuild, fallback: RuntimeBuild) -> RuntimePlan {
        RuntimePlan { id: "p".into(), platform: "test".into(), nightly, fallback, total_bytes: 0 }
    }

    #[test]
    fn builds_move_within_their_channel_and_backend() {
        let latest = plan(build("Nightly", "b200", "cuda13"), build("b150", "b150", "cuda13"));
        assert_eq!(successor(&build("Nightly", "b100", "cuda13"), &latest).unwrap().tag, "b200");
        assert_eq!(successor(&build("b120", "b120", "cuda13"), &latest).unwrap().tag, "b150");
        assert!(successor(&build("Nightly", "b200", "cuda13"), &latest).is_none(), "up to date");
        assert!(successor(&build("Nightly", "b300", "cuda13"), &latest).is_none(), "never downgrades");
        // A stable release without this backend falls back to CPU; that is not an update.
        let cpu_fallback = plan(build("Nightly", "b200", "cuda13"), build("b150", "b150", "cpu"));
        assert!(successor(&build("b120", "b120", "cuda13"), &cpu_fallback).is_none());
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
        assert_eq!(managed(&root, &folder.join("package-0/bin/llama-server").to_string_lossy()).unwrap().folder, folder);
        assert_eq!(managed(&root, &folder.to_string_lossy()).unwrap().build.tag, "b100");
        assert!(managed(&root, &root.join("custom/llama-server").to_string_lossy()).is_none(), "no install record");
        assert!(managed(&root, "/usr/bin/llama-server").is_none());
        assert!(managed(&root, "").is_none());
        std::fs::remove_dir_all(base).unwrap();
    }

    #[tokio::test]
    async fn an_update_switches_the_selection_and_removes_the_old_build_once_unused() {
        use axum::{body::Body, response::Response, routing::get, Router};
        use sha2::{Digest, Sha256};
        let base = std::env::temp_dir().join(format!("noullama-updates-{}", uuid::Uuid::new_v4()));
        let mut state = AppState::new(base.join("data")).unwrap();
        state.client = state.local_client.clone();
        // A fixture archive with a llama-server that is never executed.
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
        let fixture = bytes.clone();
        let router = Router::new().route("/archive", get(move || { let data = fixture.clone(); async move { Response::new(Body::from(data)) } }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/archive", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let remote = |release: &str, tag: &str| RuntimeBuild {
            release: release.into(),
            tag: tag.into(),
            backend: "cpu".into(),
            assets: vec![ReleaseAsset { name: format!("llama-{tag}-bin-ubuntu-x64.tar.gz"), size: bytes.len() as u64, digest: digest.clone(), url: url.clone() }],
        };
        let progress = tokio::sync::Mutex::new(InstallProgress::default());
        // As after "Download nightly + fallback": a nightly and a stable build,
        // with the stable one selected and the nightly left unselected.
        crate::runtimes::install_build(&state, &remote("Nightly", "b100"), &progress).await.unwrap();
        let old = crate::runtimes::install_build(&state, &remote("b090", "b090"), &progress).await.unwrap();
        let root = runtimes_root(&state.data_dir);
        let (old_nightly, old_stable) = (root.join("b100-cpu"), root.join("b090-cpu"));
        crate::runtime::update_settings(&state, |s| {
            s.runtime_path = old.path.clone();
            Ok(())
        })
        .await
        .unwrap();
        let latest = plan(remote("Nightly", "b200"), remote("b150", "b150"));
        update(&state, |_| { let latest = latest.clone(); async move { Ok(latest) } }).await.unwrap();
        let progress = state.runtimes.update_progress.lock().await.clone();
        assert!(!progress.active && progress.error.is_none(), "{:?}", progress.error);
        assert_eq!(progress.installed.len(), 2, "both the nightly and the stable build are updated");
        let settings = state.settings.lock().await.clone();
        assert!(settings.runtime_path.contains("b150-cpu"), "the selection follows its own channel: {}", settings.runtime_path);
        assert!(settings.cpu_runtime_path.is_empty(), "an empty setting stays empty");
        assert!(Path::new(&settings.runtime_path).is_file());
        assert!(root.join("b200-cpu").join(INSTALL_FILE).is_file(), "the unselected nightly is updated too");
        assert!(old_nightly.join(SUPERSEDED).is_file() && old_stable.join(SUPERSEDED).is_file());
        assert!(read_record(&state.data_dir).checked_at.is_some());
        let checked = read_record(&state.data_dir).checked_at;
        // A model still running from the old build keeps it until it is unloaded.
        let instance = crate::runtime::insert_test_instance(&state, "/models/a.gguf", true).await;
        instance.set_test_runtime_path(&old.path);
        sweep(&state).await;
        assert!(old_stable.exists(), "a build a loaded model runs from is kept");
        assert!(!old_nightly.exists(), "an unused replaced build is removed");
        state.instances.lock().await.clear();
        sweep(&state).await;
        assert!(!old_stable.exists(), "the replaced build is removed once nothing uses it");
        // Up to date: nothing changes and nothing is marked.
        update(&state, |_| { let latest = latest.clone(); async move { Ok(latest) } }).await.unwrap();
        assert_eq!(state.runtimes.update_progress.lock().await.message, "llama.cpp is up to date.");
        assert!(!root.join("b200-cpu").join(SUPERSEDED).exists() && !root.join("b150-cpu").join(SUPERSEDED).exists());
        // A failed check is reported and does not count as a check.
        write_record(&state.data_dir, &Record { checked_at: checked });
        update(&state, |_| async { Err::<RuntimePlan, String>("offline".into()) }).await.unwrap();
        assert_eq!(state.runtimes.update_progress.lock().await.error.as_deref(), Some("offline"));
        assert_eq!(read_record(&state.data_dir).checked_at, checked);
        // The newest nightly still uploading: waiting, not failed, and not a check.
        let uploading = format!("llama.cpp b300 {}: its linux/x64 cpu build is not there yet. Try again in a few minutes.", crate::runtimes::STILL_UPLOADING);
        assert!(update(&state, |_| { let e = uploading.clone(); async move { Err::<RuntimePlan, String>(e) } }).await.unwrap(), "waiting");
        let waiting = state.runtimes.update_progress.lock().await.clone();
        assert!(waiting.error.is_none() && waiting.message.ends_with("Noullama checks again in 10 minutes."), "{}", waiting.message);
        assert_eq!(read_record(&state.data_dir).checked_at, checked);
        // Another runtime download blocks a check.
        state.runtimes.progress.lock().await.active = true;
        assert!(update(&state, |_| { let latest = latest.clone(); async move { Ok(latest) } }).await.is_err());
        std::fs::remove_dir_all(base).unwrap();
    }
}
