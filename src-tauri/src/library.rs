//! models.yml: one entry per model file with the name the local API knows it
//! by, its aliases, its path, the Hugging Face account its avatar comes from,
//! and its sha256. People edit it in the Library tab or in any text editor;
//! Noullama rereads it whenever it changes on disk and backs it up before
//! every change it makes.

use crate::runtime::{file_stem, normalize_aliases, AppState};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    time::SystemTime,
};

pub const FILE: &str = "models.yml";
const BACKUPS_KEPT: usize = 20;
const HASH_CACHE: &str = "sha256-cache.json";

const HEADER: &str = "\
# Noullama models: one entry per model file. Edit it here or in Noullama's
# Library tab; Noullama rereads it when it changes and backs it up to
# backups/models before every change it makes.
#
#   name     The model's API ID and display name. Requests send it as
#            \"model\", exactly as written.
#   aliases  More names that reach the same model.
#   path     Absolute path to the .gguf file.
#   account  Hugging Face account the avatar comes from, such as LiquidAI.
#   sha256   Checksum of the file. Filled in automatically; see Settings.
";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Entry {
    pub name: String,
    pub aliases: Vec<String>,
    pub path: String,
    pub account: String,
    pub sha256: String,
}

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Document {
    version: u32,
    models: Vec<Entry>,
}

#[derive(Default)]
pub struct Library {
    /// The last valid contents; kept while the file on disk has an error.
    pub entries: Vec<Entry>,
    /// Why the file on disk cannot be used right now.
    pub error: Option<String>,
    /// Bumped when someone changes names, aliases, paths or accounts (in the
    /// file or in Library), so an editor working from an older copy cannot
    /// overwrite a newer file. Recording a sha256 does not bump it.
    pub revision: u64,
    stamp: Option<(SystemTime, u64)>,
    text: String,
}

impl Library {
    pub fn entry(&self, path: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.path == path)
    }
    /// The entry a request's model name reaches: its name or an alias, exact case.
    pub fn named(&self, name: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.name == name || e.aliases.iter().any(|a| a == name))
    }
}

/// sha256 checks before a load, chosen in Settings.
#[derive(Clone, Copy, PartialEq)]
pub enum Check {
    Off,
    /// Hash once; hash again only when the file's size or modified time changes.
    Once,
    EveryLoad,
    /// As Once, but a mismatch only warns.
    Warn,
}

impl Check {
    pub fn parse(value: &str) -> Self {
        match value {
            "off" => Check::Off,
            "every_load" => Check::EveryLoad,
            "warn" => Check::Warn,
            _ => Check::Once,
        }
    }
}

pub fn file_path(data_dir: &Path) -> PathBuf {
    data_dir.join(FILE)
}

pub fn backups_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("backups").join("models")
}

/// The Hugging Face account in a downloaded path such as
/// `…/LiquidAI--d1-3B-GGUF/d1-3B-Q8_0.gguf` or a hub cache `models--LiquidAI--…`.
pub fn account_from_path(path: &str) -> String {
    Path::new(path)
        .components()
        .rev()
        .filter_map(|part| part.as_os_str().to_str())
        .map(|part| part.strip_prefix("models--").unwrap_or(part))
        .find_map(|part| {
            let (account, rest) = part.split_once("--")?;
            (!account.is_empty() && !rest.is_empty() && account.chars().all(valid_account_char)).then(|| account.to_owned())
        })
        .unwrap_or_default()
}

fn valid_account_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || "_.-".contains(c)
}

fn valid_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || "._:/-".contains(c)
}

/// A free name for a new entry, made from the file name.
fn unique_name(path: &str, taken: &BTreeSet<String>) -> String {
    let base: String = file_stem(path).chars().map(|c| if valid_name_char(c) { c } else { '-' }).take(60).collect();
    let base = if base.is_empty() { "model".to_owned() } else { base };
    (1..)
        .map(|n| if n == 1 { base.clone() } else { format!("{base}-{n}") })
        .find(|name| !taken.contains(&name.to_lowercase()))
        .unwrap_or(base)
}

fn new_entry(path: &str, taken: &BTreeSet<String>) -> Entry {
    Entry { name: unique_name(path, taken), aliases: Vec::new(), path: path.to_owned(), account: account_from_path(path), sha256: String::new() }
}

fn taken_names(entries: &[Entry]) -> BTreeSet<String> {
    entries.iter().flat_map(|e| std::iter::once(&e.name).chain(&e.aliases)).map(|n| n.to_lowercase()).collect()
}

/// Checks every entry and returns them tidied (trimmed, lower-case sha256).
/// Names and aliases are matched exactly, but no two may differ only by case.
pub fn validate(entries: &[Entry]) -> Result<Vec<Entry>, String> {
    let mut out: Vec<Entry> = Vec::new();
    let mut owners: BTreeMap<String, String> = BTreeMap::new();
    for (index, raw) in entries.iter().enumerate() {
        let label = if raw.name.trim().is_empty() { format!("Entry {}", index + 1) } else { raw.name.trim().to_owned() };
        let name = raw.name.trim().to_owned();
        if name.is_empty() || name.len() > 64 || !name.chars().all(valid_name_char) {
            return Err(format!("{label}: the name needs letters, digits and . _ : / - only, up to 64 characters."));
        }
        let path = raw.path.trim().to_owned();
        if !Path::new(&path).is_absolute() || !path.to_lowercase().ends_with(".gguf") {
            return Err(format!("{label}: the path must be an absolute path to a .gguf file."));
        }
        if out.iter().any(|e| e.path == path) {
            return Err(format!("{label}: {path} is listed twice."));
        }
        let account = raw.account.trim().to_owned();
        if account.len() > 96 || !account.chars().all(valid_account_char) {
            return Err(format!("{label}: the account can use letters, digits and _ . - only."));
        }
        let sha256 = raw.sha256.trim().to_lowercase();
        if !sha256.is_empty() && (sha256.len() != 64 || !sha256.chars().all(|c| c.is_ascii_hexdigit())) {
            return Err(format!("{label}: sha256 must be 64 hexadecimal characters, or empty."));
        }
        let aliases: Vec<String> = normalize_aliases(&raw.aliases).map_err(|e| format!("{label}: {e}"))?.into_iter().filter(|a| *a != name).collect();
        for value in std::iter::once(&name).chain(&aliases) {
            if let Some(owner) = owners.insert(value.to_lowercase(), name.clone()) {
                return Err(if owner == name {
                    format!("{label}: \"{value}\" appears twice.")
                } else {
                    format!("\"{value}\" is used by both {owner} and {name}. Each name and alias must be unique, ignoring case.")
                });
            }
        }
        out.push(Entry { name, aliases, path, account, sha256 });
    }
    Ok(out)
}

fn parse(text: &str) -> Result<Vec<Entry>, String> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let document: Document = serde_norway::from_str(text).map_err(|e| format!("models.yml could not be read: {e}"))?;
    validate(&document.models).map_err(|e| format!("models.yml: {e}"))
}

fn render(entries: &[Entry]) -> Result<String, String> {
    let body = serde_norway::to_string(&Document { version: 1, models: entries.to_vec() }).map_err(|e| e.to_string())?;
    Ok(format!("{HEADER}\n{body}"))
}

fn now_millis() -> u128 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_millis()
}

/// Saves `text` to backups/models and keeps the newest few.
fn backup(data_dir: &Path, text: &str) -> Result<(), String> {
    if text.trim().is_empty() {
        return Ok(());
    }
    let dir = backups_dir(data_dir);
    std::fs::create_dir_all(&dir).map_err(|e| format!("Cannot create {}: {e}", dir.display()))?;
    std::fs::write(dir.join(format!("models-{}.yml", now_millis())), text).map_err(|e| format!("Cannot back up models.yml: {e}"))?;
    let mut backups: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|entry| entry.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("models-") && n.ends_with(".yml")))
        .collect();
    backups.sort();
    let extra = backups.len().saturating_sub(BACKUPS_KEPT);
    for old in &backups[..extra] {
        let _ = std::fs::remove_file(old);
    }
    Ok(())
}

fn stamp(path: &Path) -> Option<(SystemTime, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

/// How a write relates to people editing the file.
#[derive(Clone, Copy, PartialEq)]
enum Change {
    /// Names, aliases, paths or accounts: back up first and bump the revision.
    Edit,
    /// Only sha256 values Noullama recorded itself: no backup, so hashing a
    /// folder of models does not push out the backups people care about, and
    /// no revision bump, so it never invalidates an edit in progress.
    Checksums,
}

/// Writes the entries, backing up the file on disk first.
fn write(data_dir: &Path, library: &mut Library, entries: Vec<Entry>, change: Change) -> Result<(), String> {
    let path = file_path(data_dir);
    let text = render(&entries)?;
    if change == Change::Edit {
        if let Ok(current) = std::fs::read_to_string(&path) {
            backup(data_dir, &current)?;
        }
    }
    let temp = data_dir.join("models.yml.tmp");
    std::fs::write(&temp, &text).map_err(|e| format!("Cannot write models.yml: {e}"))?;
    std::fs::rename(&temp, &path).map_err(|e| format!("Cannot write models.yml: {e}"))?;
    library.entries = entries;
    library.text = text;
    library.stamp = stamp(&path);
    library.error = None;
    if change == Change::Edit {
        library.revision += 1;
    }
    Ok(())
}

/// Rereads models.yml if it changed on disk. A file with an error is
/// reported and the last valid contents stay in use; it is never overwritten
/// until someone saves. Creates the file (from aliases saved in settings by
/// an earlier version) when there is none.
pub async fn refresh(state: &AppState) {
    let legacy = state.settings.lock().await.model_routes.clone();
    let mut library = state.library.lock().await;
    refresh_locked(&state.data_dir, &mut library, &legacy);
}

fn refresh_locked(data_dir: &Path, library: &mut Library, legacy: &BTreeMap<String, Vec<String>>) {
    let path = file_path(data_dir);
    let Some(current) = stamp(&path) else {
        if path.exists() {
            library.error = Some("models.yml cannot be read.".into());
            return;
        }
        let mut entries: Vec<Entry> = library.entries.clone();
        for (model, aliases) in legacy {
            if entries.iter().all(|e| e.path != *model) {
                let mut entry = new_entry(model, &taken_names(&entries));
                entry.aliases = aliases.clone();
                entries.push(entry);
            }
        }
        let entries = validate(&entries).unwrap_or_default();
        if let Err(error) = write(data_dir, library, entries, Change::Edit) {
            library.error = Some(error);
        }
        return;
    };
    if library.stamp == Some(current) {
        return;
    }
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            library.error = Some(format!("models.yml cannot be read: {error}"));
            return;
        }
    };
    library.stamp = Some(current);
    if text == library.text {
        return;
    }
    match parse(&text) {
        Ok(entries) => {
            // Someone edited the file: keep the version before their edit.
            let _ = backup(data_dir, &library.text);
            library.entries = entries;
            library.text = text;
            library.error = None;
            library.revision += 1;
        }
        Err(error) => library.error = Some(format!("{error} Noullama keeps using the last valid version until it is fixed.")),
    }
}

/// Adds an entry for each model file that has none.
pub async fn ensure(state: &AppState, paths: &[String]) {
    refresh(state).await;
    let added = {
        let mut library = state.library.lock().await;
        let mut entries = library.entries.clone();
        for path in paths {
            if !path.is_empty() && entries.iter().all(|e| e.path != *path) {
                let entry = new_entry(path, &taken_names(&entries));
                entries.push(entry);
            }
        }
        if entries.len() == library.entries.len() || library.error.is_some() {
            false
        } else {
            match validate(&entries).and_then(|entries| write(&state.data_dir, &mut library, entries, Change::Edit)) {
                Ok(()) => true,
                Err(error) => {
                    state.log(format!("models.yml was not updated: {error}")).await;
                    false
                }
            }
        }
    };
    if added {
        hash_missing(state.clone());
    }
}

/// Applies one change to every entry and saves the result.
async fn update(state: &AppState, kind: Change, change: impl FnOnce(&mut Vec<Entry>) -> Result<(), String>) -> Result<(), String> {
    refresh(state).await;
    let mut library = state.library.lock().await;
    if let Some(error) = &library.error {
        return Err(format!("{error} Fix models.yml, or save from the Library tab to replace it."));
    }
    let mut entries = library.entries.clone();
    change(&mut entries)?;
    let entries = validate(&entries)?;
    write(&state.data_dir, &mut library, entries, kind)
}

pub async fn set_aliases(state: &AppState, path: &str, aliases: Vec<String>) -> Result<Vec<String>, String> {
    let path = path.to_owned();
    let mut saved = Vec::new();
    update(state, Change::Edit, |entries| {
        if entries.iter().all(|e| e.path != path) {
            let entry = new_entry(&path, &taken_names(entries));
            entries.push(entry);
        }
        let entry = entries.iter_mut().find(|e| e.path == path).expect("entry exists");
        entry.aliases = normalize_aliases(&aliases)?.into_iter().filter(|a| *a != entry.name).collect();
        saved = entry.aliases.clone();
        Ok(())
    })
    .await?;
    Ok(saved)
}

async fn set_sha256(state: &AppState, path: &str, sha256: &str) -> Result<(), String> {
    update(state, Change::Checksums, |entries| match entries.iter_mut().find(|e| e.path == path) {
        Some(entry) => {
            entry.sha256 = sha256.to_owned();
            Ok(())
        }
        None => Err("This model has no entry in models.yml.".into()),
    })
    .await
}

// ---------------------------------------------------------------- sha256

#[derive(Clone, Serialize, Deserialize)]
struct Fingerprint {
    size: u64,
    modified: u128,
    sha256: String,
}

fn fingerprint(path: &str) -> Option<(u64, u128)> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?.duration_since(SystemTime::UNIX_EPOCH).ok()?.as_nanos();
    Some((meta.len(), modified))
}

fn read_cache(data_dir: &Path) -> BTreeMap<String, Fingerprint> {
    std::fs::read(data_dir.join(HASH_CACHE)).ok().and_then(|bytes| serde_json::from_slice(&bytes).ok()).unwrap_or_default()
}

fn hash_file(path: &str) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 4 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The file's sha256: from the cache while its size and modified time are
/// unchanged, otherwise read in full.
pub async fn current_hash(state: &AppState, path: &str, force: bool) -> Result<String, String> {
    let data_dir = state.data_dir.clone();
    let print = fingerprint(path).ok_or_else(|| format!("{path} cannot be read."))?;
    if !force {
        if let Some(cached) = read_cache(&data_dir).get(path) {
            if (cached.size, cached.modified) == print {
                return Ok(cached.sha256.clone());
            }
        }
    }
    state.hashing.lock().await.insert(path.to_owned());
    let owned = path.to_owned();
    let result = tokio::task::spawn_blocking(move || hash_file(&owned)).await;
    state.hashing.lock().await.remove(path);
    let sha256 = result.map_err(|e| e.to_string())?.map_err(|e| format!("Cannot read {path}: {e}"))?;
    let _guard = state.hash_cache_lock.lock().await;
    let mut cache = read_cache(&data_dir);
    cache.insert(path.to_owned(), Fingerprint { size: print.0, modified: print.1, sha256: sha256.clone() });
    cache.retain(|path, _| Path::new(path).is_file());
    if let Ok(bytes) = serde_json::to_vec_pretty(&cache) {
        let _ = tokio::fs::write(data_dir.join(HASH_CACHE), bytes).await;
    }
    Ok(sha256)
}

/// Fills in a missing sha256 for every entry whose file exists, one file at a
/// time in the background, unless checks are off.
pub fn hash_missing(state: AppState) {
    tokio::spawn(async move {
        if state.hash_worker.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        loop {
            if Check::parse(&state.settings.lock().await.sha256_check) == Check::Off {
                break;
            }
            let next = {
                let library = state.library.lock().await;
                let busy = state.hashing.lock().await.clone();
                library.entries.iter().find(|e| e.sha256.is_empty() && !busy.contains(&e.path) && Path::new(&e.path).is_file()).map(|e| e.path.clone())
            };
            let Some(path) = next else { break };
            match current_hash(&state, &path, false).await {
                Ok(sha256) => {
                    if set_sha256(&state, &path, &sha256).await.is_err() {
                        break;
                    }
                }
                Err(error) => {
                    state.log(format!("sha256 not recorded: {error}")).await;
                    break;
                }
            }
        }
        state.hash_worker.store(false, std::sync::atomic::Ordering::SeqCst);
    });
}

/// Checks a model's file against its recorded sha256 before it loads.
pub async fn verify(state: &AppState, path: &str, check: Check, log_prefix: &str) -> Result<(), String> {
    if check == Check::Off {
        return Ok(());
    }
    refresh(state).await;
    let expected = state.library.lock().await.entry(path).map(|e| e.sha256.clone()).unwrap_or_default();
    state.log(format!("{log_prefix}Checking the file's sha256…")).await;
    let actual = current_hash(state, path, check == Check::EveryLoad).await?;
    if expected.is_empty() {
        let _ = set_sha256(state, path, &actual).await;
        return Ok(());
    }
    if expected == actual {
        state.mismatches.lock().await.remove(path);
        return Ok(());
    }
    let message = format!("The file's sha256 ({}…) does not match the one in models.yml ({}…).", &actual[..12], &expected[..12.min(expected.len())]);
    state.mismatches.lock().await.insert(path.to_owned(), actual);
    if check == Check::Warn {
        state.log(format!("{log_prefix}Warning: {message} Loading anyway, as Settings allow.")).await;
        return Ok(());
    }
    Err(format!("{message} If you replaced the file on purpose, accept its new sha256 in Library; otherwise check the file."))
}

/// What the Library tab and the rest of the UI show.
pub async fn snapshot(state: &AppState) -> Value {
    refresh(state).await;
    let library = state.library.lock().await;
    let hashing = state.hashing.lock().await.clone();
    let mismatches = state.mismatches.lock().await.clone();
    let entries: Vec<Value> = library
        .entries
        .iter()
        .map(|e| json!({"name": e.name, "aliases": e.aliases, "path": e.path, "account": e.account, "sha256": e.sha256, "missing": !Path::new(&e.path).is_file(), "hashing": hashing.contains(&e.path), "mismatch": mismatches.get(&e.path)}))
        .collect();
    json!({"entries": entries, "error": library.error, "revision": library.revision, "file": file_path(&state.data_dir), "backups": backups_dir(&state.data_dir)})
}

// ---------------------------------------------------------------- commands

#[tauri::command]
pub async fn save_library(entries: Vec<Entry>, revision: u64, state: tauri::State<'_, AppState>) -> Result<(), String> {
    save(state.inner(), entries, revision).await
}

pub async fn save(state: &AppState, entries: Vec<Entry>, revision: u64) -> Result<(), String> {
    refresh(state).await;
    let entries = validate(&entries)?;
    {
        let mut library = state.library.lock().await;
        if library.revision != revision {
            return Err("models.yml changed since you started editing. Discard your changes to see the new version, then edit again.".into());
        }
        write(&state.data_dir, &mut library, entries, Change::Edit)?;
    }
    hash_missing(state.clone());
    Ok(())
}

#[tauri::command]
pub async fn reload_library(state: tauri::State<'_, AppState>) -> Result<(), String> {
    state.library.lock().await.stamp = None;
    refresh(state.inner()).await;
    Ok(())
}

#[tauri::command]
pub async fn open_models_file(state: tauri::State<'_, AppState>) -> Result<(), String> {
    refresh(state.inner()).await;
    tauri_plugin_opener::open_path(file_path(&state.data_dir), None::<&str>).map_err(|e| format!("Cannot open models.yml: {e}"))
}

#[tauri::command]
pub async fn open_backups_folder(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let dir = backups_dir(&state.data_dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    tauri_plugin_opener::open_path(dir, None::<&str>).map_err(|e| format!("Cannot open the backups folder: {e}"))
}

/// Reads the file now and records its sha256, replacing any recorded one.
#[tauri::command]
pub async fn record_sha256(path: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let sha256 = match state.mismatches.lock().await.get(&path).cloned() {
        Some(sha256) => sha256,
        None => current_hash(state.inner(), &path, true).await?,
    };
    set_sha256(state.inner(), &path, &sha256).await?;
    state.mismatches.lock().await.remove(&path);
    Ok(())
}

#[tauri::command]
pub async fn set_sha256_check(mode: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    if !["off", "once", "every_load", "warn"].contains(&mode.as_str()) {
        return Err("Unknown sha256 check.".into());
    }
    let mut settings = state.settings.lock().await.clone();
    settings.sha256_check = mode;
    crate::runtime::persist_settings(state.inner(), settings).await?;
    hash_missing(state.inner().clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, path: &str) -> Entry {
        Entry { name: name.into(), path: path.into(), ..Entry::default() }
    }

    #[test]
    fn accounts_come_from_download_folders() {
        assert_eq!(account_from_path("/m/LiquidAI--d1-3B-GGUF/d1-3B-Q8_0.gguf"), "LiquidAI");
        assert_eq!(account_from_path("/c/models--EldanRing--Winnow/snapshots/x/w.gguf"), "EldanRing");
        assert_eq!(account_from_path("/home/u/models/plain.gguf"), "");
    }

    #[test]
    fn names_and_aliases_are_unique_ignoring_case() {
        let mut a = entry("triage", "/m/a.gguf");
        a.aliases = vec!["fast".into(), "triage".into()];
        let mut b = entry("sorter", "/m/b.gguf");
        b.sha256 = "AB".repeat(32);
        let tidy = validate(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(tidy[0].aliases, ["fast"], "an alias equal to its own name is dropped");
        assert_eq!(tidy[1].sha256, "ab".repeat(32));
        b.aliases = vec!["FAST".into()];
        assert!(validate(&[a.clone(), b.clone()]).unwrap_err().contains("used by both triage and sorter"));
        assert!(validate(&[entry("x", "relative.gguf")]).is_err());
        assert!(validate(&[entry("x", "/m/a.gguf"), entry("y", "/m/a.gguf")]).unwrap_err().contains("listed twice"));
        assert!(validate(&[Entry { sha256: "nothex".into(), ..entry("x", "/m/a.gguf") }]).is_err());
    }

    #[test]
    fn the_file_round_trips_with_its_header() {
        let mut a = entry("d1-3B-Q8_0", "/m/LiquidAI--d1/d1-3B-Q8_0.gguf");
        a.aliases = vec!["triage".into()];
        a.account = "LiquidAI".into();
        let text = render(&[a.clone()]).unwrap();
        assert!(text.starts_with("# Noullama models"));
        assert_eq!(parse(&text).unwrap(), [a]);
        assert!(parse("models: [ {name: ").is_err());
    }

    #[test]
    fn edits_on_disk_are_picked_up_backed_up_and_errors_keep_the_last_good_version() {
        let dir = std::env::temp_dir().join(format!("noullama-library-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut library = Library::default();
        let legacy = BTreeMap::from([("/m/LiquidAI--d1/d1.gguf".to_owned(), vec!["triage".to_owned()])]);
        refresh_locked(&dir, &mut library, &legacy);
        assert_eq!(library.entries.len(), 1, "aliases saved by an earlier version move into models.yml");
        assert_eq!(library.entries[0].name, "d1");
        assert_eq!(library.entries[0].account, "LiquidAI");
        let first = library.revision;

        let edited = render(&[Entry { name: "renamed".into(), ..library.entries[0].clone() }]).unwrap();
        std::fs::write(file_path(&dir), &edited).unwrap();
        library.stamp = None;
        refresh_locked(&dir, &mut library, &legacy);
        assert_eq!(library.entries[0].name, "renamed");
        assert!(library.revision > first);

        std::fs::write(file_path(&dir), "models: [ {name: ").unwrap();
        library.stamp = None;
        refresh_locked(&dir, &mut library, &legacy);
        assert!(library.error.is_some());
        assert_eq!(library.entries[0].name, "renamed", "the last valid version stays in use");
        assert_eq!(std::fs::read_to_string(file_path(&dir)).unwrap(), "models: [ {name: ", "a broken file is not overwritten");

        let backups = std::fs::read_dir(backups_dir(&dir)).unwrap().count();
        assert!(backups >= 1, "the version before the edit was backed up");
        std::fs::remove_dir_all(dir).unwrap();
    }

    async fn temp_state() -> (AppState, PathBuf) {
        let dir = std::env::temp_dir().join(format!("noullama-library-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(dir.clone()).unwrap();
        (state, dir)
    }

    #[tokio::test]
    async fn recording_a_sha256_does_not_invalidate_an_edit_or_use_up_backups() {
        let (state, dir) = temp_state().await;
        let model = dir.join("acme--m-GGUF").join("m.gguf");
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"abc").unwrap();
        let model = model.to_str().unwrap().to_owned();
        ensure(&state, &[model.clone()]).await;
        let (revision, mut entries) = {
            let library = state.library.lock().await;
            (library.revision, library.entries.clone())
        };
        let backups = || std::fs::read_dir(backups_dir(&dir)).map(|d| d.count()).unwrap_or(0);
        let before = backups();
        // The background hasher records the sum while someone edits the entry.
        set_sha256(&state, &model, &"ab".repeat(32)).await.unwrap();
        assert_eq!(state.library.lock().await.revision, revision);
        assert_eq!(backups(), before, "a recorded sha256 is not worth a backup");
        entries[0].name = "renamed".into();
        entries[0].sha256 = "ab".repeat(32);
        save(&state, entries, revision).await.unwrap();
        let library = state.library.lock().await;
        assert_eq!((library.entries[0].name.as_str(), library.entries[0].sha256.as_str()), ("renamed", "ab".repeat(32).as_str()));
        drop(library);
        state.shutdown().await;
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn a_changed_file_stops_the_load_unless_settings_say_warn_or_off() {
        let (state, dir) = temp_state().await;
        let model = dir.join("m.gguf");
        std::fs::write(&model, b"abc").unwrap();
        let model = model.to_str().unwrap().to_owned();
        let right = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        ensure(&state, &[model.clone()]).await;
        // An empty sha256 is filled in on the first load.
        state.library.lock().await.entries[0].sha256.clear();
        verify(&state, &model, Check::Once, "").await.unwrap();
        assert_eq!(state.library.lock().await.entries[0].sha256, right);
        set_sha256(&state, &model, &"0".repeat(64)).await.unwrap();
        assert!(verify(&state, &model, Check::Once, "").await.unwrap_err().contains("does not match"));
        assert_eq!(state.mismatches.lock().await.get(&model).map(String::as_str), Some(right));
        verify(&state, &model, Check::Warn, "").await.unwrap();
        verify(&state, &model, Check::Off, "").await.unwrap();
        assert!(verify(&state, &model, Check::EveryLoad, "").await.is_err());
        set_sha256(&state, &model, right).await.unwrap();
        verify(&state, &model, Check::EveryLoad, "").await.unwrap();
        assert!(state.mismatches.lock().await.is_empty());
        state.shutdown().await;
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn hashes_match_sha256sum() {
        let dir = std::env::temp_dir().join(format!("noullama-hash-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("a.gguf");
        std::fs::write(&file, b"abc").unwrap();
        assert_eq!(hash_file(file.to_str().unwrap()).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
