use crate::runtime::{server_binary, AppState};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::Ordering,
    time::Duration,
};
use tokio::{io::AsyncWriteExt, sync::Mutex, task::JoinHandle};

const REPO_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp";
const MAX_ARCHIVE: u64 = 1_500_000_000;
const MAX_EXTRACTED: u64 = 8_000_000_000;

#[derive(Default)]
pub struct RuntimeManager {
    plan: Mutex<Option<RuntimePlan>>,
    pub progress: Mutex<InstallProgress>,
    pub task: Mutex<Option<JoinHandle<()>>>,
}

#[derive(Clone, Default, Serialize)]
pub struct InstallProgress {
    pub active: bool,
    pub message: String,
    pub downloaded: u64,
    pub total: u64,
    pub error: Option<String>,
    pub installed: Vec<LocalRuntime>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LocalRuntime {
    pub path: String,
    pub label: String,
    pub backend: Option<String>,
    pub source: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ReleaseAsset {
    pub name: String,
    pub size: u64,
    pub digest: String,
    pub url: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RuntimeBuild {
    pub release: String,
    pub tag: String,
    pub backend: String,
    pub assets: Vec<ReleaseAsset>,
}

#[derive(Clone, Serialize)]
pub struct RuntimePlan {
    pub id: String,
    pub platform: String,
    pub nightly: RuntimeBuild,
    pub fallback: RuntimeBuild,
    pub total_bytes: u64,
}

fn platform_arch(arch: &str) -> Result<&str, String> {
    match arch { "x86_64" => Ok("x64"), "aarch64" => Ok("arm64"), _ => Err(format!("No automatic runtime package is available for architecture {arch}. Use a custom build.")) }
}

fn asset_matches(name: &str, os: &str, arch: &str, backend: &str) -> bool {
    if !name.starts_with("llama-") || name.starts_with("cudart-") {
        return false;
    }
    let suffix = match (os, backend) {
        ("linux", "cpu") => format!("-bin-ubuntu-{arch}.tar.gz"),
        ("linux", "vulkan") => format!("-bin-ubuntu-vulkan-{arch}.tar.gz"),
        ("windows", "cpu") => format!("-bin-win-cpu-{arch}.zip"),
        ("windows", "vulkan") => format!("-bin-win-vulkan-{arch}.zip"),
        ("macos", "metal") => format!("-bin-macos-{arch}.tar.gz"),
        ("linux" | "windows", "cuda12" | "cuda13") => {
            let major = if backend == "cuda12" { "12" } else { "13" };
            let target = if os == "linux" { "ubuntu" } else { "win" };
            let ext = if os == "linux" { "tar.gz" } else { "zip" };
            let ext = regex::escape(ext);
            return regex::Regex::new(&format!(r"-bin-{target}-cuda-{major}\.\d+-{arch}\.{ext}$"))
                .is_ok_and(|pattern| pattern.is_match(name));
        }
        _ => return false,
    };
    name.ends_with(&suffix)
}

fn safe_tag(tag: &str) -> bool {
    !tag.is_empty()
        && tag.len() < 80
        && tag
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b".-_".contains(&c))
}

fn release_build(
    release: &Value,
    os: &str,
    arch: &str,
    backend: &str,
    label: String,
) -> Result<RuntimeBuild, String> {
    let tag = release["tag_name"]
        .as_str()
        .filter(|tag| safe_tag(tag))
        .ok_or("GitHub returned an invalid release tag")?;
    let assets = release["assets"]
        .as_array()
        .ok_or("GitHub returned no release assets")?;
    let asset = assets.iter().find(|asset| asset["name"].as_str().is_some_and(|name| asset_matches(name, os, arch, backend)))
        .ok_or_else(|| format!("{tag} has no official {os}/{arch} {backend} build. Choose another backend or a custom runtime."))?;
    let mut selected = vec![asset];
    if backend.starts_with("cuda") {
        let name = asset["name"].as_str().unwrap();
        let companion = if os == "windows" {
            format!("cudart-llama{}", &name[name.find("-bin-").unwrap()..])
        } else {
            format!("cudart-{name}")
        };
        let libraries = assets.iter().find(|asset| asset["name"] == companion).ok_or("This CUDA release has no matching CUDA runtime libraries; choose CPU or Vulkan instead")?;
        selected.push(libraries);
    }
    let assets = selected.into_iter().map(|asset| {
        let name = asset["name"].as_str().ok_or("Invalid asset name")?;
        if !name.bytes().all(|c| c.is_ascii_alphanumeric() || b".-_".contains(&c)) { return Err("Unsafe release asset name".into()); }
        let url = asset["browser_download_url"].as_str().ok_or("Asset has no download URL")?;
        let expected = format!("https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}");
        if url != expected { return Err("Release asset does not belong to the official llama.cpp repository".into()); }
        let digest = asset["digest"].as_str().filter(|digest| digest.strip_prefix("sha256:").is_some_and(|hash| hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()))).ok_or("GitHub did not provide a SHA-256 digest for this asset; installation requires a verified archive")?;
        let size = asset["size"].as_u64().filter(|size| *size > 0 && *size <= MAX_ARCHIVE).ok_or("Release archive exceeds the supported size")?;
        Ok(ReleaseAsset { name: name.into(), size, digest: digest.into(), url: url.into() })
    }).collect::<Result<Vec<_>, String>>()?;
    Ok(RuntimeBuild {
        release: label,
        tag: tag.into(),
        backend: backend.into(),
        assets,
    })
}

async fn github_json(state: &AppState, path: &str) -> Result<Value, String> {
    let response = state
        .client
        .get(format!("{REPO_API}/{path}"))
        .header("User-Agent", "Noullama/0.1")
        .header("Accept", "application/vnd.github+json")
        .timeout(Duration::from_secs(25))
        .send()
        .await
        .map_err(|e| format!("Cannot contact GitHub: {e}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "GitHub returned {}. Its anonymous API may be rate limited; try again later.",
            response.status()
        ));
    }
    response
        .json()
        .await
        .map_err(|e| format!("Invalid GitHub response: {e}"))
}

#[tauri::command]
pub async fn runtime_releases(
    backend: String,
    state: tauri::State<'_, AppState>,
) -> Result<RuntimePlan, String> {
    plan_releases(state.inner(), &backend).await
}

async fn plan_releases(state: &AppState, backend: &str) -> Result<RuntimePlan, String> {
    let os = std::env::consts::OS;
    let arch = platform_arch(std::env::consts::ARCH)?;
    let (releases, stable) = tokio::try_join!(
        github_json(&state, "releases?per_page=30"),
        github_json(&state, "releases/latest")
    )?;
    let nightly = releases
        .as_array()
        .and_then(|releases| {
            releases.iter().find(|r| {
                !r["draft"].as_bool().unwrap_or(true)
                    && r["tag_name"].as_str().is_some_and(|tag| {
                        tag.starts_with('b') && tag[1..].bytes().all(|b| b.is_ascii_digit())
                    })
            })
        })
        .ok_or("No official nightly release was found")?;
    let nightly = release_build(nightly, os, arch, backend, "Nightly".into())?;
    // Versioned releases currently link their binary build in the official notes.
    let stable_tag = stable["tag_name"]
        .as_str()
        .ok_or("Official release has no tag")?
        .to_owned();
    let fallback_backend = if os == "macos" { "metal" } else { "cpu" };
    let fallback = match release_build(&stable, os, arch, fallback_backend, stable_tag.clone()) {
        Ok(build) => build,
        Err(_) => {
            let body = stable["body"].as_str().unwrap_or_default();
            let pattern =
                regex::Regex::new(r"https://github\.com/ggml-org/llama\.cpp/releases/tag/(b\d+)")
                    .unwrap();
            let tag = pattern
                .captures(body)
                .and_then(|c| c.get(1))
                .ok_or("Official release has no platform binaries or linked binary build")?
                .as_str();
            let linked = github_json(&state, &format!("releases/tags/{tag}")).await?;
            release_build(
                &linked,
                os,
                arch,
                fallback_backend,
                format!("Official {stable_tag}"),
            )?
        }
    };
    let total_bytes = nightly
        .assets
        .iter()
        .chain(&fallback.assets)
        .map(|a| a.size)
        .sum();
    let plan = RuntimePlan {
        id: uuid::Uuid::new_v4().to_string(),
        platform: format!("{os} / {}", std::env::consts::ARCH),
        nightly,
        fallback,
        total_bytes,
    };
    *state.runtimes.plan.lock().await = Some(plan.clone());
    Ok(plan)
}

#[tauri::command]
pub async fn discover_runtimes(state: tauri::State<'_, AppState>) -> Result<Value, String> {
    let settings = state.settings.lock().await.clone();
    let managed = state.data_dir.join("runtimes");
    tokio::task::spawn_blocking(move || {
        let mut roots = vec![(managed, 4, "managed")];
        let binary = server_binary(&settings.runtime_path);
        if let Some(parent) = binary.parent().filter(|p| !p.as_os_str().is_empty()) {
            roots.push((parent.to_path_buf(), 2, "custom"));
            if let Some(parent) = parent.parent().filter(|p| p.parent().is_some()) { roots.push((parent.to_path_buf(), 3, "near custom runtime")); }
        }
        if let Some(paths) = std::env::var_os("PATH") { roots.extend(std::env::split_paths(&paths).filter(|p| p.is_absolute()).map(|p| (p, 0, "PATH"))); }
        let (local, warnings) = scan_runtimes(roots, 12_000);
        json!({"platform":format!("{} / {}",std::env::consts::OS,std::env::consts::ARCH), "backends": if cfg!(target_os="macos") { vec!["metal"] } else { vec!["cpu","vulkan","cuda12","cuda13"] }, "local":local,"warnings":warnings})
    }).await.map_err(|e| e.to_string())
}

fn scan_runtimes(
    roots: Vec<(PathBuf, usize, &str)>,
    limit: usize,
) -> (Vec<LocalRuntime>, Vec<String>) {
    let started = std::time::Instant::now();
    let mut pending = roots;
    let mut seen = HashSet::new();
    let mut found = Vec::new();
    let mut warnings = Vec::new();
    let mut count = 0;
    'scan: while let Some((path, depth, source)) = pending.pop() {
        if count >= limit || started.elapsed() > Duration::from_secs(2) {
            warnings.push(
                "Runtime scan reached its limit; use a custom path to check another directory."
                    .into(),
            );
            break;
        }
        let Ok(path) = std::fs::canonicalize(&path) else {
            continue;
        };
        if !seen.insert((path.clone(), depth)) {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            count += 1;
            if count >= limit || started.elapsed() > Duration::from_secs(2) {
                warnings.push(
                    "Runtime scan reached its limit; use a custom path to check another directory."
                        .into(),
                );
                break 'scan;
            }
            let name = entry.file_name();
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() && depth > 0 {
                pending.push((entry.path(), depth - 1, source));
            } else if (kind.is_file() || kind.is_symlink())
                && (name == "llama-server" || name == "llama-server.exe")
                && entry.path().is_file()
            {
                let label = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
                let manifest = std::fs::read(path.join("noullama-runtime.json"))
                    .ok()
                    .and_then(|bytes| serde_json::from_slice::<RuntimeBuild>(&bytes).ok());
                found.push(LocalRuntime {
                    path: entry.path().to_string_lossy().into_owned(),
                    label: manifest
                        .as_ref()
                        .map(|m| format!("{} · {} · {}", m.release, m.tag, m.backend))
                        .unwrap_or(label),
                    backend: manifest.map(|m| m.backend),
                    source: source.into(),
                });
            }
        }
    }
    found.sort_by(|a, b| a.path.cmp(&b.path));
    found.dedup_by(|a, b| a.path == b.path);
    (found, warnings)
}

fn safe_relative(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && path
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
        && !path.to_string_lossy().contains(['\\', ':'])
}

fn extract_archive(
    archive: &Path,
    destination: &Path,
    exiting: &crate::lifecycle::AppControl,
) -> Result<(), String> {
    let mut links = Vec::new();
    let mut total = 0u64;
    let mut count = 0;
    let mut copy_entry =
        |path: &Path, size: u64, directory: bool, input: &mut dyn Read| -> Result<(), String> {
            count += 1;
            total = total.checked_add(size).ok_or("Archive size overflow")?;
            if exiting.exiting.load(Ordering::SeqCst) {
                return Err("Runtime installation stopped because Noullama is exiting".into());
            }
            if count > 20_000 || total > MAX_EXTRACTED || !safe_relative(path) {
                return Err("Archive contains an unsafe path or exceeds extraction limits".into());
            }
            let target = destination.join(path);
            if directory {
                std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
                return Ok(());
            }
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let mut file = match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&target)
            {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if !std::fs::symlink_metadata(&target)
                        .map_err(|e| e.to_string())?
                        .file_type()
                        .is_file()
                    {
                        return Err("Archive entry collides with a non-file".into());
                    }
                    let mut existing = std::fs::File::open(&target).map_err(|e| e.to_string())?;
                    if existing.metadata().map_err(|e| e.to_string())?.len() != size {
                        return Err("Runtime archives contain conflicting files".into());
                    }
                    let mut left = [0u8; 64 * 1024];
                    let mut right = [0u8; 64 * 1024];
                    let mut read = 0u64;
                    loop {
                        if exiting.exiting.load(Ordering::SeqCst) {
                            return Err("Runtime installation cancelled".into());
                        }
                        let n = input.read(&mut left).map_err(|e| e.to_string())?;
                        if n == 0 {
                            break;
                        }
                        read += n as u64;
                        if read > size {
                            return Err("Archive entry exceeds its declared size".into());
                        }
                        existing
                            .read_exact(&mut right[..n])
                            .map_err(|e| e.to_string())?;
                        if left[..n] != right[..n] {
                            return Err("Runtime archives contain conflicting files".into());
                        }
                    }
                    if read != size {
                        return Err("Truncated duplicate archive entry".into());
                    }
                    return Ok(());
                }
                Err(error) => return Err(format!("Cannot create extracted file: {error}")),
            };
            let mut buffer = [0u8; 64 * 1024];
            let mut written = 0u64;
            loop {
                if exiting.exiting.load(Ordering::SeqCst) {
                    return Err("Runtime installation cancelled".into());
                }
                let n = input.read(&mut buffer).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                written += n as u64;
                if written > size {
                    return Err("Archive entry exceeds declared size".into());
                }
                file.write_all(&buffer[..n]).map_err(|e| e.to_string())?;
            }
            if written != size {
                return Err("Truncated archive entry".into());
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(
                    &target,
                    std::fs::Permissions::from_mode(
                        if path.file_name().is_some_and(|n| {
                            n == "llama-server" || n.to_string_lossy().starts_with("llama-")
                        }) {
                            0o755
                        } else {
                            0o644
                        },
                    ),
                )
                .map_err(|e| e.to_string())?;
            }
            Ok(())
        };
    let file = std::fs::File::open(archive).map_err(|e| e.to_string())?;
    if archive.to_string_lossy().ends_with(".tar.gz") {
        let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(file));
        for entry in archive.entries().map_err(|e| e.to_string())? {
            let mut entry = entry.map_err(|e| e.to_string())?;
            let kind = entry.header().entry_type();
            let path = entry.path().map_err(|e| e.to_string())?.into_owned();
            if !safe_relative(&path) {
                return Err("Runtime archive path escapes the installation".into());
            }
            if kind.is_symlink() || kind.is_hard_link() {
                let target = entry
                    .link_name()
                    .map_err(|e| e.to_string())?
                    .ok_or("Archive link has no target")?
                    .into_owned();
                if target.is_absolute() || target.to_string_lossy().contains(['\\', ':']) {
                    return Err("Archive link has an unsafe target".into());
                }
                let target = if kind.is_symlink() {
                    path.parent().unwrap_or(Path::new("")).join(target)
                } else {
                    target
                };
                links.push((path, target));
                continue;
            }
            if !kind.is_file() && !kind.is_dir() {
                return Err(
                    "Runtime archive contains links or special files; refusing extraction".into(),
                );
            }
            let size = entry.size();
            copy_entry(&path, size, kind.is_dir(), &mut entry)?;
        }
    } else if archive.to_string_lossy().ends_with(".zip") {
        let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).map_err(|e| e.to_string())?;
            let path = entry
                .enclosed_name()
                .ok_or("ZIP entry escapes the runtime directory")?
                .to_path_buf();
            if entry
                .unix_mode()
                .is_some_and(|mode| mode & 0o170000 == 0o120000)
            {
                return Err("Runtime ZIP contains a symbolic link".into());
            }
            let size = entry.size();
            let directory = entry.is_dir();
            copy_entry(&path, size, directory, &mut entry)?;
        }
    } else {
        return Err("Unsupported runtime archive format".into());
    }
    // Resolve versioned library links only after all regular files are present.
    // Materialize them as ordinary files; no archive-created directory symlink
    // can redirect extraction of a later entry or CUDA companion archive.
    let root = std::fs::canonicalize(destination).map_err(|e| e.to_string())?;
    let mut passes = 0;
    if links.len() > 512 {
        return Err("Too many archive library links".into());
    }
    while !links.is_empty() {
        passes += 1;
        if passes > 32 {
            return Err("Archive library link chain is too deep".into());
        }
        let mut pending = Vec::new();
        let mut resolved = false;
        for (path, target) in links {
            match std::fs::canonicalize(destination.join(&target)) {
                Ok(source) => {
                    if !source.starts_with(&root) || !source.is_file() {
                        return Err(
                            "Archive link points outside the installation or to a directory".into(),
                        );
                    }
                    let mut file = std::fs::File::open(&source).map_err(|e| e.to_string())?;
                    let size = file.metadata().map_err(|e| e.to_string())?.len();
                    copy_entry(&path, size, false, &mut file)?;
                    resolved = true;
                }
                Err(_) => pending.push((path, target)),
            }
        }
        if !resolved {
            return Err("Runtime archive contains unresolved or cyclic library links".into());
        }
        links = pending;
    }
    Ok(())
}

async fn fetch_asset(state: &AppState, asset: &ReleaseAsset, target: &Path) -> Result<(), String> {
    let mut stop = state.app.api_stop.subscribe();
    let request = state
        .client
        .get(&asset.url)
        .header("User-Agent", "Noullama/0.1")
        .timeout(Duration::from_secs(900))
        .send();
    let mut response = tokio::select! { r=request => r.map_err(|e|e.to_string())?, _=stop.wait_for(|s|*s) => return Err("Runtime download cancelled".into()) };
    if !response.status().is_success() {
        return Err(format!("Runtime download returned {}", response.status()));
    }
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target)
        .await
        .map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    loop {
        let chunk = tokio::select! { r=response.chunk()=>r.map_err(|e|e.to_string())?, _=stop.wait_for(|s|*s)=>return Err("Runtime download cancelled".into()) };
        let Some(chunk) = chunk else {
            break;
        };
        bytes += chunk.len() as u64;
        if bytes > asset.size || bytes > MAX_ARCHIVE {
            return Err("Runtime archive is larger than advertised".into());
        }
        file.write_all(&chunk).await.map_err(|e| e.to_string())?;
        hash.update(&chunk);
        state.runtimes.progress.lock().await.downloaded += chunk.len() as u64;
    }
    file.flush().await.map_err(|e| e.to_string())?;
    if bytes != asset.size || format!("sha256:{:x}", hash.finalize()) != asset.digest.to_lowercase()
    {
        return Err(
            "Runtime archive failed size/SHA-256 verification; it was not installed".into(),
        );
    }
    Ok(())
}

fn find_server(directory: &Path) -> Option<PathBuf> {
    scan_runtimes(vec![(directory.to_path_buf(), 5, "managed")], 20_000)
        .0
        .into_iter()
        .next()
        .map(|r| PathBuf::from(r.path))
}

async fn install_build(state: &AppState, build: &RuntimeBuild) -> Result<LocalRuntime, String> {
    let root = state.data_dir.join("runtimes");
    tokio::fs::create_dir_all(&root)
        .await
        .map_err(|e| e.to_string())?;
    let destination = root.join(format!("{}-{}", build.tag, build.backend));
    if destination.join("noullama-install.json").is_file() {
        let installed: RuntimeBuild = serde_json::from_slice(
            &tokio::fs::read(destination.join("noullama-install.json"))
                .await
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if installed
            .assets
            .iter()
            .map(|a| &a.digest)
            .eq(build.assets.iter().map(|a| &a.digest))
        {
            if let Some(server) = find_server(&destination) {
                state.runtimes.progress.lock().await.downloaded +=
                    build.assets.iter().map(|a| a.size).sum::<u64>();
                return Ok(LocalRuntime {
                    path: server.to_string_lossy().into_owned(),
                    label: format!("{} · {}", build.release, build.tag),
                    backend: Some(build.backend.clone()),
                    source: "managed".into(),
                });
            }
        }
        return Err("A managed runtime already exists with different metadata; select it locally or use another release".into());
    }
    if destination.exists() {
        return Err("Runtime destination already exists without a completed installation; it was left untouched".into());
    }
    let staging = root.join(format!(".{}-partial", uuid::Uuid::new_v4()));
    tokio::fs::create_dir(&staging)
        .await
        .map_err(|e| e.to_string())?;
    let result = async {
        let extracted = staging.join("files");
        tokio::fs::create_dir(&extracted)
            .await
            .map_err(|e| e.to_string())?;
        for (index, asset) in build.assets.iter().enumerate() {
            state.runtimes.progress.lock().await.message = format!("Downloading {}", asset.name);
            let archive = staging.join(&asset.name);
            fetch_asset(state, asset, &archive).await?;
            state.runtimes.progress.lock().await.message =
                format!("Verifying and extracting {}", asset.name);
            let package = extracted.join(format!("package-{index}"));
            tokio::fs::create_dir(&package)
                .await
                .map_err(|e| e.to_string())?;
            let (input, output, app) = (archive.clone(), package, state.app.clone());
            tokio::task::spawn_blocking(move || extract_archive(&input, &output, &app))
                .await
                .map_err(|e| e.to_string())??;
        }
        let server =
            find_server(&extracted).ok_or("Downloaded archive does not contain llama-server")?;
        // CUDA companions sometimes unpack into a separate sibling directory.
        // Keep them beside the executable so its normal library search finds them.
        let server_dir = server.parent().ok_or("Invalid runtime layout")?;
        for entry in walk_library_files(&extracted) {
            if entry.parent() != Some(server_dir) {
                let name = entry.file_name().unwrap();
                let target = server_dir.join(name);
                if !target.exists() {
                    std::fs::copy(&entry, &target).map_err(|e| e.to_string())?;
                }
            }
        }
        let bytes = serde_json::to_vec_pretty(build).map_err(|e| e.to_string())?;
        tokio::fs::write(server_dir.join("noullama-runtime.json"), &bytes)
            .await
            .map_err(|e| e.to_string())?;
        tokio::fs::write(extracted.join("noullama-install.json"), bytes)
            .await
            .map_err(|e| e.to_string())?;
        if state.app.exiting.load(Ordering::SeqCst) {
            return Err("Runtime installation cancelled".into());
        }
        let relative = server
            .strip_prefix(&extracted)
            .map_err(|e| e.to_string())?
            .to_path_buf();
        tokio::fs::rename(&extracted, &destination)
            .await
            .map_err(|e| e.to_string())?;
        Ok(LocalRuntime {
            path: destination.join(relative).to_string_lossy().into_owned(),
            label: format!("{} · {}", build.release, build.tag),
            backend: Some(build.backend.clone()),
            source: "managed".into(),
        })
    }
    .await;
    let _ = tokio::fs::remove_dir_all(&staging).await;
    result
}

fn walk_library_files(root: &Path) -> Vec<PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    let mut entries = 0;
    while let Some(dir) = pending.pop() {
        if entries > 20_000 {
            break;
        }
        if let Ok(items) = std::fs::read_dir(dir) {
            for item in items.flatten() {
                entries += 1;
                let path = item.path();
                if path.is_dir() {
                    pending.push(path);
                } else if path.file_name().is_some_and(|n| {
                    n.to_string_lossy().contains(".so")
                        || n.to_string_lossy().ends_with(".dll")
                        || n.to_string_lossy().ends_with(".dylib")
                }) {
                    files.push(path);
                }
            }
        }
    }
    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    #[test]
    fn zip_installation_extracts_windows_files_and_rejects_traversal() {
        let base = directory();
        let out = base.join("out");
        std::fs::create_dir(&out).unwrap();
        for (name, valid) in [("build/llama-server.exe", true), ("../escaped.exe", false)] {
            let archive = base.join(if valid { "good.zip" } else { "bad.zip" });
            let mut zip = zip::ZipWriter::new(std::fs::File::create(&archive).unwrap());
            zip.start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zip.write_all(b"test executable").unwrap();
            zip.finish().unwrap();
            let result = extract_archive(&archive, &out, &crate::lifecycle::AppControl::default());
            assert_eq!(result.is_ok(), valid);
            assert!(!base.join("escaped.exe").exists());
        }
        std::fs::remove_dir_all(base).unwrap();
    }

    #[tokio::test]
    async fn approved_pair_download_is_verified_atomic_cached_and_does_not_change_custom_path() {
        use axum::{body::Body, response::Response, routing::get, Router};
        let base = directory();
        let mut state = AppState::new(base.join("data")).unwrap();
        state.client = state.local_client.clone();
        let archive = base.join("fixture.tar.gz");
        let gzip = flate2::write::GzEncoder::new(
            std::fs::File::create(&archive).unwrap(),
            flate2::Compression::default(),
        );
        let mut tar = tar::Builder::new(gzip);
        let data = b"fixture only; never executed";
        let mut header = tar::Header::new_gnu();
        header.set_mode(0o755);
        header.set_size(data.len() as u64);
        header.set_cksum();
        tar.append_data(&mut header, "build/llama-server", &data[..])
            .unwrap();
        tar.into_inner().unwrap().finish().unwrap();
        let bytes = std::fs::read(&archive).unwrap();
        let digest = format!("sha256:{:x}", Sha256::digest(&bytes));
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = calls.clone();
        let fixture = bytes.clone();
        let router = Router::new().route(
            "/archive",
            get(move || {
                let data = fixture.clone();
                let calls = counter.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Response::new(Body::from(data))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/archive", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        let build = |tag: &str| RuntimeBuild {
            release: "Fixture".into(),
            tag: tag.into(),
            backend: "cpu".into(),
            assets: vec![ReleaseAsset {
                name: format!("llama-{tag}-bin-ubuntu-x64.tar.gz"),
                size: bytes.len() as u64,
                digest: digest.clone(),
                url: url.clone(),
            }],
        };
        let plan = RuntimePlan {
            id: "approved".into(),
            platform: "test".into(),
            nightly: build("btest1"),
            fallback: build("btest2"),
            total_bytes: bytes.len() as u64 * 2,
        };
        *state.runtimes.plan.lock().await = Some(plan.clone());
        assert!(!state.data_dir.join("runtimes").exists());
        assert!(begin_runtime_download(&state, "unapproved").await.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        for _ in 0..2 {
            begin_runtime_download(&state, "approved").await.unwrap();
            let task = state.runtimes.task.lock().await.take().unwrap();
            tokio::time::timeout(Duration::from_secs(5), task)
                .await
                .unwrap()
                .unwrap();
            let progress = state.runtimes.progress.lock().await.clone();
            assert!(!progress.active);
            assert!(progress.error.is_none(), "{:?}", progress.error);
            assert_eq!(progress.installed.len(), 2);
            for runtime in progress.installed {
                assert!(Path::new(&runtime.path).is_file());
            }
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "Existing verified installs must be reused"
        );
        assert!(
            state.settings.lock().await.runtime_path.is_empty(),
            "Installing a runtime must not silently change the selected path"
        );
        let mut bad = plan;
        bad.id = "corrupt".into();
        bad.nightly = build("btest3");
        bad.nightly.assets[0].digest = format!("sha256:{}", "0".repeat(64));
        *state.runtimes.plan.lock().await = Some(bad);
        begin_runtime_download(&state, "corrupt").await.unwrap();
        let task = state.runtimes.task.lock().await.take().unwrap();
        task.await.unwrap();
        assert!(state
            .runtimes
            .progress
            .lock()
            .await
            .error
            .as_ref()
            .unwrap()
            .contains("SHA-256"));
        assert!(!state.data_dir.join("runtimes/btest3-cpu").exists());
        server.abort();
        let _ = server.await;
        std::fs::remove_dir_all(base).unwrap();
    }
    #[tokio::test]
    #[ignore = "reads live GitHub metadata only; does not download binaries"]
    async fn official_release_plan_resolves_nightly_and_versioned_fallback_without_downloading() {
        let base = directory();
        let state = AppState::new(base.clone()).unwrap();
        let backend = if cfg!(target_os = "macos") {
            "metal"
        } else {
            "cpu"
        };
        let plan = plan_releases(&state, backend).await.unwrap();
        assert!(plan.nightly.tag.starts_with('b'));
        assert!(plan.fallback.release.contains('v'));
        assert!(!plan.nightly.assets.is_empty() && !plan.fallback.assets.is_empty());
        assert!(!base.join("runtimes").exists());
        assert!(!state.runtimes.progress.lock().await.active);
        println!(
            "Metadata only: {} {}, fallback {} {}, {} bytes; no runtime downloaded",
            plan.platform,
            plan.nightly.tag,
            plan.fallback.release,
            plan.fallback.tag,
            plan.total_bytes
        );
        std::fs::remove_dir_all(base).unwrap();
    }
    fn directory() -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("noullama-runtime-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }
    #[test]
    fn platform_assets_do_not_match_other_operating_systems_or_cuda_libraries() {
        assert!(asset_matches(
            "llama-b11968-bin-ubuntu-x64.tar.gz",
            "linux",
            "x64",
            "cpu"
        ));
        assert!(asset_matches(
            "llama-b11968-bin-ubuntu-cuda-13.4-x64.tar.gz",
            "linux",
            "x64",
            "cuda13"
        ));
        assert!(asset_matches(
            "llama-b11968-bin-win-cpu-arm64.zip",
            "windows",
            "arm64",
            "cpu"
        ));
        assert!(asset_matches(
            "llama-b11968-bin-macos-arm64.tar.gz",
            "macos",
            "arm64",
            "metal"
        ));
        assert!(!asset_matches(
            "cudart-llama-b11968-bin-ubuntu-cuda-13.4-x64.tar.gz",
            "linux",
            "x64",
            "cuda13"
        ));
        assert!(!asset_matches(
            "llama-b11968-bin-ubuntu-cuda-13.4-arm64.tar.gz",
            "linux",
            "x64",
            "cuda13"
        ));
        assert!(!asset_matches(
            "llama-b11968-bin-ubuntu-x64.tar.gz",
            "windows",
            "x64",
            "cpu"
        ));
        assert!(platform_arch("riscv64").is_err());
    }
    #[test]
    fn release_assets_require_official_urls_and_hashes() {
        let name = "llama-b11968-bin-ubuntu-x64.tar.gz";
        let mut release = json!({"tag_name":"b11968","assets":[{"name":name,"size":123,"digest":format!("sha256:{}","a".repeat(64)),"browser_download_url":format!("https://github.com/ggml-org/llama.cpp/releases/download/b11968/{name}")}]});
        assert!(release_build(&release, "linux", "x64", "cpu", "Nightly".into()).is_ok());
        release["assets"][0]["browser_download_url"] = json!("https://evil.example/build.tar.gz");
        assert!(release_build(&release, "linux", "x64", "cpu", "Nightly".into()).is_err());
        release["assets"][0]["browser_download_url"] = json!(format!(
            "https://github.com/ggml-org/llama.cpp/releases/download/b11968/{name}"
        ));
        release["assets"][0]["digest"] = Value::Null;
        assert!(release_build(&release, "linux", "x64", "cpu", "Nightly".into()).is_err());
    }
    #[test]
    fn tar_libraries_are_materialized_without_creating_symlinks() {
        let base = directory();
        let archive = base.join("build.tar.gz");
        let out = base.join("out");
        std::fs::create_dir(&out).unwrap();
        let gzip = flate2::write::GzEncoder::new(
            std::fs::File::create(&archive).unwrap(),
            flate2::Compression::default(),
        );
        let mut builder = tar::Builder::new(gzip);
        for (name, contents) in [
            ("build/llama-server", b"server".as_slice()),
            ("build/libllama.so.0.8.0", b"library".as_slice()),
        ] {
            let mut header = tar::Header::new_gnu();
            header.set_mode(0o755);
            header.set_size(contents.len() as u64);
            header.set_cksum();
            builder.append_data(&mut header, name, contents).unwrap();
        }
        for (name, target) in [
            ("build/libllama.so", "libllama.so.0"),
            ("build/libllama.so.0", "libllama.so.0.8.0"),
        ] {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(tar::EntryType::Symlink);
            header.set_size(0);
            header.set_mode(0o777);
            builder.append_link(&mut header, name, target).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap();
        extract_archive(&archive, &out, &crate::lifecycle::AppControl::default()).unwrap();
        assert_eq!(
            std::fs::read(out.join("build/libllama.so")).unwrap(),
            b"library"
        );
        assert!(!std::fs::symlink_metadata(out.join("build/libllama.so"))
            .unwrap()
            .file_type()
            .is_symlink());
        assert!(find_server(&out).is_some());
        std::fs::remove_dir_all(base).unwrap();
    }
    #[test]
    fn archive_paths_cannot_escape_the_installation() {
        assert!(!safe_relative(Path::new("../escape")));
        assert!(!safe_relative(Path::new("/absolute")));
        assert!(!safe_relative(Path::new("C:\\escape")));
        assert!(safe_relative(Path::new("llama-build/bin/llama-server")));
    }
    #[test]
    fn local_scan_finds_existing_builds_without_running_or_modifying_them() {
        let base = directory();
        let bin = base.join("llama-b11954");
        std::fs::create_dir(&bin).unwrap();
        std::fs::write(bin.join("llama-server"), b"not executed").unwrap();
        let (found, _) = scan_runtimes(vec![(base.clone(), 3, "local")], 100);
        assert_eq!(found.len(), 1);
        assert!(found[0].path.ends_with("llama-server"));
        assert_eq!(
            std::fs::read(bin.join("llama-server")).unwrap(),
            b"not executed"
        );
        std::fs::remove_dir_all(base).unwrap();
    }
}

#[tauri::command]
pub async fn download_runtimes(
    plan_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    begin_runtime_download(state.inner(), &plan_id).await
}

async fn begin_runtime_download(state: &AppState, plan_id: &str) -> Result<(), String> {
    let _guard = state.lifecycle.lock().await;
    if state.app.exiting.load(Ordering::SeqCst) {
        return Err("Noullama is shutting down".into());
    }
    let mut progress = state.runtimes.progress.lock().await;
    if progress.active {
        return Err("A runtime download is already active".into());
    }
    let plan = state
        .runtimes
        .plan
        .lock()
        .await
        .clone()
        .filter(|p| p.id == plan_id)
        .ok_or("Check releases again before approving this download")?;
    *progress = InstallProgress {
        active: true,
        message: "Preparing the approved runtime downloads…".into(),
        total: plan.total_bytes,
        ..Default::default()
    };
    drop(progress);
    let worker = state.clone();
    let task = tokio::spawn(async move {
        let result = async {
            for build in [&plan.nightly, &plan.fallback] {
                let installed = install_build(&worker, build).await?;
                worker
                    .runtimes
                    .progress
                    .lock()
                    .await
                    .installed
                    .push(installed);
            }
            Ok::<_, String>(())
        }
        .await;
        let mut progress = worker.runtimes.progress.lock().await;
        progress.active = false;
        match result {
            Ok(()) => {
                progress.message =
                    "Nightly and official fallback are ready. Choose which runtime to use.".into()
            }
            Err(error) => {
                progress.message = "Runtime installation did not finish.".into();
                progress.error = Some(error);
            }
        }
    });
    *state.runtimes.task.lock().await = Some(task);
    Ok(())
}
