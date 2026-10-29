use semver::Version;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, sync::Arc, time::Duration};
use tauri::{utils::config::BundleType, AppHandle, Manager};
use tauri_plugin_updater::{Update, UpdaterExt};
use tokio::sync::Mutex;

const RELEASES_URL: &str = "https://api.github.com/repos/parasite6/noullama/releases?per_page=30";
const MANIFEST_NAME: &str = "latest.json";
/// Points the check at one manifest instead of GitHub, for testing updates
/// against a local server. The package signature is still verified.
const MANIFEST_OVERRIDE: &str = "NOULLAMA_UPDATE_URL";
/// rpm rejects semver pre-release versions, so a pre-release keeps a plain app
/// version and is marked by a tag suffix (`v0.3.0-pre`, `v0.3.0-nightly`).
const PRERELEASE_SUFFIXES: [&str; 2] = ["-pre", "-nightly"];
/// Set by the release workflow when it builds a pre-release.
const BUILT_AS_PRERELEASE: bool = option_env!("NOULLAMA_PRERELEASE").is_some();

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    /// `None` follows the installed build: pre-release builds get pre-releases.
    pub include_prereleases: Option<bool>,
    pub check_on_launch: bool,
}
impl Default for Prefs {
    fn default() -> Self {
        Self {
            include_prereleases: None,
            check_on_launch: true,
        }
    }
}

#[derive(Clone, Serialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Status {
    #[default]
    Idle,
    Checking,
    UpToDate,
    Available,
    Downloading {
        received: u64,
        total: Option<u64>,
    },
    Installing,
    Error {
        message: String,
    },
}

#[derive(Clone, Serialize)]
pub struct Offer {
    pub version: String,
    pub notes: String,
    pub page: String,
    pub prerelease: bool,
}

#[derive(Default)]
struct Inner {
    status: Status,
    offer: Option<Offer>,
    pending: Option<Update>,
    checked_at: Option<u64>,
}

#[derive(Clone)]
pub struct Updates {
    path: PathBuf,
    prefs: Arc<Mutex<Prefs>>,
    inner: Arc<Mutex<Inner>>,
}

#[derive(Serialize)]
pub struct View {
    current_version: String,
    include_prereleases: bool,
    check_on_launch: bool,
    /// Why this copy cannot install updates itself, if it cannot.
    unsupported: Option<String>,
    status: Status,
    offer: Option<Offer>,
    checked_at: Option<u64>,
}

#[derive(Deserialize)]
struct GithubRelease {
    tag_name: String,
    html_url: String,
    #[serde(default)]
    body: Option<String>,
    draft: bool,
    prerelease: bool,
    assets: Vec<GithubAsset>,
}

#[derive(Deserialize)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
}

struct Candidate {
    version: Version,
    manifest: String,
    offer: Offer,
}

/// The newest published release above `current` that carries an update
/// manifest. Pre-releases count only when `include_prereleases` is set.
fn pick_release(
    releases: Vec<GithubRelease>,
    current: &Version,
    include_prereleases: bool,
) -> Option<Candidate> {
    releases
        .into_iter()
        .filter(|release| !release.draft)
        .filter_map(|release| {
            let tag = release.tag_name.trim_start_matches('v');
            let plain = PRERELEASE_SUFFIXES
                .iter()
                .find_map(|suffix| tag.strip_suffix(suffix));
            let marked = plain.is_some();
            let version = Version::parse(plain.unwrap_or(tag)).ok()?;
            let prerelease = release.prerelease || marked || !version.pre.is_empty();
            if prerelease && !include_prereleases {
                return None;
            }
            let manifest = release
                .assets
                .iter()
                .find(|asset| asset.name == MANIFEST_NAME)?
                .browser_download_url
                .clone();
            Some(Candidate {
                offer: Offer {
                    version: version.to_string(),
                    notes: release.body.unwrap_or_default(),
                    page: release.html_url,
                    prerelease,
                },
                version,
                manifest,
            })
        })
        .filter(|candidate| candidate.version > *current)
        .max_by(|a, b| a.version.cmp(&b.version))
}

/// Only the packages the release ships can be replaced in place; a source
/// build or `tauri dev` has nothing the updater could safely overwrite.
fn unsupported_reason() -> Option<String> {
    match tauri::utils::platform::bundle_type() {
        Some(BundleType::Deb) | Some(BundleType::Rpm) => None,
        _ => Some("This copy was not installed from the .deb or .rpm package, so it cannot update itself. Download new versions from GitHub.".into()),
    }
}

#[derive(Deserialize)]
struct Manifest {
    version: String,
    #[serde(default)]
    notes: Option<String>,
}

async fn read_manifest_version(
    url: &str,
    current: &Version,
) -> Result<Option<(Option<Update>, Offer)>, String> {
    let manifest = reqwest::Client::new()
        .get(url)
        .timeout(Duration::from_secs(20))
        .send()
        .await
        .and_then(|response| response.error_for_status())
        .map_err(|error| format!("Could not read the update manifest: {error}"))?
        .json::<Manifest>()
        .await
        .map_err(|error| format!("The update manifest is not valid: {error}"))?;
    let version = Version::parse(manifest.version.trim_start_matches('v'))
        .map_err(|error| format!("The update manifest has an invalid version: {error}"))?;
    Ok((version > *current).then(|| {
        (
            None,
            Offer {
                prerelease: !version.pre.is_empty(),
                version: version.to_string(),
                notes: manifest.notes.unwrap_or_default(),
                page: String::new(),
            },
        )
    }))
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or_default()
}

impl Updates {
    pub fn load(data_dir: &std::path::Path) -> Self {
        let path = data_dir.join("updates.json");
        let prefs = std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        Self {
            path,
            prefs: Arc::new(Mutex::new(prefs)),
            inner: Default::default(),
        }
    }

    async fn view(&self, app: &AppHandle) -> View {
        let prefs = self.prefs.lock().await.clone();
        let inner = self.inner.lock().await;
        let current = app.package_info().version.clone();
        View {
            include_prereleases: prefs.include_prereleases.unwrap_or(BUILT_AS_PRERELEASE),
            current_version: current.to_string(),
            check_on_launch: prefs.check_on_launch,
            unsupported: unsupported_reason(),
            status: inner.status.clone(),
            offer: inner.offer.clone(),
            checked_at: inner.checked_at,
        }
    }

    /// The newer release, if any, and the installable update for it. A build
    /// that cannot install updates still learns about the newer version.
    async fn find(&self, app: &AppHandle) -> Result<Option<(Option<Update>, Offer)>, String> {
        let current = app.package_info().version.clone();
        let (manifest, offer) = match std::env::var(MANIFEST_OVERRIDE) {
            Ok(url) if !url.is_empty() => (url, None),
            _ => {
                let include = self
                    .prefs
                    .lock()
                    .await
                    .include_prereleases
                    .unwrap_or(BUILT_AS_PRERELEASE);
                let releases = reqwest::Client::new()
                    .get(RELEASES_URL)
                    .header("User-Agent", "noullama-updater")
                    .header("Accept", "application/vnd.github+json")
                    .timeout(Duration::from_secs(20))
                    .send()
                    .await
                    .and_then(|response| response.error_for_status())
                    .map_err(|error| format!("Could not reach GitHub releases: {error}"))?
                    .json::<Vec<GithubRelease>>()
                    .await
                    .map_err(|error| {
                        format!("GitHub returned an unexpected release list: {error}")
                    })?;
                match pick_release(releases, &current, include) {
                    Some(candidate) => (candidate.manifest, Some(candidate.offer)),
                    None => return Ok(None),
                }
            }
        };
        if unsupported_reason().is_some() {
            // The updater would look for a generic `linux-x86_64` package,
            // which releases do not ship, so only the version is read.
            if let Some(offer) = offer {
                return Ok(Some((None, offer)));
            }
            return read_manifest_version(&manifest, &current).await;
        }
        let url = manifest
            .parse()
            .map_err(|error| format!("Invalid update manifest address {manifest}: {error}"))?;
        let update = app
            .updater_builder()
            .endpoints(vec![url])
            .and_then(|builder| builder.timeout(Duration::from_secs(30)).build())
            .map_err(|error| error.to_string())?
            .check()
            .await
            .map_err(|error| format!("Could not read the update manifest: {error}"))?;
        Ok(update.map(|update| {
            let offer = offer.unwrap_or_else(|| Offer {
                version: update.version.clone(),
                notes: update.body.clone().unwrap_or_default(),
                page: String::new(),
                prerelease: Version::parse(&update.version)
                    .map(|version| !version.pre.is_empty())
                    .unwrap_or(false),
            });
            (Some(update), offer)
        }))
    }

    async fn check(&self, app: &AppHandle) -> Result<(), String> {
        {
            let mut inner = self.inner.lock().await;
            if matches!(
                inner.status,
                Status::Checking | Status::Downloading { .. } | Status::Installing
            ) {
                return Ok(());
            }
            inner.status = Status::Checking;
        }
        let found = self.find(app).await;
        let mut inner = self.inner.lock().await;
        inner.checked_at = Some(now());
        match found {
            Ok(Some((update, offer))) => {
                inner.status = Status::Available;
                inner.offer = Some(offer);
                inner.pending = update;
                Ok(())
            }
            Ok(None) => {
                inner.status = Status::UpToDate;
                inner.offer = None;
                inner.pending = None;
                Ok(())
            }
            Err(message) => {
                inner.status = Status::Error {
                    message: message.clone(),
                };
                Err(message)
            }
        }
    }

    pub fn check_on_launch(&self, app: AppHandle) {
        let updates = self.clone();
        tauri::async_runtime::spawn(async move {
            if !updates.prefs.lock().await.check_on_launch {
                return;
            }
            // Let the window and gateway come up before touching the network.
            tokio::time::sleep(Duration::from_secs(4)).await;
            let _ = updates.check(&app).await;
        });
    }
}

#[tauri::command]
pub async fn update_state(
    app: AppHandle,
    updates: tauri::State<'_, Updates>,
) -> Result<View, String> {
    Ok(updates.view(&app).await)
}

#[tauri::command]
pub async fn set_update_prefs(
    include_prereleases: bool,
    check_on_launch: bool,
    app: AppHandle,
    updates: tauri::State<'_, Updates>,
) -> Result<View, String> {
    let prefs = Prefs {
        include_prereleases: Some(include_prereleases),
        check_on_launch,
    };
    let bytes = serde_json::to_vec_pretty(&prefs).map_err(|error| error.to_string())?;
    let temp = updates.path.with_extension("json.tmp");
    tokio::fs::write(&temp, bytes)
        .await
        .and(tokio::fs::rename(&temp, &updates.path).await)
        .map_err(|error| format!("Could not save update preferences: {error}"))?;
    *updates.prefs.lock().await = prefs;
    // A different channel may change which release is offered.
    {
        let mut inner = updates.inner.lock().await;
        if matches!(inner.status, Status::Available | Status::UpToDate) {
            *inner = Inner::default();
        }
    }
    Ok(updates.view(&app).await)
}

#[tauri::command]
pub async fn check_for_update(
    app: AppHandle,
    updates: tauri::State<'_, Updates>,
) -> Result<View, String> {
    updates.check(&app).await?;
    Ok(updates.view(&app).await)
}

/// Downloads and installs the offered update, then restarts through the normal
/// shutdown path so the runtime, gateway and downloads stop first.
#[tauri::command]
pub async fn install_update(
    app: AppHandle,
    updates: tauri::State<'_, Updates>,
) -> Result<(), String> {
    if let Some(reason) = unsupported_reason() {
        return Err(reason);
    }
    let update = {
        let mut inner = updates.inner.lock().await;
        if !matches!(inner.status, Status::Available | Status::Error { .. }) {
            return Err("An update is already being checked or installed.".into());
        }
        let update = inner
            .pending
            .clone()
            .ok_or("Check for updates before installing.")?;
        inner.status = Status::Downloading {
            received: 0,
            total: None,
        };
        update
    };
    let updates = updates.inner().clone();
    tauri::async_runtime::spawn(async move {
        let progress = updates.inner.clone();
        let mut received = 0u64;
        let result = update
            .download(
                |chunk, total| {
                    received += chunk as u64;
                    if let Ok(mut inner) = progress.try_lock() {
                        inner.status = Status::Downloading { received, total };
                    }
                },
                || {},
            )
            .await
            .map_err(|error| format!("The update could not be downloaded or verified: {error}"));
        let result = match result {
            Ok(bytes) => {
                updates.inner.lock().await.status = Status::Installing;
                // Waits on the pkexec password prompt.
                tauri::async_runtime::spawn_blocking(move || update.install(bytes))
                    .await
                    .map_err(|error| error.to_string())
                    .and_then(|installed| {
                        installed.map_err(|error| {
                            format!("The update was not installed: {error}. The password prompt may have been cancelled.")
                        })
                    })
            }
            Err(message) => Err(message),
        };
        match result {
            Ok(()) => {
                let state = app.state::<crate::runtime::AppState>().inner().clone();
                crate::lifecycle::shutdown_app(&state).await;
                app.restart();
            }
            Err(message) => {
                // The offer is kept, so Install can be tried again.
                updates.inner.lock().await.status = Status::Error { message };
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn release(tag: &str, prerelease: bool, manifest: bool) -> GithubRelease {
        GithubRelease {
            tag_name: tag.into(),
            html_url: format!("https://example.test/{tag}"),
            body: None,
            draft: false,
            prerelease,
            assets: if manifest {
                vec![GithubAsset {
                    name: MANIFEST_NAME.into(),
                    browser_download_url: format!("https://example.test/{tag}/latest.json"),
                }]
            } else {
                Vec::new()
            },
        }
    }

    #[test]
    fn picks_newest_stable_release_with_a_manifest() {
        let current = Version::parse("0.2.0").unwrap();
        let releases = vec![
            release("v0.4.0", false, false),
            release("v0.3.1-beta.1", true, true),
            release("v0.3.0", false, true),
            release("v0.2.0", false, true),
        ];
        let picked = pick_release(releases, &current, false).unwrap();
        assert_eq!(picked.offer.version, "0.3.0");
        assert!(!picked.offer.prerelease);
    }

    #[test]
    fn pre_tags_carry_plain_versions() {
        let current = Version::parse("0.3.0").unwrap();
        let releases = vec![
            release("v0.3.10-pre", true, true),
            release("v0.3.2-pre", true, true),
            release("v0.3.1", false, true),
        ];
        let picked = pick_release(releases, &current, true).unwrap();
        assert_eq!(picked.offer.version, "0.3.10");
        assert!(picked.offer.prerelease);
        let releases = vec![release("v0.3.0-pre", true, true)];
        assert!(pick_release(releases, &current, true).is_none());
        let releases = vec![release("v0.3.1-nightly", true, true)];
        assert_eq!(
            pick_release(releases, &current, true)
                .unwrap()
                .offer
                .version,
            "0.3.1"
        );
    }

    #[test]
    fn nothing_newer_means_no_offer() {
        let current = Version::parse("0.3.0").unwrap();
        let mut draft = release("v0.9.0", false, true);
        draft.draft = true;
        let releases = vec![
            draft,
            release("v0.3.0", false, true),
            release("v0.2.0", false, true),
            release("nightly", false, true),
        ];
        assert!(pick_release(releases, &current, true).is_none());
    }

    #[test]
    fn a_pre_tag_counts_as_prerelease_even_if_github_says_otherwise() {
        let current = Version::parse("0.2.0").unwrap();
        let releases = vec![release("v0.3.0-pre", false, true)];
        assert!(pick_release(releases, &current, false).is_none());
    }
}
