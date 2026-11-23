use crate::{library, runtime::AppState};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::{self, File, OpenOptions},
    io::{ErrorKind, Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::Mutex;

#[derive(Clone, Serialize, Deserialize)]
pub struct Binding {
    path: String,
    name: String,
    sha256: String,
    aliases: Vec<String>,
}

// None means unrestricted. A restricted grant is tied to Library entries,
// irrespective of the names used to reach their loaded instances.
#[derive(Clone, Default)]
pub struct Access(pub Option<Vec<Binding>>);

impl Access {
    pub fn allows_alias_owner(&self, base: &str, path: &str) -> bool {
        self.0.as_ref().is_none_or(|bindings| {
            !bindings.iter().any(|binding| {
                binding.path != path
                    && binding
                        .aliases
                        .iter()
                        .any(|alias| alias.eq_ignore_ascii_case(base))
            })
        })
    }

    pub async fn allowed_paths(
        &self,
        state: &AppState,
        library: &library::Library,
    ) -> Option<HashSet<String>> {
        let bindings = self.0.as_ref()?;
        if library.error.is_some() {
            return Some(HashSet::new());
        }
        let missing: Vec<&str> = bindings
            .iter()
            .filter_map(|binding| {
                let entry = library.entry(&binding.path)?;
                entry.sha256.is_empty().then_some(binding.path.as_str())
            })
            .collect();
        let cached = library::cached_hashes(state, &missing).await;
        Some(
            bindings
                .iter()
                .filter_map(|binding| {
                    let entry = library.entry(&binding.path)?;
                    let sum = if entry.sha256.is_empty() {
                        cached.get(&binding.path)?.as_str()
                    } else {
                        &entry.sha256
                    };
                    (entry.name == binding.name && sum == binding.sha256)
                        .then(|| entry.path.clone())
                })
                .collect(),
        )
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct KeyInfo {
    id: String,
    name: String,
    prefix: String,
    created: u64,
    expires: Option<u64>,
    models: Option<Vec<Binding>>,
}

#[derive(Serialize, Deserialize)]
struct StoredKey {
    digest: [u8; 32],
    #[serde(flatten)]
    info: KeyInfo,
}

#[derive(Serialize)]
pub struct CreatedKey {
    key: String,
    info: KeyInfo,
}

pub struct KeyStore {
    directory: PathBuf,
    lock: Mutex<()>,
}

pub enum Rejection {
    Unknown,
    Expired,
    Unavailable,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl KeyStore {
    pub fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            lock: Mutex::new(()),
        }
    }

    fn read(&self) -> Result<Vec<StoredKey>, String> {
        match fs::read(self.directory.join("api-keys.json")) {
            Ok(bytes) => {
                serde_json::from_slice(&bytes).map_err(|_| "Cannot read API key file.".into())
            }
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(Vec::new()),
            Err(_) => Err("Cannot read API key file.".into()),
        }
    }

    fn write(&self, keys: &[StoredKey]) -> Result<(), String> {
        fn replace(directory: &Path, bytes: &[u8]) -> std::io::Result<()> {
            fs::create_dir_all(directory)?;
            let temporary = directory.join(format!(".api-keys-{}.tmp", uuid::Uuid::new_v4()));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temporary)?;
            let result = file
                .write_all(bytes)
                .and_then(|()| file.sync_all())
                .and_then(|()| fs::rename(&temporary, directory.join("api-keys.json")));
            if result.is_err() {
                let _ = fs::remove_file(&temporary);
            }
            result
        }

        let bytes = serde_json::to_vec_pretty(keys).map_err(|_| "Cannot encode API keys.")?;
        replace(&self.directory, &bytes).map_err(|_| "Cannot save API keys.".into())
    }

    pub async fn authorize(&self, key: &str) -> Result<Access, Rejection> {
        let digest: [u8; 32] = Sha256::digest(key).into();
        let _guard = self.lock.lock().await;
        let found = self
            .read()
            .map_err(|_| Rejection::Unavailable)?
            .into_iter()
            .find(|stored| stored.digest == digest)
            .ok_or(Rejection::Unknown)?;
        if found.info.expires.is_some_and(|expiry| expiry <= now()) {
            return Err(Rejection::Expired);
        }
        Ok(Access(found.info.models))
    }

    async fn list(&self) -> Result<Vec<KeyInfo>, String> {
        let _guard = self.lock.lock().await;
        Ok(self.read()?.into_iter().map(|stored| stored.info).collect())
    }

    async fn insert(
        &self,
        name: &str,
        expires: Option<u64>,
        models: Option<Vec<Binding>>,
    ) -> Result<CreatedKey, String> {
        let _guard = self.lock.lock().await;
        let mut keys = self.read()?;
        let created = now();
        if expires.is_some_and(|expiry| expiry <= created) {
            return Err("Choose an expiry in the future.".into());
        }
        if keys
            .iter()
            .any(|key| key.info.name.to_lowercase() == name.to_lowercase())
        {
            return Err("An API key already has that name.".into());
        }
        let mut random = [0u8; 32];
        File::open("/dev/urandom")
            .and_then(|mut file| file.read_exact(&mut random))
            .map_err(|_| "Cannot generate an API key.")?;
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
        let id = uuid::Uuid::new_v4().simple().to_string();
        let prefix = format!("da-noul-{}", &id[..8]);
        let key = format!("{prefix}-{}", URL_SAFE_NO_PAD.encode(random));
        let info = KeyInfo {
            id,
            name: name.into(),
            prefix,
            created,
            expires,
            models,
        };
        keys.push(StoredKey {
            digest: Sha256::digest(&key).into(),
            info: info.clone(),
        });
        self.write(&keys)?;
        Ok(CreatedKey { key, info })
    }

    async fn delete(&self, id: &str) -> Result<Vec<KeyInfo>, String> {
        let _guard = self.lock.lock().await;
        let mut keys = self.read()?;
        keys.retain(|key| key.info.id != id);
        self.write(&keys)?;
        Ok(keys.into_iter().map(|stored| stored.info).collect())
    }
}

#[derive(Deserialize)]
pub struct ModelSelection {
    pub path: String,
    pub name: String,
}

impl ModelSelection {
    fn entry<'a>(&self, library: &'a library::Library) -> Option<&'a library::Entry> {
        library
            .entry(&self.path)
            .filter(|entry| entry.name == self.name || entry.aliases.contains(&self.name))
    }
}

pub async fn create(
    state: &AppState,
    name: &str,
    expires: Option<u64>,
    models: Option<Vec<ModelSelection>>,
) -> Result<CreatedKey, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Enter a name.".into());
    }
    let Some(models) = models else {
        return state.api_keys.insert(name, expires, None).await;
    };
    if models.is_empty() {
        return Err("Select at least one model.".into());
    }
    library::refresh(state).await;
    let entries = {
        let library = state.library.lock().await;
        if library.error.is_some() {
            return Err("Fix the Library error before creating a restricted key.".into());
        }
        let mut seen = HashSet::new();
        let mut entries = Vec::new();
        for model in &models {
            let entry = model
                .entry(&library)
                .ok_or("A selected model or alias changed in Library. Try again.")?;
            if seen.insert(&entry.path) {
                entries.push(entry.clone());
            }
        }
        entries
    };
    let mut bindings = Vec::new();
    for entry in entries {
        let sha256 = if entry.sha256.is_empty() {
            library::current_hash(state, &entry.path, false).await?
        } else {
            entry.sha256
        };
        bindings.push(Binding {
            path: entry.path,
            name: entry.name,
            sha256,
            aliases: entry.aliases,
        });
    }
    library::refresh(state).await;
    let library = state.library.lock().await;
    if models.iter().any(|model| model.entry(&library).is_none()) {
        return Err("A selected model or alias changed in Library. Try again.".into());
    }
    let access = Access(Some(bindings));
    let allowed = access
        .allowed_paths(state, &library)
        .await
        .unwrap_or_default();
    let mut bindings = access.0.unwrap_or_default();
    for binding in &mut bindings {
        let entry = library
            .entry(&binding.path)
            .filter(|entry| allowed.contains(&entry.path))
            .ok_or("A selected model changed while creating the key. Try again.")?;
        binding.aliases.clone_from(&entry.aliases);
    }
    state.api_keys.insert(name, expires, Some(bindings)).await
}

#[tauri::command]
pub async fn set_require_api_key(
    on: bool,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    crate::runtime::update_settings(&state, |settings| {
        settings.require_api_key = on;
        Ok(())
    })
    .await
}

#[tauri::command]
pub async fn list_api_keys(state: tauri::State<'_, AppState>) -> Result<Vec<KeyInfo>, String> {
    state.api_keys.list().await
}

#[tauri::command]
pub async fn create_api_key(
    name: String,
    expires: Option<u64>,
    models: Option<Vec<ModelSelection>>,
    state: tauri::State<'_, AppState>,
) -> Result<CreatedKey, String> {
    create(&state, &name, expires, models).await
}

#[tauri::command]
pub async fn delete_api_key(
    id: String,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<KeyInfo>, String> {
    state.api_keys.delete(&id).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[tokio::test]
    async fn api_key_file_is_private_and_hash_only_and_survives_reload() {
        let directory =
            std::env::temp_dir().join(format!("noullama-test-{}", uuid::Uuid::new_v4()));
        let state =
            AppState::with_config_dir(directory.join("data"), directory.join("config")).unwrap();
        let (first, duplicate) = tokio::join!(
            create(&state, "  Example  ", None, None),
            create(&state, "example", None, None)
        );
        let created = match (first, duplicate) {
            (Ok(key), Err(_)) | (Err(_), Ok(key)) => key,
            _ => panic!("Exactly one key can use a name"),
        };
        assert_eq!(created.info.name.to_lowercase(), "example");
        assert!(created.key.starts_with("da-noul-"));
        assert_eq!(
            created.info.prefix,
            format!("da-noul-{}", &created.info.id[..8])
        );
        assert!(created
            .key
            .starts_with(&format!("{}-", created.info.prefix)));
        let path = directory.join("config/api-keys.json");
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let raw = fs::read_to_string(&path).unwrap();
        assert!(!raw.contains(&created.key));
        assert!(!raw.contains(&created.key[created.info.prefix.len() + 1..]));
        assert!(!directory.join("data/api-keys.json").exists());

        let legacy = "nk-1234abcd-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let mut stored = state.api_keys.read().unwrap();
        stored.push(StoredKey {
            digest: Sha256::digest(legacy).into(),
            info: KeyInfo {
                id: "1234abcd000000000000000000000000".into(),
                name: "Legacy".into(),
                prefix: "nk-1234abcd".into(),
                created: 1,
                expires: None,
                models: None,
            },
        });
        state.api_keys.write(&stored).unwrap();
        let reloaded = KeyStore::new(directory.join("config"));
        let listed = reloaded.list().await.unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].id, created.info.id);
        assert_eq!(listed[1].prefix, "nk-1234abcd");
        assert!(reloaded.authorize(&created.key).await.is_ok());
        assert!(reloaded.authorize(legacy).await.is_ok());
        assert_eq!(reloaded.delete(&created.info.id).await.unwrap().len(), 1);
        assert!(reloaded.authorize(legacy).await.is_ok());
        assert!(matches!(
            reloaded.authorize(&created.key).await,
            Err(Rejection::Unknown)
        ));
        fs::remove_dir_all(directory).unwrap();
    }
}
