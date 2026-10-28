use crate::runtime::{validate_request, AppState};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

#[derive(Clone, Serialize, Deserialize)]
pub struct SavedPreset {
    pub id: String,
    pub name: String,
    pub questions: Value,
    pub state: String,
    pub state_mode: String,
}

fn read_presets(directory: &Path) -> Result<Vec<SavedPreset>, String> {
    let path = directory.join("presets.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("Cannot read saved presets: {e}"))
}

fn write_presets(directory: &Path, presets: &[SavedPreset]) -> Result<(), String> {
    let temporary = directory.join("presets.json.tmp");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary).map_err(|e| e.to_string())?;
    use std::io::Write;
    file.write_all(&serde_json::to_vec_pretty(presets).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    std::fs::rename(temporary, directory.join("presets.json")).map_err(|e| e.to_string())
}

fn store_preset(directory: &Path, mut preset: SavedPreset) -> Result<Vec<SavedPreset>, String> {
    preset.name = preset.name.trim().to_owned();
    if preset.name.is_empty() || preset.name.chars().count() > 80 {
        return Err("Choose a preset name between 1 and 80 characters.".into());
    }
    let state = match preset.state_mode.as_str() {
        "text" => json!(preset.state),
        "json" => serde_json::from_str(&preset.state).map_err(|e| format!("State JSON: {e}"))?,
        _ => return Err("Unknown state format.".into()),
    };
    validate_request(&json!({"state":state,"questions":preset.questions}))?;
    let mut presets = read_presets(directory)?;
    if presets.iter().any(|other| {
        other.id != preset.id && other.name.to_lowercase() == preset.name.to_lowercase()
    }) {
        return Err(
            "A saved preset already has that name. Choose another name or select it to update it."
                .into(),
        );
    }
    if preset.id.is_empty() {
        preset.id = format!("saved-{}", uuid::Uuid::new_v4());
        presets.push(preset);
    } else {
        let current = presets
            .iter_mut()
            .find(|other| other.id == preset.id)
            .ok_or("Saved preset no longer exists. Select Custom to save a new one.")?;
        *current = preset;
    }
    write_presets(directory, &presets)?;
    Ok(presets)
}

fn remove_preset(directory: &Path, id: &str) -> Result<Vec<SavedPreset>, String> {
    let mut presets = read_presets(directory)?;
    if !presets.iter().any(|preset| preset.id == id) {
        return Err("Saved preset not found; built-in presets cannot be deleted.".into());
    }
    presets.retain(|preset| preset.id != id);
    write_presets(directory, &presets)?;
    Ok(presets)
}

#[tauri::command]
pub async fn list_presets(state: tauri::State<'_, AppState>) -> Result<Vec<SavedPreset>, String> {
    let _guard = state.presets_lock.lock().await;
    read_presets(&state.data_dir)
}
#[tauri::command]
pub async fn save_preset(
    preset: SavedPreset,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<SavedPreset>, String> {
    let _guard = state.presets_lock.lock().await;
    store_preset(&state.data_dir, preset)
}
#[tauri::command]
pub async fn delete_preset(
    id: String,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<SavedPreset>, String> {
    let _guard = state.presets_lock.lock().await;
    remove_preset(&state.data_dir, &id)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saved_presets_survive_reopen_update_and_delete_without_overwriting_names() {
        let dir = std::env::temp_dir().join(format!("noullama-presets-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let preset = SavedPreset {
            id: String::new(),
            name: " My router ".into(),
            questions: json!({"route":{"type":"choice","instructions":"Which model?","criteria":{"fast":null,"slow":null}}}),
            state: "hello".into(),
            state_mode: "text".into(),
        };
        let stored = store_preset(&dir, preset.clone()).unwrap();
        let reopened = read_presets(&dir).unwrap();
        assert_eq!(reopened[0].name, "My router");
        assert_eq!(reopened[0].state, "hello");
        assert!(store_preset(&dir, preset).is_err());
        let mut updated = reopened[0].clone();
        updated.state = "updated".into();
        assert_eq!(store_preset(&dir, updated).unwrap()[0].state, "updated");
        assert!(remove_preset(&dir, "triage").is_err());
        assert!(remove_preset(&dir, &stored[0].id).unwrap().is_empty());
        assert!(read_presets(&dir).unwrap().is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
