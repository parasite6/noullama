use std::path::Path;
use tauri_plugin_dialog::DialogExt;

#[tauri::command]
pub async fn pick_model_file(app: tauri::AppHandle) -> Result<Option<String>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        app.dialog()
            .file()
            .add_filter("GGUF model", &["gguf"])
            .blocking_pick_file()
            .map(|file| {
                file.into_path()
                    .map(|path| path.to_string_lossy().into_owned())
                    .map_err(|e| e.to_string())
            })
            .transpose()
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn reveal_model_file(path: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || reveal_file(&path))
        .await
        .map_err(|e| e.to_string())?
}

fn reveal_file(path: &str) -> Result<(), String> {
    if !Path::new(path).is_file() {
        return Err("This file is missing or has not finished downloading.".into());
    }
    tauri_plugin_opener::reveal_item_in_dir(path)
        .map_err(|e| format!("Could not reveal the file in your file manager: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "opens the system file manager; needs NOULLAMA_REVEAL_TEST_FILE"]
    fn real_file_manager_reveal() {
        let file = std::env::var("NOULLAMA_REVEAL_TEST_FILE").unwrap();
        let start = std::time::Instant::now();
        let result = reveal_file(&file);
        println!(
            "File manager response after {:?}: {result:?}",
            start.elapsed()
        );
        assert!(result.is_ok());
    }
}
