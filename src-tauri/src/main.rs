mod api;
mod files;
mod hub;
mod presets;
mod runtime;

use runtime::AppState;
use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let state = AppState::new(app.path().app_data_dir()?)?;
            app.manage(state.clone());
            tauri::async_runtime::spawn(async move {
                if let Err(error) = api::serve(state.clone()).await {
                    state.log(format!("API failed: {error}")).await;
                    *state.api_error.lock().await = Some(error.to_string());
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            runtime::snapshot,
            runtime::save_settings,
            runtime::probe_runtime,
            runtime::start_runtime,
            runtime::stop_runtime,
            runtime::decide,
            hub::hf_session,
            hub::set_hf_token,
            hub::search_models,
            hub::inspect_model,
            hub::download_model,
            hub::local_models,
            presets::list_presets,
            presets::save_preset,
            presets::delete_preset,
            files::pick_model_file,
            files::reveal_model_file
        ])
        .build(tauri::generate_context!())
        .expect("Could not initialize noullama")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                let state = app.state::<AppState>();
                tauri::async_runtime::block_on(async {
                    state.shutdown().await;
                });
            }
        });
}

#[cfg(all(test, feature = "custom-protocol"))]
mod build_tests {
    #[test]
    fn standalone_build_embeds_html_and_script_assets() {
        assert!(
            !tauri::is_dev(),
            "Standalone builds must not depend on Vite"
        );
        let context: tauri::Context<tauri::Wry> = tauri::generate_context!();
        let html = context
            .assets()
            .get(&"index.html".into())
            .expect("Embedded index.html");
        let html = String::from_utf8_lossy(&html);
        let scripts = regex::Regex::new(r#"src="([^"]+\.js)""#).unwrap();
        let mut count = 0;
        for script in scripts.captures_iter(&html) {
            assert!(
                context.assets().get(&script[1].into()).is_some(),
                "Missing embedded script {}",
                &script[1]
            );
            count += 1;
        }
        assert!(count > 0, "Embedded HTML must reference the built frontend");
    }
}
