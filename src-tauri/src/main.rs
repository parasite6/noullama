mod api;
mod desktop;
mod files;
mod hardware;
mod hub;
mod instance;
mod lifecycle;
mod presets;
mod runtime;
mod runtimes;

use runtime::AppState;
use std::sync::atomic::Ordering;
use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            let lease = instance::acquire(&data_dir)?;
            app.manage(lease);
            runtimes::sweep_staging(&data_dir);
            // Probe GPU and memory once in the background; the first snapshot
            // would otherwise wait on nvidia-smi.
            std::thread::spawn(|| {
                hardware::hardware();
            });
            let state = AppState::new(data_dir)?;
            app.manage(state.clone());
            desktop::install_signal_handler(app.handle().clone());
            tauri::async_runtime::block_on(api::start(state));
            if let Err(error) = desktop::install_tray(app) {
                let message = format!("Tray unavailable: {error}");
                eprintln!("{message}");
                tauri::async_runtime::block_on(app.state::<AppState>().log(message));
            }
            Ok(())
        })
        .on_window_event(desktop::handle_window_event)
        .invoke_handler(tauri::generate_handler![
            desktop::quit_app,
            desktop::request_quit,
            desktop::cancel_quit,
            runtimes::discover_runtimes,
            runtimes::runtime_releases,
            runtimes::download_runtimes,
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
            hub::author_avatar,
            hub::download_model,
            hub::local_models,
            hub::cancel_download,
            presets::list_presets,
            presets::save_preset,
            presets::delete_preset,
            files::pick_model_file,
            files::reveal_model_file
        ])
        .build(tauri::generate_context!())
        .expect("Could not initialize noullama")
        .run(|app, event| {
            match event {
                tauri::RunEvent::ExitRequested { api, code, .. } => {
                    if !app
                        .state::<AppState>()
                        .app
                        .exit_ready
                        .load(Ordering::SeqCst)
                    {
                        api.prevent_exit();
                        desktop::request_exit(app, code.unwrap_or(0));
                    }
                }
                tauri::RunEvent::Exit => {
                    // Fallback for a platform exit that bypasses ExitRequested.
                    tauri::async_runtime::block_on(lifecycle::shutdown_app(
                        &app.state::<AppState>(),
                    ));
                }
                _ => {}
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
