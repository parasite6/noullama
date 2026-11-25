mod api;
mod api_keys;
mod context;
mod library;
mod desktop;
mod files;
mod decision;
mod devices;
mod gguf;
mod hardware;
mod oom;
mod hub;
mod instance;
mod lifecycle;
mod presets;
mod runtime;
mod runtimes;
mod updates;

use runtime::AppState;
use std::sync::atomic::Ordering;
use tauri::Manager;

fn main() {
    // WebKitGTK keeps its default DMA-BUF renderer. Forcing the shared-memory
    // path copies every frame through the CPU and makes all animation choppy.
    // A user who needs it can still set WEBKIT_DISABLE_DMABUF_RENDERER=1.
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
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
            let updates = updates::Updates::load(&data_dir);
            updates.check_on_launch(app.handle().clone());
            app.manage(updates);
            let state = AppState::with_config_dir(data_dir, app.path().app_config_dir()?)?;
            app.manage(state.clone());
            runtime::start_memory_guard(state.clone());
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
            runtimes::download_cpu_runtime,
            devices::runtime_devices,
            runtime::snapshot,
            api_keys::set_require_api_key,
            api_keys::list_api_keys,
            api_keys::create_api_key,
            api_keys::delete_api_key,
            runtime::save_settings,
            runtime::probe_runtime,
            runtime::start_runtime,
            runtime::stop_runtime,
            runtime::decide,
            runtime::model_memory,
            runtime::set_model_aliases,
            runtime::unload_model,
            library::save_library,
            library::reload_library,
            library::open_models_file,
            library::open_backups_folder,
            library::record_sha256,
            library::set_sha256_check,
            runtime::set_memory_preference,
            runtime::crowded_load_warning,
            runtime::load_fit,
            runtime::dismiss_load_error,
            runtime::set_instance_suffix,
            runtime::set_fall_through,
            runtime::set_instance_aliases,
            runtime::set_any_available_when_busy,
            runtime::set_any_available_wait_ms,
            context::set_instance_keep_context,
            context::clear_context,
            hub::hf_session,
            hub::set_hf_token,
            hub::set_download_transport,
            hub::search_models,
            hub::inspect_model,
            hub::author_avatar,
            hub::download_model,
            hub::local_models,
            hub::model_mtp,
            hub::model_repo,
            hub::cancel_download,
            hub::dismiss_download,
            presets::list_presets,
            presets::save_preset,
            presets::delete_preset,
            files::pick_model_file,
            files::reveal_model_file,
            updates::update_state,
            updates::set_update_prefs,
            updates::check_for_update,
            updates::install_update
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
