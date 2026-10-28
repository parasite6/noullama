use crate::{lifecycle::shutdown_app, runtime::AppState};
use std::sync::atomic::Ordering;
use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    Manager,
};

pub fn request_exit(app: &tauri::AppHandle, code: i32) {
    let state = app.state::<AppState>().inner().clone();
    if state.app.exiting.swap(true, Ordering::SeqCst) {
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        shutdown_app(&state).await;
        app.exit(code);
    });
}

#[tauri::command]
pub fn quit_app(app: tauri::AppHandle) -> Result<(), String> {
    if !app
        .state::<AppState>()
        .app
        .quit_confirmation
        .swap(false, Ordering::SeqCst)
    {
        return Err("Request the quit confirmation before exiting.".into());
    }
    request_exit(&app, 0);
    Ok(())
}

#[tauri::command]
pub fn request_quit(app: tauri::AppHandle) {
    app.state::<AppState>()
        .app
        .quit_confirmation
        .store(true, Ordering::SeqCst);
    open_window(&app);
}

#[tauri::command]
pub fn cancel_quit(state: tauri::State<'_, AppState>) {
    state.app.quit_confirmation.store(false, Ordering::SeqCst);
}

fn open_window(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

#[cfg(target_os = "linux")]
pub fn tray_host_available() -> bool {
    use dbus::blocking::stdintf::org_freedesktop_dbus::Properties;
    let Ok(connection) = dbus::blocking::Connection::new_session() else {
        return false;
    };
    let proxy = connection.with_proxy(
        "org.kde.StatusNotifierWatcher",
        "/StatusNotifierWatcher",
        std::time::Duration::from_millis(300),
    );
    proxy
        .get::<bool>(
            "org.kde.StatusNotifierWatcher",
            "IsStatusNotifierHostRegistered",
        )
        .unwrap_or(false)
}

#[cfg(not(target_os = "linux"))]
pub fn tray_host_available() -> bool {
    true
}

pub fn install_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "linux")]
    {
        // The backend loads these at runtime and panics if neither is present.
        // Probe first so machines without the tray library retain normal close.
        let available = ["libayatana-appindicator3.so.1", "libappindicator3.so.1"]
            .iter()
            .any(|library| {
                // SAFETY: these are the same fixed system library names used by
                // tray-icon's AppIndicator backend; no symbols escape this scope.
                unsafe { libloading::Library::new(library).is_ok() }
            });
        if !available {
            return Err(
                "AppIndicator library is unavailable; closing the window will quit Noullama."
                    .into(),
            );
        }
    }
    let open = MenuItem::with_id(app, "open", "Open Noullama", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Noullama", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open, &quit])?;
    let icon = app
        .default_window_icon()
        .ok_or("Tray icon is unavailable")?
        .clone();
    TrayIconBuilder::with_id("noullama")
        .icon(icon)
        .tooltip("Noullama")
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "open" => open_window(app),
            "quit" => request_quit(app.clone()),
            _ => {}
        })
        .build(app)?;
    app.state::<AppState>()
        .app
        .tray_created
        .store(true, Ordering::SeqCst);
    Ok(())
}

pub fn handle_window_event(window: &tauri::Window, event: &tauri::WindowEvent) {
    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
        let state = window.state::<AppState>();
        if state.app.tray_created.load(Ordering::SeqCst)
            && !state.app.exiting.load(Ordering::SeqCst)
            && tray_host_available()
            && window.hide().is_ok()
        {
            api.prevent_close();
        } else if !state.app.exiting.load(Ordering::SeqCst) {
            api.prevent_close();
            request_quit(window.app_handle().clone());
        }
    }
}

pub fn install_signal_handler(app: tauri::AppHandle) {
    #[cfg(unix)]
    {
        // Register before starting any listeners, rather than waiting for a
        // background future to be polled after GTK/tray initialization.
        let signals = tauri::async_runtime::block_on(async {
            use tokio::signal::unix::{signal, SignalKind};
            Ok::<_, std::io::Error>((
                signal(SignalKind::terminate())?,
                signal(SignalKind::interrupt())?,
            ))
        });
        if let Ok((mut terminate, mut interrupt)) = signals {
            tauri::async_runtime::spawn(async move {
                tokio::select! {
                    _ = interrupt.recv() => {},
                    _ = terminate.recv() => {},
                }
                request_exit(&app, 0);
            });
            return;
        }
    }
    tauri::async_runtime::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            request_exit(&app, 0);
        }
    });
}
