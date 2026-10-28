use std::{
    sync::atomic::{AtomicBool, AtomicU16, Ordering},
    time::Duration,
};
use tokio::{
    sync::{watch, Mutex},
    task::JoinHandle,
};

use crate::runtime::AppState;

pub struct AppControl {
    pub exiting: AtomicBool,
    pub quit_confirmation: AtomicBool,
    pub exit_ready: AtomicBool,
    pub tray_created: AtomicBool,
    pub active_api_port: AtomicU16,
    pub api_stop: watch::Sender<bool>,
    pub api_task: Mutex<Option<JoinHandle<()>>>,
    pub download_task: Mutex<Option<JoinHandle<()>>>,
    exit_lock: Mutex<()>,
}

impl Default for AppControl {
    fn default() -> Self {
        Self {
            exiting: AtomicBool::new(false),
            quit_confirmation: AtomicBool::new(false),
            exit_ready: AtomicBool::new(false),
            tray_created: AtomicBool::new(false),
            active_api_port: AtomicU16::new(0),
            api_stop: watch::channel(false).0,
            api_task: Mutex::new(None),
            download_task: Mutex::new(None),
            exit_lock: Mutex::new(()),
        }
    }
}

pub async fn shutdown_app(state: &AppState) {
    state.app.exiting.store(true, Ordering::SeqCst);
    let _guard = state.app.exit_lock.lock().await;
    if state.app.exit_ready.load(Ordering::SeqCst) {
        return;
    }
    // Stop accepting API requests before unloading the runtime. Any in-flight
    // inference then fails promptly when the owned child is killed and reaped.
    state.app.api_stop.send_replace(true);
    state.shutdown().await;
    if let Some(mut task) = state.runtimes.task.lock().await.take() {
        if tokio::time::timeout(Duration::from_secs(2), &mut task)
            .await
            .is_err()
        {
            task.abort();
            let _ = task.await;
        }
    }
    if let Some(mut task) = state.app.download_task.lock().await.take() {
        if tokio::time::timeout(Duration::from_secs(2), &mut task)
            .await
            .is_err()
        {
            task.abort();
            let _ = task.await;
        }
    }
    if let Some(mut task) = state.app.api_task.lock().await.take() {
        if tokio::time::timeout(Duration::from_secs(2), &mut task)
            .await
            .is_err()
        {
            task.abort();
            let _ = task.await;
        }
    }
    state.app.exit_ready.store(true, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn full_exit_releases_gateway_port_and_rejects_new_work() {
        let path =
            std::env::temp_dir().join(format!("noullama-full-exit-{}", uuid::Uuid::new_v4()));
        let mut state = AppState::new(path.clone()).unwrap();
        let reservation = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        state.api_port = reservation.local_addr().unwrap().port();
        crate::api::start_with_listener(state.clone(), reservation).await;
        let url = format!("http://127.0.0.1:{}/health", state.api_port);
        tokio::time::timeout(Duration::from_secs(2), async {
            while state.local_client.get(&url).send().await.is_err() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        tokio::time::timeout(Duration::from_secs(3), shutdown_app(&state))
            .await
            .unwrap();
        assert!(
            tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, state.api_port))
                .await
                .is_ok()
        );
        assert!(state.app.api_task.lock().await.is_none());
        assert!(state.app.exit_ready.load(Ordering::SeqCst));
        assert!(crate::runtime::load_model(&state)
            .await
            .unwrap_err()
            .contains("shutting down"));
        crate::api::start(state.clone()).await;
        assert!(state.app.api_task.lock().await.is_none());
        shutdown_app(&state).await; // duplicate quit is harmless
        std::fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test]
    #[ignore = "requires NOULLAMA_TEST_MODEL and a local llama-server runtime"]
    async fn real_full_exit_releases_gateway_and_runtime_ports() {
        let path =
            std::env::temp_dir().join(format!("noullama-real-exit-{}", uuid::Uuid::new_v4()));
        let mut state = AppState::new(path.clone()).unwrap();
        let api_reservation = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let runtime_reservation = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        state.api_port = api_reservation.local_addr().unwrap().port();
        let runtime_port = runtime_reservation.local_addr().unwrap().port();
        {
            let mut settings = state.settings.lock().await;
            settings.model_path = std::env::var("NOULLAMA_TEST_MODEL").unwrap();
            settings.runtime_path = std::env::var("NOULLAMA_TEST_RUNTIME")
                .unwrap_or_else(|_| crate::runtime::DEFAULT_RUNTIME.into());
            settings.runtime_port = runtime_port;
            settings.gpu_layers = std::env::var("NOULLAMA_TEST_GPU_LAYERS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        }
        drop(runtime_reservation);
        crate::api::start_with_listener(state.clone(), api_reservation).await;
        crate::runtime::load_model(&state).await.unwrap();
        tokio::time::timeout(Duration::from_secs(90), async {
            while !*state.ready.lock().await {
                assert!(
                    state.session.lock().await.is_some(),
                    "Startup failed: {:?}",
                    state.logs.lock().await
                );
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .unwrap();
        for port in [state.api_port, runtime_port] {
            assert!(
                tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
                    .await
                    .is_err()
            );
        }
        shutdown_app(&state).await;
        for port in [state.api_port, runtime_port] {
            assert!(
                tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
                    .await
                    .is_ok(),
                "Port {port} remained open"
            );
        }
        assert!(state.child.lock().await.is_none());
        assert!(state.session.lock().await.is_none());
        println!(
            "Full exit released gateway {} and runtime {runtime_port}",
            state.api_port
        );
        std::fs::remove_dir_all(path).unwrap();
    }
}
