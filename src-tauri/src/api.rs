use crate::runtime::{validate_request, AppState, RouteError};
use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};

type ApiResult = Result<Json<Value>, ApiError>;
struct ApiError(StatusCode, String);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error":{"message":self.1,"type":"invalid_request_error","code":self.0.as_u16()}}))).into_response()
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(systemone))
        .route("/v1/chat/completions", post(chat))
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(state.clone(), loopback_only))
        .with_state(state)
}

async fn loopback_only(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let hosts = [
        format!("127.0.0.1:{}", state.listening_port()),
        format!("localhost:{}", state.listening_port()),
    ];
    let headers = request.headers();
    let host_valid = headers.get_all("host").iter().count() == 1
        && headers
            .get("host")
            .and_then(|h| h.to_str().ok())
            .is_some_and(|h| hosts.iter().any(|allowed| h.eq_ignore_ascii_case(allowed)));
    let origin_valid = !headers.contains_key("origin")
        || (headers.get_all("origin").iter().count() == 1
            && headers
                .get("origin")
                .and_then(|h| h.to_str().ok())
                .is_some_and(|origin| hosts.iter().any(|host| origin == format!("http://{host}"))));
    if !host_valid || !origin_valid {
        return ApiError(
            StatusCode::FORBIDDEN,
            "Only requests addressed to the local API from a local origin are allowed.".into(),
        )
        .into_response();
    }
    next.run(request).await
}

pub async fn start(state: AppState) {
    if state.app.exiting.load(std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    let runtime_port = state.settings.lock().await.runtime_port;
    let listener = match bind_available(state.api_port, runtime_port).await {
        Ok(listener) => listener,
        Err(error) => {
            state.log(format!("API failed: {error}")).await;
            *state.api_error.lock().await = Some(error.to_string());
            return;
        }
    };
    let port = listener
        .local_addr()
        .expect("Bound TCP listener has an address")
        .port();
    if port != state.api_port {
        let mut settings = state.settings.lock().await.clone();
        settings.api_port = port;
        let saved = crate::runtime::persist_settings(&state, settings).await;
        let notice = match saved {
            Ok(())=>format!("Local API moved to http://127.0.0.1:{port}/v1 because port {} is in use. This endpoint is saved for future launches.",state.api_port),
            Err(error)=>format!("Local API is available at http://127.0.0.1:{port}/v1, but its new port could not be saved: {error}"),
        };
        state.log(notice.clone()).await;
        *state.api_notice.lock().await = Some(notice);
    }
    *state.api_error.lock().await = None;
    start_with_listener(state, listener).await;
}

async fn bind_available(
    preferred: u16,
    runtime_port: u16,
) -> Result<tokio::net::TcpListener, std::io::Error> {
    for offset in 0..32u16 {
        let Some(port) = preferred.checked_add(offset).filter(|port| *port != 0) else {
            break;
        };
        if port == runtime_port {
            continue;
        }
        match tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)).await {
            Ok(listener) => return Ok(listener),
            Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => continue,
            Err(error) => return Err(error),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AddrInUse,
        "No free local API port near the configured port; choose another port in Runtime",
    ))
}

pub async fn start_with_listener(state: AppState, listener: tokio::net::TcpListener) {
    let _guard = state.lifecycle.lock().await;
    if state.app.exiting.load(std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    if state.app.api_task.lock().await.is_some() {
        return;
    }
    state.app.active_api_port.store(
        listener.local_addr().expect("Bound listener").port(),
        std::sync::atomic::Ordering::SeqCst,
    );
    let worker_state = state.clone();
    let task = tokio::spawn(async move {
        if let Err(error) = serve(listener, worker_state.clone()).await {
            worker_state.log(format!("API failed: {error}")).await;
            *worker_state.api_error.lock().await = Some(error.to_string());
        }
    });
    *state.app.api_task.lock().await = Some(task);
}

pub async fn serve(
    listener: tokio::net::TcpListener,
    state: AppState,
) -> Result<(), std::io::Error> {
    let mut stop = state.app.api_stop.subscribe();
    state
        .log(format!(
            "Local API listening at http://127.0.0.1:{}/v1",
            state.listening_port()
        ))
        .await;
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async move {
            let _ = stop.wait_for(|stop| *stop).await;
        })
        .await
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(
        json!({"status":"ok", "service":"noullama", "api_port":state.listening_port(), "model_ready": state.any_ready().await}),
    )
}
async fn models(State(state): State<AppState>) -> Json<Value> {
    // Every name that reaches a ready model; several can reach the same one.
    let data: Vec<Value> = state
        .model_names()
        .await
        .into_iter()
        .map(|id| json!({"id":id, "object":"model", "created":0, "owned_by":"local", "architecture":{"output_modalities":["decisions"]}}))
        .collect();
    Json(json!({"object":"list", "data":data}))
}
/// Routes a decision to the loaded model its `model` field names and returns
/// the result with the name the caller used.
async fn infer(state: &AppState, payload: Value) -> Result<(Value, String), ApiError> {
    validate_request(&payload).map_err(|e| ApiError(StatusCode::BAD_REQUEST, e))?;
    let requested = match payload.get("model") {
        None | Some(Value::Null) => None,
        Some(Value::String(model)) => Some(model.clone()),
        Some(_) => return Err(ApiError(StatusCode::BAD_REQUEST, "model must be a string.".into())),
    };
    let instance = state.route(requested.as_deref()).await.map_err(|error| match error {
        RouteError::Missing(m) => ApiError(StatusCode::BAD_REQUEST, m),
        RouteError::Unavailable(m) => ApiError(StatusCode::SERVICE_UNAVAILABLE, m),
        RouteError::Unknown(m) => ApiError(StatusCode::NOT_FOUND, m),
        RouteError::Ambiguous(m) => ApiError(StatusCode::CONFLICT, m),
    })?;
    let requested = requested.unwrap_or_default();
    let mut result = state
        .infer_on(&instance, payload)
        .await
        .map_err(|e| ApiError(StatusCode::BAD_GATEWAY, e))?;
    if let Some(model) = result.get_mut("model") {
        *model = json!(requested);
    }
    Ok((result, requested))
}
async fn systemone(State(state): State<AppState>, Json(payload): Json<Value>) -> ApiResult {
    infer(&state, payload).await.map(|(result, _)| Json(result))
}

pub fn chat_payload(body: &Value) -> Result<Value, String> {
    if body["stream"].as_bool() == Some(true) {
        return Err(
            "Decision models return one result; streaming is not supported. Set stream to false."
                .into(),
        );
    }
    if body.get("tools").is_some() || body.get("tool_choice").is_some() {
        return Err("Tool calling is not supported by the decision adapter.".into());
    }
    if body.get("n").is_some_and(|n| n != 1) {
        return Err("The decision adapter supports n=1 only.".into());
    }
    if let Some(format) = body.get("response_format") {
        if format["type"] != "json_object" {
            return Err("Use response_format type json_object, or omit it.".into());
        }
    }
    let messages = body["messages"].as_array().filter(|m| m.len() == 1).ok_or("Send one user message containing JSON with state and questions. Include all instructions in the questions object.")?;
    let message = &messages[0];
    if message["role"] != "user" {
        return Err(
            "The decision adapter expects one user message with a JSON decision request.".into(),
        );
    }
    let content = message["content"]
        .as_str()
        .ok_or("Message content must be a JSON string containing state and questions.")?;
    let mut payload: Value = serde_json::from_str(content)
        .map_err(|e| format!("Message content is not valid JSON: {e}"))?;
    validate_request(&payload)?;
    if let Some(model) = body.get("model") {
        payload["model"] = model.clone();
    }
    Ok(payload)
}

async fn chat(State(state): State<AppState>, Json(body): Json<Value>) -> ApiResult {
    let payload = chat_payload(&body).map_err(|e| ApiError(StatusCode::BAD_REQUEST, e))?;
    let (result, model) = infer(&state, payload).await?;
    let input = result["usage"]["input_tokens"].as_u64().unwrap_or(0);
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    Ok(Json(json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()), "object":"chat.completion", "created":created, "model":model,
        "choices":[{"index":0,"message":{"role":"assistant","content":serde_json::to_string(&result).unwrap_or_default()},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":input,"completion_tokens":0,"total_tokens":input}
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use tower::ServiceExt;
    #[tokio::test]
    async fn replacement_port_is_saved_and_relaunch_does_not_repeat_the_conflict() {
        let path =
            std::env::temp_dir().join(format!("noullama-remember-port-{}", uuid::Uuid::new_v4()));
        let mut state = AppState::new(path.clone()).unwrap();
        // Keep this two-launch fixture outside the kernel's ephemeral client
        // port range, so parallel HTTP tests cannot claim the freed endpoint.
        let mut reservation = None;
        for port in 22000..22500 {
            if let Ok(listener) =
                tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)).await
            {
                reservation = Some(listener);
                break;
            }
        }
        let occupied = reservation.expect("A test port outside the ephemeral range");
        let preferred = occupied.local_addr().unwrap().port();
        state.api_port = preferred;
        {
            let mut settings = state.settings.lock().await;
            settings.api_port = preferred;
            settings.runtime_port = preferred + 1;
        }
        start(state.clone()).await;
        let actual = state.listening_port();
        assert_ne!(actual, preferred);
        crate::lifecycle::shutdown_app(&state).await;
        let relaunched = AppState::new(path.clone()).unwrap();
        assert_eq!(
            relaunched.api_port, actual,
            "A replacement endpoint must survive restart"
        );
        start(relaunched.clone()).await;
        assert_eq!(relaunched.listening_port(), actual);
        assert!(relaunched.api_notice.lock().await.is_none());
        crate::lifecycle::shutdown_app(&relaunched).await;
        assert!(
            tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, preferred))
                .await
                .is_err()
        );
        drop(occupied);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[tokio::test]
    async fn occupied_preferred_port_uses_a_free_port_without_touching_its_owner() {
        let path =
            std::env::temp_dir().join(format!("noullama-port-conflict-{}", uuid::Uuid::new_v4()));
        let mut state = AppState::new(path.clone()).unwrap();
        let occupied = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        state.api_port = occupied.local_addr().unwrap().port();
        state.settings.lock().await.runtime_port = state.api_port.saturating_add(1);
        start(state.clone()).await;
        let port = state.listening_port();
        assert_ne!(port, state.api_port);
        assert_ne!(port, state.settings.lock().await.runtime_port);
        let response = state
            .local_client
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert!(state.api_error.lock().await.is_none());
        assert!(state
            .api_notice
            .lock()
            .await
            .as_ref()
            .unwrap()
            .contains(&port.to_string()));
        crate::lifecycle::shutdown_app(&state).await;
        assert!(
            tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
                .await
                .is_ok()
        );
        assert!(
            tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, state.api_port))
                .await
                .is_err()
        );
        drop(occupied);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[tokio::test]
    async fn runtime_cleanup_alone_does_not_release_the_gateway_listener() {
        // This is the old Exit callback's cleanup path. Unloading a model must
        // keep the gateway available, so full exit needs its own shutdown path.
        let path =
            std::env::temp_dir().join(format!("noullama-exit-baseline-{}", uuid::Uuid::new_v4()));
        let mut state = AppState::new(path.clone()).unwrap();
        let reservation = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        state.api_port = reservation.local_addr().unwrap().port();
        let server_state = state.clone();
        let server = tokio::spawn(async move { serve(reservation, server_state).await });
        let url = format!("http://127.0.0.1:{}/health", state.api_port);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while state.local_client.get(&url).send().await.is_err() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        state.shutdown().await;
        assert!(
            tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, state.api_port))
                .await
                .is_err()
        );
        server.abort();
        let _ = server.await;
        assert!(
            tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, state.api_port))
                .await
                .is_ok()
        );
        std::fs::remove_dir_all(path).unwrap();
    }
    #[tokio::test]
    async fn gateway_rejects_foreign_hosts_and_browser_origins() {
        let path = std::env::temp_dir().join(format!("noullama-origin-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(path.clone()).unwrap();
        let app = router(state);
        for (host, origin, expected) in [
            ("127.0.0.1:11435", None, StatusCode::OK),
            ("localhost:11435", None, StatusCode::OK),
            ("attacker.example:11435", None, StatusCode::FORBIDDEN),
            (
                "127.0.0.1:11435",
                Some("https://evil.example"),
                StatusCode::FORBIDDEN,
            ),
            ("127.0.0.1:11435", Some("null"), StatusCode::FORBIDDEN),
            (
                "127.0.0.1:11435",
                Some("http://localhost:11435"),
                StatusCode::OK,
            ),
            ("127.0.0.1:11436", None, StatusCode::FORBIDDEN),
        ] {
            let mut request = Request::builder().uri("/health").header("host", host);
            if let Some(origin) = origin {
                request = request.header("origin", origin);
            }
            let response = app
                .clone()
                .oneshot(request.body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                expected,
                "host={host}, origin={origin:?}"
            );
        }
        std::fs::remove_dir_all(path).unwrap();
    }
    fn request() -> Value {
        json!({"state":"charged twice", "questions":{"route":{"type":"choice","instructions":"Which team?", "criteria":{"billing":null,"technical":null}}}})
    }
    #[test]
    fn chat_keeps_state_and_questions_without_inventing_prompts() {
        let body = json!({"model":"d1-3B-Q8_0","messages":[{"role":"user","content":request().to_string()}]});
        let result = chat_payload(&body).unwrap();
        assert_eq!(result["questions"], request()["questions"]);
        assert_eq!(result["state"], "charged twice");
        assert!(chat_payload(&json!({"stream":true})).is_err());
        assert!(chat_payload(&json!({"messages":[{"role":"system","content":"foo"},{"role":"user","content":request().to_string()}]})).is_err());
    }
    #[test]
    fn validates_native_types_and_rejects_unsupported_features() {
        assert!(validate_request(&request()).is_ok());
        assert!(validate_request(&json!({"state":"", "questions":{}})).is_err());
        assert!(validate_request(
            &json!({"state":null, "questions":{"x":{"type":"text","instructions":"Extract"}}})
        )
        .is_err());
        assert!(validate_request(&json!({"state":"", "questions":{"x":{"type":"choice","instructions":"Choose", "multiple":true,"criteria":{"a":null,"b":null}}}})).is_err());
    }
    #[tokio::test]
    async fn unloaded_model_is_unavailable_and_models_is_empty() {
        let path = std::env::temp_dir().join(format!("noullama-test-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(path.clone()).unwrap();
        let app = router(state);
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header("host", "127.0.0.1:11435")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(body["data"], json!([]));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/systemone")
                    .header("host", "127.0.0.1:11435")
                    .header("content-type", "application/json")
                    .body(Body::from(request().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        // A request must name its model, loaded or not.
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        std::fs::remove_dir_all(path).unwrap();
    }

    async fn status_and_message(app: &Router, model: &str) -> (StatusCode, String) {
        let mut body = request();
        body["model"] = json!(model);
        let request = Request::builder().method("POST").uri("/v1/systemone").header("host", "127.0.0.1:11435").header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let body: Value = serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        (status, body["error"]["message"].as_str().unwrap_or_default().to_owned())
    }

    #[tokio::test]
    async fn requests_route_to_the_loaded_model_their_name_or_alias_points_to() {
        let path = std::env::temp_dir().join(format!("noullama-test-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(path.clone()).unwrap();
        let d1 = crate::runtime::insert_test_instance(&state, "/models/acme/d1-3B-Q8_0.gguf", true).await;
        let winnow = crate::runtime::insert_test_instance(&state, "/models/eldan/Winnow-E4B-Q8_0.gguf", true).await;
        crate::runtime::set_aliases(&state, d1.path().into(), vec!["triage".into()]).await.unwrap();
        crate::runtime::set_aliases(&state, winnow.path().into(), vec!["sorter".into()]).await.unwrap();
        crate::runtime::set_aliases(&state, "/models/other/unloaded.gguf".into(), vec!["later".into()]).await.unwrap();
        // Each name and alias belongs to one model file, and no two may differ
        // only by case.
        assert!(crate::runtime::set_aliases(&state, winnow.path().into(), vec!["Triage".into()]).await.unwrap_err().contains("used by both"));
        assert!(crate::runtime::set_aliases(&state, d1.path().into(), vec!["D1-3B-Q8_0".into()]).await.unwrap_err().contains("appears twice"));
        let app = router(state.clone());
        let get = |uri: &'static str| Request::builder().uri(uri).header("host", "127.0.0.1:11435").body(Body::empty()).unwrap();
        let response = app.clone().oneshot(get("/v1/models")).await.unwrap();
        let body: Value = serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        let ids: Vec<&str> = body["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["d1-3B-Q8_0", "triage", "Winnow-E4B-Q8_0", "sorter"]);
        for (name, expected) in [("triage", &d1), ("d1-3B-Q8_0", &d1), ("sorter", &winnow), ("Winnow-E4B-Q8_0", &winnow)] {
            assert_eq!(state.route(Some(name)).await.ok().map(|i| i.id.clone()), Some(expected.id.clone()), "{name}");
        }
        // Names are exact: no other casing, no default name, no missing name.
        for name in ["TRIAGE", "winnow-e4b-q8_0", "noullama", " triage"] {
            let (status, message) = status_and_message(&app, name).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{name}: {message}");
        }
        let (_, message) = status_and_message(&app, "Sorter").await;
        assert!(message.contains("did you mean 'sorter'"), "{message}");
        let (status, message) = status_and_message(&app, "").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(message.contains("d1-3B-Q8_0, triage"), "{message}");
        assert!(matches!(state.route(None).await, Err(RouteError::Missing(_))));
        let (status, message) = status_and_message(&app, "other").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(message.contains("d1-3B-Q8_0, triage, Winnow-E4B-Q8_0, sorter"), "{message}");
        let (status, message) = status_and_message(&app, "later").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(message.contains("unloaded, which is not loaded"), "{message}");
        // A routed name gets past routing; with no runtime it then fails upstream.
        for name in ["triage", "sorter", "d1-3B-Q8_0"] {
            assert_eq!(status_and_message(&app, name).await.0, StatusCode::BAD_GATEWAY, "{name}");
        }
        // Renaming a model in models.yml renames it in the API; the file stays as it is.
        let revision = state.library.lock().await.revision;
        let mut entries = state.library.lock().await.entries.clone();
        entries.iter_mut().find(|e| e.path == d1.path()).unwrap().name = "billing-router".into();
        crate::library::save(&state, entries.clone(), revision).await.unwrap();
        assert_eq!(state.route(Some("billing-router")).await.ok().map(|i| i.id.clone()), Some(d1.id.clone()));
        assert_eq!(status_and_message(&app, "d1-3B-Q8_0").await.0, StatusCode::NOT_FOUND);
        // An editor working from an older copy cannot overwrite the newer file.
        assert!(crate::library::save(&state, entries, revision).await.unwrap_err().contains("changed since you started editing"));
        crate::runtime::unload(&state, &d1.id).await.unwrap();
        assert_eq!(status_and_message(&app, "triage").await.0, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(status_and_message(&app, "billing-router").await.0, StatusCode::SERVICE_UNAVAILABLE);
        state.shutdown().await;
        std::fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test]
    async fn a_file_name_shared_by_two_loaded_models_needs_an_alias() {
        let path = std::env::temp_dir().join(format!("noullama-test-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(path.clone()).unwrap();
        let first = crate::runtime::insert_test_instance(&state, "/a/model-Q4_K_M.gguf", true).await;
        crate::runtime::insert_test_instance(&state, "/b/model-Q4_K_M.gguf", true).await;
        let loading = crate::runtime::insert_test_instance(&state, "/c/slow.gguf", false).await;
        let app = router(state.clone());
        let (status, message) = status_and_message(&app, "model-Q4_K_M").await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert!(message.contains("/a/model-Q4_K_M.gguf, /b/model-Q4_K_M.gguf"), "{message}");
        assert!(!state.model_names().await.iter().any(|n| n == "model-Q4_K_M"));
        assert_eq!(status_and_message(&app, "slow").await.0, StatusCode::SERVICE_UNAVAILABLE);
        crate::runtime::set_aliases(&state, first.path().into(), vec!["fast".into()]).await.unwrap();
        assert_eq!(state.route(Some("fast")).await.ok().map(|i| i.id.clone()), Some(first.id.clone()));
        assert!(!loading.is_ready());
        state.shutdown().await;
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn aliases_are_trimmed_checked_and_deduplicated() {
        let names = crate::runtime::normalize_aliases(&[" triage ".into(), "Triage".into(), "".into(), "noullama".into(), "acme/d1:q8".into()]).unwrap();
        assert_eq!(names, ["triage", "noullama", "acme/d1:q8"]);
        assert!(crate::runtime::normalize_aliases(&["has space".into()]).is_err());
    }

    #[tokio::test]
    async fn settings_form_cannot_overwrite_aliases_or_keep_context() {
        let path = std::env::temp_dir().join(format!("noullama-test-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(path.clone()).unwrap();
        let stale = state.settings.lock().await.clone();
        {
            let mut settings = state.settings.lock().await;
            settings.model_routes.insert("/m/a.gguf".into(), vec!["triage".into()]);
            settings.keep_context = true;
        }
        let mut edited = stale;
        edited.context_size = 4096;
        crate::runtime::save_from_editor(&state, edited).await.unwrap();
        let settings = state.settings.lock().await.clone();
        assert_eq!(settings.context_size, 4096);
        assert_eq!(settings.model_routes["/m/a.gguf"], ["triage"]);
        assert!(settings.keep_context);
        std::fs::remove_dir_all(path).unwrap();
    }

    /// Loads two different models at once, routes requests to each by alias,
    /// then ejects one and checks the other keeps answering.
    #[tokio::test]
    #[ignore = "requires NOULLAMA_TEST_MODEL, NOULLAMA_TEST_MODEL_2 and a local llama-server runtime"]
    async fn real_runtime_two_models_route_by_alias_and_eject_one() {
        let env_layers = |name: &str| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let path = std::env::temp_dir().join(format!("noullama-two-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(path.clone()).unwrap();
        let mut ids = Vec::new();
        for (model, layers) in [("NOULLAMA_TEST_MODEL", "NOULLAMA_TEST_GPU_LAYERS"), ("NOULLAMA_TEST_MODEL_2", "NOULLAMA_TEST_GPU_LAYERS_2")] {
            {
                let mut settings = state.settings.lock().await;
                settings.runtime_path = std::env::var("NOULLAMA_TEST_RUNTIME").unwrap_or_default();
                settings.model_path = std::env::var(model).unwrap_or_else(|_| panic!("Set {model}"));
                settings.gpu_layers = env_layers(layers);
                settings.context_size = 2048;
                settings.runtime_port = 18440;
            }
            // The second load starts while the first is still loading.
            ids.push(crate::runtime::load_model(&state).await.unwrap());
        }
        let first = crate::runtime::wait_until_ready(&state, &ids[0], 300).await;
        let second = crate::runtime::wait_until_ready(&state, &ids[1], 300).await;
        let ports: Vec<u16> = state.instances.lock().await.iter().map(|i| i.settings.runtime_port).collect();
        assert_eq!(ports, [18440, 18440], "both keep the configured starting port in their settings");
        let busy = |port: u16| std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)).is_err();
        assert!(busy(18440) && busy(18441), "each model runs on its own private port");
        crate::runtime::set_aliases(&state, first.path().into(), vec!["triage".into()]).await.unwrap();
        crate::runtime::set_aliases(&state, second.path().into(), vec!["sorter".into()]).await.unwrap();
        let app = router(state.clone());
        let ask = |model: Option<&str>| {
            let mut body = json!({"state":"I was charged twice this month. Please refund the duplicate payment.",
                "questions":{"route":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and refunds","technical":"Bugs and errors"}}}});
            if let Some(model) = model {
                body["model"] = json!(model);
            }
            let app = app.clone();
            async move {
                let request = Request::builder().method("POST").uri("/v1/systemone").header("host", "127.0.0.1:11435").header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
                let response = app.oneshot(request).await.unwrap();
                let status = response.status();
                let body: Value = serde_json::from_slice(&to_bytes(response.into_body(), 65536).await.unwrap()).unwrap();
                (status, body)
            }
        };
        let first_name = first.stem();
        for model in ["triage", "sorter", first_name.as_str()] {
            let (status, body) = ask(Some(model)).await;
            println!("{model}: {status} {body}");
            assert_eq!(status, StatusCode::OK, "{model}: {body}");
            assert_eq!(body["answers"]["route"]["choice"], "billing", "{model}: {body}");
            assert_eq!(body["model"], model);
        }
        assert_eq!(ask(None).await.0, StatusCode::BAD_REQUEST);
        assert_eq!(ask(Some("TRIAGE")).await.0, StatusCode::NOT_FOUND);
        crate::runtime::unload(&state, &first.id).await.unwrap();
        assert!(!busy(18440), "the ejected model's port is released");
        assert!(busy(18441));
        let (status, body) = ask(Some("sorter")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(ask(Some("triage")).await.0, StatusCode::SERVICE_UNAVAILABLE);
        state.shutdown().await;
        assert!(!busy(18441));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test]
    #[ignore = "requires NOULLAMA_TEST_MODEL and a local llama-server runtime"]
    async fn real_runtime_native_and_openai_requests() {
        let model = std::env::var("NOULLAMA_TEST_MODEL").expect("Set NOULLAMA_TEST_MODEL");
        let runtime = std::env::var("NOULLAMA_TEST_RUNTIME").unwrap_or_default();
        let path =
            std::env::temp_dir().join(format!("noullama-inference-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(path.clone()).unwrap();
        {
            let mut settings = state.settings.lock().await;
            settings.runtime_path = runtime;
            settings.model_path = model;
            settings.gpu_layers = std::env::var("NOULLAMA_TEST_GPU_LAYERS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            settings.context_size = std::env::var("NOULLAMA_TEST_CONTEXT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8192);
            settings.runtime_port = 18437;
        }
        let id = crate::runtime::load_model(&state).await.unwrap();
        let instance = crate::runtime::wait_until_ready(&state, &id, 90).await;
        let private_port = state.settings.lock().await.runtime_port;
        let unauthenticated = state
            .local_client
            .post(format!("http://127.0.0.1:{private_port}/v1/systemone"))
            .json(&request())
            .send()
            .await
            .unwrap();
        assert_eq!(unauthenticated.status(), reqwest::StatusCode::UNAUTHORIZED);
        let app = router(state.clone());
        let loaded = instance.model();
        state.settings.lock().await.model_path = "/a/different/selected.gguf".into();
        state.settings.lock().await.runtime_path = "/a/different/runtime".into();
        assert_eq!(instance.model().path, loaded.path);
        assert_eq!(instance.model().runtime_path, loaded.runtime_path);
        let name = instance.stem();
        let mut decision_request = request();
        decision_request["model"] = json!(name);
        decision_request["state"] =
            json!("I was charged twice. Please refund the duplicate payment.");
        decision_request["questions"]["refund"] =
            json!({"type":"noul", "instructions":"Is the customer requesting a refund?"});
        decision_request["questions"]["urgency"] = json!({"type":"score", "instructions":"How urgent is the request?", "criteria":["Can wait", "Today", "Immediately"]});
        let native = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/systemone")
                    .header("host", "127.0.0.1:11435")
                    .header("content-type", "application/json")
                    .body(Body::from(decision_request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = native.status();
        let result: Value =
            serde_json::from_slice(&to_bytes(native.into_body(), 65536).await.unwrap()).unwrap();
        let chat_body = json!({"model": name, "messages":[{"role":"user", "content":decision_request.to_string()}]});
        let chat = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("host", "127.0.0.1:11435")
                    .header("content-type", "application/json")
                    .body(Body::from(chat_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let chat_status = chat.status();
        let envelope: Value =
            serde_json::from_slice(&to_bytes(chat.into_body(), 65536).await.unwrap()).unwrap();
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = std::time::Instant::now();
            state.infer_on(&instance, decision_request.clone()).await.unwrap();
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        println!("Warm app-backend latency samples (ms): {samples:?}");
        state.shutdown().await;
        assert!(state.instances.lock().await.is_empty());
        std::fs::remove_dir_all(path).unwrap();
        assert_eq!(status, StatusCode::OK, "Native response: {result}");
        assert_eq!(result["answers"]["route"]["choice"], "billing");
        assert_eq!(result["model"], name);
        assert!(result["answers"]["refund"]["noul"].as_f64().is_some());
        assert!(result["answers"]["urgency"]["score"].as_f64().is_some());
        assert_eq!(chat_status, StatusCode::OK, "OpenAI response: {envelope}");
        assert_eq!(envelope["object"], "chat.completion");
        let wrapped: Value = serde_json::from_str(
            envelope["choices"][0]["message"]["content"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(wrapped["answers"]["route"]["choice"], "billing");
        assert_eq!(envelope["usage"]["completion_tokens"], 0);
        println!("Native decision response: {result}");
        println!("OpenAI adapter response: {envelope}");
    }
}
