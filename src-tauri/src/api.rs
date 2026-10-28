use crate::runtime::{validate_request, AppState};
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
        json!({"status":"ok", "service":"noullama", "api_port":state.listening_port(), "model_ready": *state.ready.lock().await}),
    )
}
async fn models(State(state): State<AppState>) -> Json<Value> {
    let data = if *state.ready.lock().await {
        vec![
            json!({"id":"noullama", "object":"model", "created":0, "owned_by":"local", "architecture":{"output_modalities":["decisions"]}}),
        ]
    } else {
        vec![]
    };
    Json(json!({"object":"list", "data":data}))
}
async fn infer(state: &AppState, payload: Value) -> ApiResult {
    validate_request(&payload).map_err(|e| ApiError(StatusCode::BAD_REQUEST, e))?;
    if let Some(model) = payload.get("model") {
        if model != "noullama" {
            return Err(ApiError(
                StatusCode::NOT_FOUND,
                "Use model 'noullama'; only the loaded model is available.".into(),
            ));
        }
    }
    if !*state.ready.lock().await {
        return Err(ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "No decision model is ready. Load one in noullama.".into(),
        ));
    }
    state
        .infer(payload)
        .await
        .map(Json)
        .map_err(|e| ApiError(StatusCode::BAD_GATEWAY, e))
}
async fn systemone(State(state): State<AppState>, Json(payload): Json<Value>) -> ApiResult {
    infer(&state, payload).await
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
    payload["model"] = body.get("model").cloned().unwrap_or(json!("noullama"));
    Ok(payload)
}

async fn chat(State(state): State<AppState>, Json(body): Json<Value>) -> ApiResult {
    let payload = chat_payload(&body).map_err(|e| ApiError(StatusCode::BAD_REQUEST, e))?;
    let Json(result) = infer(&state, payload).await?;
    let input = result["usage"]["input_tokens"].as_u64().unwrap_or(0);
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    Ok(Json(json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()), "object":"chat.completion", "created":created, "model":"noullama",
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
        let body = json!({"model":"noullama","messages":[{"role":"user","content":request().to_string()}]});
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
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test]
    #[ignore = "requires NOULLAMA_TEST_MODEL and a local llama-server runtime"]
    async fn real_runtime_native_and_openai_requests() {
        let model = std::env::var("NOULLAMA_TEST_MODEL").expect("Set NOULLAMA_TEST_MODEL");
        let runtime = std::env::var("NOULLAMA_TEST_RUNTIME")
            .unwrap_or_else(|_| crate::runtime::DEFAULT_RUNTIME.into());
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
        crate::runtime::load_model(&state).await.unwrap();
        for _ in 0..180 {
            if *state.ready.lock().await {
                break;
            }
            if state.session.lock().await.is_none() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        if !*state.ready.lock().await {
            let logs = state.logs.lock().await.clone();
            state.shutdown().await;
            panic!("Model failed to start: {logs:#?}");
        }
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
        let loaded_path = state
            .loaded_model
            .lock()
            .await
            .as_ref()
            .unwrap()
            .path
            .clone();
        let loaded_runtime = state
            .loaded_model
            .lock()
            .await
            .as_ref()
            .unwrap()
            .runtime_path
            .clone();
        state.settings.lock().await.model_path = "/a/different/selected.gguf".into();
        state.settings.lock().await.runtime_path = "/a/different/runtime".into();
        assert_eq!(
            state.loaded_model.lock().await.as_ref().unwrap().path,
            loaded_path
        );
        assert_eq!(
            state
                .loaded_model
                .lock()
                .await
                .as_ref()
                .unwrap()
                .runtime_path,
            loaded_runtime
        );
        let mut decision_request = request();
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
        let chat_body = json!({"model":"noullama", "messages":[{"role":"user", "content":decision_request.to_string()}]});
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
            state.infer(decision_request.clone()).await.unwrap();
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        println!("Warm app-backend latency samples (ms): {samples:?}");
        state.shutdown().await;
        assert!(state.loaded_model.lock().await.is_none());
        std::fs::remove_dir_all(path).unwrap();
        assert_eq!(status, StatusCode::OK, "Native response: {result}");
        assert_eq!(result["answers"]["route"]["choice"], "billing");
        assert_eq!(result["model"], "noullama");
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
