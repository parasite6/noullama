//! Keep context: earlier decisions ride along in each new request's state, so
//! the model sees what it decided before. Each loaded model keeps its own
//! history, shared by the Playground and every local API caller that reaches
//! it. When a request no longer fits the context window, the oldest decisions
//! are flushed until it does.

use crate::runtime::{AppState, Instance};
use std::sync::Arc;
use serde_json::{json, Map, Value};
use std::collections::VecDeque;

/// Decisions kept at most, whatever the context size.
const MAX_ENTRIES: usize = 64;

#[derive(Clone)]
pub struct Entry {
    seq: u64,
    /// The caller's own state, never one that already carries history.
    state: Value,
    /// Question name to a short answer, such as "billing" or "yes (0.82)".
    decisions: Map<String, Value>,
}

#[derive(Default)]
pub struct History {
    entries: VecDeque<Entry>,
    next_seq: u64,
    /// Decisions flushed to make the most recent request fit.
    pub last_flushed: usize,
}

impl History {
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn clear(&mut self) {
        self.entries.clear();
        self.last_flushed = 0;
    }
    fn push(&mut self, state: Value, decisions: Map<String, Value>) {
        self.next_seq += 1;
        self.entries.push_back(Entry { seq: self.next_seq, state, decisions });
        while self.entries.len() > MAX_ENTRIES {
            self.entries.pop_front();
        }
    }
    /// Drops every entry up to and including `seq`; returns how many went.
    fn flush_through(&mut self, seq: u64) -> usize {
        let before = self.entries.len();
        self.entries.retain(|entry| entry.seq > seq);
        before - self.entries.len()
    }
}

/// A short, readable form of each answer for later requests.
pub fn summarize(result: &Value) -> Map<String, Value> {
    let mut out = Map::new();
    for (name, answer) in result["answers"].as_object().into_iter().flatten() {
        let short = if let Some(choice) = answer["choice"].as_str() {
            choice.to_owned()
        } else if let Some(p) = answer["noul"].as_f64() {
            if p >= 0.5 { format!("yes ({p:.2})") } else { format!("no ({:.2})", 1.0 - p) }
        } else if let Some(score) = answer["score"].as_f64() {
            format!("{score:.2}")
        } else {
            continue;
        };
        out.insert(name.clone(), json!(short));
    }
    out
}

fn as_text(state: &Value) -> String {
    match state {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// The state to send: the current state with earlier decisions before it, in
/// the same shape the caller used (text stays text, JSON stays JSON).
pub fn with_history(state: &Value, earlier: &[Entry]) -> Value {
    if earlier.is_empty() {
        return state.clone();
    }
    match state {
        Value::Object(_) | Value::Array(_) => json!({
            "earlier_decisions": earlier.iter().map(|e| json!({"state": e.state, "decisions": e.decisions})).collect::<Vec<_>>(),
            "current_state": state,
        }),
        _ => {
            let mut text = String::from("Earlier decisions, oldest first:\n");
            for (index, entry) in earlier.iter().enumerate() {
                let decisions = entry
                    .decisions
                    .iter()
                    .map(|(name, value)| format!("{name} = {}", as_text(value)))
                    .collect::<Vec<_>>()
                    .join("; ");
                text.push_str(&format!("{}. State: {}\n   Decisions: {decisions}\n", index + 1, as_text(&entry.state)));
            }
            text.push_str("\nCurrent state:\n");
            text.push_str(&as_text(state));
            json!(text)
        }
    }
}

/// Why a request failed: the context window was full, or anything else.
pub enum Failure {
    ContextFull(String),
    Other(String),
}

/// Runs one decision, adding earlier decisions when Keep context is on.
pub async fn infer(state: &AppState, instance: &Arc<Instance>, payload: Value) -> Result<Value, String> {
    if !state.settings.lock().await.keep_context {
        return state.infer_once(instance, payload, false).await.map_err(|failure| match failure {
            Failure::ContextFull(message) | Failure::Other(message) => message,
        });
    }
    let earlier: Vec<Entry> = instance.context.lock().await.entries.iter().cloned().collect();
    // Leave out the oldest decision each time the request does not fit. The
    // history only loses them once a request succeeds, so a state too long to
    // fit even alone fails without wiping the kept decisions.
    let mut skipped = 0;
    loop {
        let carried = &earlier[skipped..];
        let mut request = payload.clone();
        request["state"] = with_history(&payload["state"], carried);
        match state.infer_once(instance, request, !carried.is_empty()).await {
            Ok(result) => {
                let mut history = instance.context.lock().await;
                history.last_flushed = match skipped.checked_sub(1).map(|last| earlier[last].seq) {
                    Some(seq) => history.flush_through(seq),
                    None => 0,
                };
                history.push(payload["state"].clone(), summarize(&result));
                return Ok(result);
            }
            Err(Failure::ContextFull(message)) if carried.is_empty() => return Err(message),
            Err(Failure::ContextFull(_)) => skipped += 1,
            Err(Failure::Other(message)) => return Err(message),
        }
    }
}

#[tauri::command]
pub async fn set_keep_context(on: bool, state: tauri::State<'_, AppState>) -> Result<(), String> {
    crate::library::refresh(state.inner()).await;
    let mut settings = state.settings.lock().await.clone();
    settings.keep_context = on;
    crate::runtime::persist_settings(state.inner(), settings).await?;
    if !on {
        clear_all(state.inner()).await;
    }
    Ok(())
}

async fn clear_all(state: &AppState) {
    let instances = state.instances.lock().await.clone();
    for instance in instances {
        instance.context.lock().await.clear();
    }
}

/// Empties one loaded model's history, or every model's when `model` is None.
#[tauri::command]
pub async fn clear_context(model: Option<String>, state: tauri::State<'_, AppState>) -> Result<(), String> {
    match model {
        Some(id) => {
            if let Some(instance) = state.instance(&id).await {
                instance.context.lock().await.clear();
            }
        }
        None => clear_all(state.inner()).await,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(seq: u64, state: Value, decisions: Value) -> Entry {
        Entry { seq, state, decisions: decisions.as_object().unwrap().clone() }
    }

    #[test]
    fn text_state_lists_earlier_decisions_before_the_current_state() {
        let earlier = [entry(1, json!("charged twice"), json!({"route":"billing"}))];
        let state = with_history(&json!("cannot log in"), &earlier);
        assert_eq!(
            state,
            json!("Earlier decisions, oldest first:\n1. State: charged twice\n   Decisions: route = billing\n\nCurrent state:\ncannot log in")
        );
        assert_eq!(with_history(&json!("alone"), &[]), json!("alone"));
    }

    #[test]
    fn json_state_stays_json() {
        let earlier = [entry(1, json!({"msg":"a"}), json!({"urgent":"yes (0.90)"}))];
        let state = with_history(&json!({"msg":"b"}), &earlier);
        assert_eq!(state["current_state"], json!({"msg":"b"}));
        assert_eq!(state["earlier_decisions"][0]["state"], json!({"msg":"a"}));
        assert_eq!(state["earlier_decisions"][0]["decisions"]["urgent"], "yes (0.90)");
    }

    #[test]
    fn answers_are_summarized_by_type() {
        let result = json!({"answers":{
            "route":{"choice":"billing","probabilities":{"billing":0.9}},
            "urgent":{"noul":0.2},
            "level":{"score":0.734}
        }});
        let summary = summarize(&result);
        assert_eq!(summary["route"], "billing");
        assert_eq!(summary["urgent"], "no (0.80)");
        assert_eq!(summary["level"], "0.73");
    }

    #[test]
    fn history_flushes_oldest_and_caps_its_length() {
        let mut history = History::default();
        for n in 0..(MAX_ENTRIES + 5) {
            history.push(json!(n), Map::new());
        }
        assert_eq!(history.len(), MAX_ENTRIES);
        assert_eq!(history.entries[0].state, json!(5));
        let oldest = history.entries[0].seq;
        assert_eq!(history.flush_through(oldest), 1);
        assert_eq!(history.entries[0].state, json!(6));
    }

    /// Decides on a real runtime at a 512-token context with Keep context on:
    /// earlier decisions are flushed oldest first and every request succeeds.
    #[tokio::test]
    #[ignore = "requires NOULLAMA_TEST_MODEL and a local llama-server runtime"]
    async fn real_runtime_flushes_oldest_decisions_at_the_context_limit() {
        let path = std::env::temp_dir().join(format!("noullama-context-{}", uuid::Uuid::new_v4()));
        let state = AppState::new(path.clone()).unwrap();
        {
            let mut settings = state.settings.lock().await;
            settings.runtime_path = std::env::var("NOULLAMA_TEST_RUNTIME").unwrap_or_default();
            settings.model_path = std::env::var("NOULLAMA_TEST_MODEL").expect("Set NOULLAMA_TEST_MODEL");
            settings.gpu_layers = 0;
            settings.context_size = 512;
            settings.runtime_port = 18438;
            settings.keep_context = true;
        }
        let id = crate::runtime::load_model(&state).await.unwrap();
        let model = crate::runtime::wait_until_ready(&state, &id, 120).await;
        let questions = json!({"route":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and refunds","technical":"Bugs and errors"}}});
        let mut results = Vec::new();
        let mut most_flushed = 0;
        for n in 0..12 {
            let state_text = format!("Ticket {n}: I was charged twice for my subscription this month and the duplicate payment put my account into overdraft. Please refund the second charge as soon as possible, it is urgent.");
            results.push(state.infer_on(&model, json!({"state": state_text, "questions": questions})).await);
            most_flushed = most_flushed.max(model.context.lock().await.last_flushed);
        }
        let kept = model.context.lock().await.len();
        // A state too long to fit even alone fails and keeps the history intact.
        let oversized = state.infer_on(&model, json!({"state": "charged twice. ".repeat(400), "questions": questions})).await;
        let kept_after_failure = model.context.lock().await.len();
        let tokens: Vec<_> = results.iter().map(|r| r.as_ref().map(|v| v["usage"]["input_tokens"].clone()).unwrap_or_default()).collect();
        println!("kept {kept}, flushed up to {most_flushed}, input tokens {tokens:?}");
        state.shutdown().await;
        std::fs::remove_dir_all(path).unwrap();
        for result in &results {
            assert_eq!(result.as_ref().unwrap()["answers"]["route"]["choice"], "billing", "{result:?}");
        }
        assert!(most_flushed > 0, "the context never filled");
        assert!(oversized.is_err());
        assert_eq!(kept_after_failure, kept);
        assert!(kept < 12, "kept {kept}");
        println!("kept {kept} decisions, flushed up to {most_flushed} at once");
    }
}
