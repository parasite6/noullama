<h1 align="center">Noullama</h1>

<p align="center">
  <img src="logo33%25.png" alt="noullama logo">
</p>

<p align="center">A local workbench and API for native decision models using llama.cpp</p>

[Quick start](#quick-start) · [Cheat sheet](#cheat-sheet) · [Requirements](#requirements) · [Getting started](#getting-started) · [Models and presets](#models-and-presets) · [Local API](#local-api) · [Data and verification](#data-and-verification)

## Quick start

```sh
npm install
npm run desktop
```

In the app: **Models** → download or browse for a GGUF → **Runtime** → select your llama.cpp path → **Load selected** → **Playground**.

## Cheat sheet

| I want to… | Do this |
| --- | --- |
| Download a model | Open **Models**, inspect a decision-model repository, and select a GGUF file. |
| Use an existing model | Choose **Browse file**, select your GGUF, then click **Load selected**. |
| Run a decision | Choose a preset in **Playground**, edit the questions and state, then click **Run decision**. |
| Save a preset | Enter a name beneath the questions editor and click **Save custom**. |
| Use CPU inference | Set **GPU layers** to `0` in **Runtime**. |
| Connect an application | Use `http://127.0.0.1:11435/v1`; native decisions are at `/systemone`. |

Question types: **`choice`** selects an option, **`noul`** answers yes/no, and **`score`** rates an ordered set of levels.

## Features

- Discover and download decision-model GGUFs from Hugging Face using an existing login or a session token.
- Load a local model with a compatible llama.cpp runtime and configure GPU offloading and an optional multimodal projector.
- Run typed questions against text or JSON state using editable presets, JSON linting, and syntax highlighting.
- Save, update, and delete named custom presets.
- Connect through a local System One endpoint or an OpenAI-compatible chat adapter.

## Requirements

- Node.js 22+, Rust, and Blaze for the provided build commands.
- The Hugging Face CLI (`hf`) for downloads. Public models do not require a login; gated models require access and authentication.
- A compatible `llama-server`. Liquid d1-2B requires a build containing [llama.cpp PR #31275](https://github.com/ggml-org/llama.cpp/pull/31275), such as b11954 or a compatible newer build.
- On Linux, the [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/#linux), including GTK 3 and WebKitGTK 4.1 development packages.

## Getting started

To build a standalone executable:

```sh
npm run desktop:build             # release
npm run desktop:build -- --debug  # debug
```

On Linux, the binaries are written to `src-tauri/target/release/noullama` and `src-tauri/target/debug/noullama`. Standalone builds include the frontend and do not require a Vite server. Development builds use a separate target directory. Packaging is not yet configured.

1. Open **Models**, search for a decision model or inspect a repository such as `LiquidAI/d1-2B-GGUF`, and download a GGUF. You can also browse for an existing local file.
2. In **Runtime**, set the path to your compatible llama.cpp directory or `llama-server` executable. Configure GPU layers and, if needed, a multimodal projector. Use **Check runtime** to inspect available devices; set GPU layers to `0` for CPU inference.
3. Select a model and click **Load selected** in the model bar. The bar shows the loaded model and device and provides controls to switch or unload it.
4. In **Playground**, choose a preset, edit the questions and state, and run a decision.

The initial runtime path is machine-specific; change it in **Runtime** on another machine. For GPU builds, configure **CUDA / shared-library directories** if the runtime cannot find its libraries. On Linux, noullama also checks common CUDA 13 locations and passes the library paths to the runtime process.

## Models and presets

Hugging Face discovery uses decision-model tags and architecture metadata. Downloads use `hf download` at the inspected repository revision. noullama supports the standard Hugging Face token locations and `HF_TOKEN`; a token entered in the app is held for the session and passed to `hf` through its environment.

Set an absolute **Model download directory** in **Models** to choose where new downloads go. Changing it does not move existing files. The library scans the current download directory and the original app-data models directory. Download progress reports file reconstruction and network bytes separately; Xet compression and deduplication can make those values differ.

Questions use the `choice`, `noul`, and `score` types. The JSON editors provide syntax highlighting and inline lint diagnostics. The state editor accepts text or JSON; switching modes converts the representation while preserving the content.

Enter a name beneath the questions editor and choose **Save custom** to save the questions, state, and state format across restarts. Select a saved preset to update it, or use its trash button to delete it. Built-in presets are starting points; calibrate decision thresholds and criteria against your own data.

## Local API

While the app is open, the gateway listens on `http://127.0.0.1:11435` by default. The managed llama-server uses port `11436`. Both bind to localhost; changing the gateway port requires an app restart.

Native System One request:

```sh
curl http://127.0.0.1:11435/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "I was charged twice. Please refund the duplicate payment.",
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges and refunds", "technical": "App errors"}
      },
      "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}
    }
  }'
```

The gateway also exposes `/v1/models`, `/v1/chat/completions`, and `/health`. The OpenAI adapter expects one user message containing a JSON object with `state` and `questions`; the assistant message contains the native decision result as JSON.

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="local")
request = {
    "state": "I was charged twice. Please refund the duplicate payment.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}
    },
}
response = client.chat.completions.create(
    model="noullama",
    messages=[{"role": "user", "content": json.dumps(request)}],
    stream=False,
)
print(json.loads(response.choices[0].message.content)["answers"])
```

The adapter does not support chat history, tools, streaming, or arbitrary chat prompts. Multi-select, extraction, estimates, boxes, and video are not supported by this integration. Native clients can provide image data URLs when the selected model, runtime, and projector support them.

## Data and verification

Settings, saved presets, and downloads use the platform app-data directory (`~/.local/share/dev.noullama.desktop` on a typical Linux desktop). Existing GGUFs are used in place.

```sh
npm run build
npm test
blaze test --manifest-path src-tauri/Cargo.toml
```

An optional real-model integration test can be run with `NOULLAMA_TEST_MODEL` and `NOULLAMA_TEST_RUNTIME` set to absolute paths. Set `NOULLAMA_TEST_GPU_LAYERS=99` to test GPU offloading:

```sh
blaze test --manifest-path src-tauri/Cargo.toml \
  real_runtime_native_and_openai_requests -- --ignored --nocapture
```

The current prototype supports one loaded model and one download at a time. It has no download cancellation UI, bundled runtime, installer, or automatic playground history. Named custom presets persist.
