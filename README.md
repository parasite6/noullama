<p align="center">
  <img src="logo33%25.png" alt="noullama logo">
</p>

<h1 align="center">Noullama</h1>

<p align="center">A local workbench and API for native decision models using llama.cpp</p>

[Quick start](#quick-start) · [Cheat sheet](#cheat-sheet) · [Features](#features) · [Requirements](#requirements) · [Getting started](#getting-started) · [Runtime](#runtime) · [Models and presets](#models-and-presets) · [Local API](#local-api) · [Data and verification](#data-and-verification)

## Quick start

```sh
git clone https://github.com/parasite6/noullama.git
cd noullama
npm install
npm run desktop
```

In the app: **Hub** → download a GGUF (or **Models** → browse for a local file) → **Runtime** → select your llama.cpp path → **Load selected** → **Playground**.

## Cheat sheet

| I want to… | Do this |
| --- | --- |
| Download a model | Open **Hub**, inspect a decision-model repository, and select a GGUF file. |
| Use an existing model | Open **Models**, choose **Browse local model**, then click **Load selected**. |
| Run a decision | Choose a preset in **Playground**, edit the questions and state, then click **Run decision**. |
| Save a preset | Enter a name beneath the questions editor and click **Save custom**. |
| Use CPU inference | Set **GPU layers** to `0` in **Runtime**. |
| Connect an application | Use `http://127.0.0.1:11435/v1`; native decisions are at `/systemone`. |

Question types:

- **`choice`** selects an option.
- **`noul`** answers yes/no.
- **`score`** rates an ordered set of levels.

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

### Install or build

Tagged releases publish `.deb` and `.rpm` packages for 64-bit Linux, linked from [noullama.dev/download](https://noullama.dev/download/).

To build a standalone executable instead:

```sh
npm run desktop:build             # release
npm run desktop:build -- --debug  # debug
```

- On Linux, the binaries are written to `src-tauri/target/release/noullama` and `src-tauri/target/debug/noullama`.
- Standalone builds include the frontend and do not require a Vite server.
- Development builds use a separate target directory.

### First run

1. Open **Hub**, search for a decision model or inspect a repository such as `LiquidAI/d1-2B-GGUF`, and download a GGUF. Already have a file? Open **Models** and browse for it instead.
2. In **Runtime**, set the path to your compatible llama.cpp directory or `llama-server` executable. Configure GPU layers and, if needed, a multimodal projector. Use **Check runtime** to inspect available devices; set GPU layers to `0` for CPU inference.
3. Select a model and click **Load selected** in the model bar. The bar shows the loaded model and device and provides controls to switch or unload it.
4. In **Playground**, choose a preset, edit the questions and state, and run a decision.

## Runtime

### Finding llama.cpp

The runtime path starts unset. Noullama searches managed installs, nearby builds, and `PATH`; select a discovered runtime or enter a custom path in **Runtime**.

For GPU builds, configure **CUDA / shared-library directories** if the runtime cannot find its libraries. On Linux, noullama also checks common CUDA 12 and 13 locations and passes the library paths to the runtime process.

The **Selected llama.cpp** control on every page identifies a custom or downloaded runtime. While a model is running, it shows the build in use and marks any different selection as not applied.

### Downloading llama.cpp builds

**Runtime → llama.cpp builds** scans the configured location, nearby builds, PATH, and managed installs before offering a download.

1. Choose a backend and **Check official releases** to preview OS/architecture-matched assets from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases). Checking fetches metadata only.
2. **Download nightly + fallback** explicitly approves the displayed downloads: the latest nightly and the binary build linked by the latest official release. The fallback uses CPU, or Metal on macOS.
3. Select **Use** beside an installed runtime, then check or load it normally.

Archives are checked against GitHub's SHA-256 digests and installed under the app-data `runtimes/` directory; existing custom paths are preserved. Linux uses the official Ubuntu-built archives; unsupported platforms can use a custom build.

### Tray and lifecycle

On desktops with an active system tray, closing the window keeps the model and local API running. Use **Open Noullama** in the tray menu to restore the window.

**Quit Noullama**, available in the tray menu and sidebar, opens a centered confirmation. Confirming shuts down the API, owned runtime, and active downloads before exiting; **Keep running** or Escape cancels.

Linux tray mode needs an AppIndicator library and a desktop tray host. GNOME setups without tray support may need an AppIndicator extension. If the library or host is unavailable, closing the window asks for confirmation before quitting.

Noullama releases its own ports on full exit; a port occupied by another service remains that service's responsibility.

On Linux, relaunching the same executable with the same app-data directory shuts down its older instance before starting the API. Startup also cleans up orphaned llama-server processes carrying Noullama's runtime identity and app-data key-file markers. Processes belonging to other applications or app-data directories are left running.

## Models and presets

### Hub and downloads

Hugging Face discovery uses decision-model tags and architecture metadata. Downloads use `hf download` at the inspected repository revision. noullama supports the standard Hugging Face token locations and `HF_TOKEN`; a token entered in the app is held for the session and passed to `hf` through its environment.

Author logos are fetched from each author's public Hugging Face profile the first time they appear in the Hub or a downloaded model, then cached for the session. Images are filtered at the display's pixel density for sharper small icons; authors without a profile image show their initials. No author artwork is bundled with the app.

Hub search results show a local memory-fit estimate when a repository lists GGUF file sizes:

- Search cards use the smallest non-projector GGUF.
- Inspecting a repository shows a separate estimate beside each model weight file before downloading.
- Projector files are excluded, and missing metadata is shown as “Fit unavailable”.
- Estimates use the current context-size and GPU-layer settings; they are a rough guide, not a guarantee.

### Local models

**Models** lists the GGUF files on this machine; **Hub** is where you find and download more. Set an absolute **Model download directory** in **Models** to choose where new downloads go. Changing it does not move existing files. The library scans the current download directory and the original app-data models directory.

Download progress reports file reconstruction and network bytes separately; Xet compression and deduplication can make those values differ.

### Questions and presets

Questions use the `choice`, `noul`, and `score` types. The JSON editors provide syntax highlighting and inline lint diagnostics. The state editor accepts text or JSON; switching modes converts the representation while preserving the content.

Enter a name beneath the questions editor and choose **Save custom** to save the questions, state, and state format across restarts. Select a saved preset to update it, or use its trash button to delete it. Built-in presets are starting points; calibrate decision thresholds and criteria against your own data.

## Local API

### Ports and access

While the app is open, the gateway prefers `http://127.0.0.1:11435` by default. If that port is occupied, Noullama binds a nearby free port, skips the private runtime port, and saves the replacement as its preferred port for future launches. The actual endpoint appears in **Local API** and a one-time notice when it changes. Copy that endpoint into your client; the examples below assume the default port.

The managed llama-server uses port `11436`. Both bind to localhost; changing the preferred gateway port takes effect after restarting the app.

Local SDK and curl clients use the gateway without an API key. The gateway accepts requests addressed to `127.0.0.1` or `localhost` on its active port and rejects foreign browser origins. The private llama-server uses a temporary key generated for each model load; connect integrations to the gateway rather than the private port.

### Native System One request

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

### OpenAI-compatible adapter

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

## Current limits

The current prototype supports one loaded model, one model download, and one runtime installation at a time. It has no download cancellation UI, bundled runtime, or automatic playground history. Named custom presets persist.
