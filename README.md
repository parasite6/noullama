<p align="center">
  <img src="logo33%25.png" alt="noullama logo">
</p>

<h1 align="center">Noullama</h1>

<p align="center">
  <a href="https://noullama.dev">noullama.dev</a>
</p>

<p align="center">A local workbench and API for native decision models using llama.cpp</p>

[Quick start](#quick-start) · [Cheat sheet](#cheat-sheet) · [Features](#features) · [Requirements](#requirements) · [Getting started](#getting-started) · [Runtime](#runtime) · [Models and presets](#models-and-presets) · [Images](#images) · [Local API](#local-api) · [Data and verification](#data-and-verification)

## Quick start

```sh
git clone https://github.com/parasite6/noullama.git
cd noullama
npm install
npm run desktop
```

In the app: **Hub** → download a GGUF (or **Models** → browse for a local file) → **Runtime** → choose a llama.cpp build → **Load model** → **Playground**.

## Cheat sheet

| I want to… | Do this |
| --- | --- |
| Download a model | Open **Hub**, search or type `organization/repository`, pick a repository, and download a GGUF file. |
| Use an existing model | Open **Models**, choose **Browse local model**, then click **Load model**. |
| Run a decision | In **Playground**, pick a preset under **Run settings**, edit the questions and state, then click **Run decision**. |
| Ask about images | Turn on **Vision** in the model bar, load the model, then add images in the **State** box. |
| Save a preset | Enter a name under the questions editor and click **Save preset**. |
| See why a load failed | Read the red panel under the model bar, or open **Runtime logs**. |
| Use CPU inference | Set **GPU layers** to `0` in **Runtime**. |
| Connect an application | Use `http://127.0.0.1:11435/v1`; native decisions are at `/systemone`. |

Question types:

- **`choice`** selects an option.
- **`noul`** answers yes/no.
- **`score`** rates an ordered set of levels.

## Features

- Discover and download decision-model GGUFs from Hugging Face using an existing login or a session token.
- Load a local model with a compatible llama.cpp runtime and configure GPU offloading. A **Vision** switch loads the model's multimodal projector.
- Run typed questions against text or JSON state, and against images when a projector is loaded, using editable presets, JSON linting, and syntax highlighting.
- See why a model failed to load, with llama.cpp's own error lines, without opening the logs.
- Hover the runtime card in the sidebar for the loaded model's settings and live graphics-memory use.
- Save, update, and delete named custom presets.
- Connect through a local System One endpoint or an OpenAI-compatible chat adapter.

## Requirements

Needed to build from source. The `.deb` and `.rpm` packages install the system libraries themselves, but still need `hf` and a `llama-server`.

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

1. Open **Hub**, search for a decision model or type a repository such as `LiquidAI/d1-2B-GGUF`, pick it, and download a GGUF. Already have a file? Open **Models** and browse for it instead.
2. In **Runtime → llama.cpp**, choose an installed build, download one, or enter a custom `llama-server` path and click **Check** to list its devices. Under **Model**, set GPU layers (`0` for CPU inference) and, for image-capable models, turn on the multimodal projector.
3. Click **Load model** in the model bar. The bar shows the loaded model and device and has controls to switch or unload it. If loading fails, a red panel under it gives the reason.
4. In **Playground**, choose a preset under **Run settings**, edit the questions and state, and run a decision.

## Runtime

### Finding llama.cpp

The runtime path starts unset. Noullama searches managed installs, nearby builds, and `PATH`. In **Runtime → llama.cpp**, select a discovered build under **Installed builds** or enter one under **Custom path**.

For GPU builds, set **Shared-library folders** if the runtime cannot find its libraries. On Linux, noullama also checks common CUDA 12 and 13 locations and passes the library paths to the runtime process.

The runtime card at the bottom of the sidebar shows whether a model is ready and which llama.cpp build is selected. While a model is running, it shows the build in use and marks any different selection as not applied. Hover or focus it for the loaded model's device, projector, context size, GPU layers, API address, and graphics memory: the runtime's own use (NVIDIA) and the card's total use (NVIDIA and AMD), read only while the card is open.

### When loading fails

Loading runs in the background. If it fails, or the runtime stops on its own after loading, a red panel under the model bar gives the reason together with the last error lines llama.cpp printed during that load. **View logs** opens the full runtime output.

### Downloading llama.cpp builds

**Runtime → llama.cpp** scans the configured location, nearby builds, PATH, and managed installs before offering a download.

1. Under **Download a build**, choose a backend and **Check official releases** to preview OS/architecture-matched assets from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases). Checking fetches metadata only.
2. **Download nightly + fallback** explicitly approves the displayed downloads: the latest nightly and the binary build linked by the latest official release. The fallback uses CPU, or Metal on macOS.
3. Select **Use** beside an installed runtime, then check or load it normally.

Archives are checked against GitHub's SHA-256 digests and installed under the app-data `runtimes/` directory; existing custom paths are preserved. Linux uses the official Ubuntu-built archives; unsupported platforms can use a custom build.

### Tray and lifecycle

On desktops with an active system tray, closing the window keeps the model and local API running. Use **Open Noullama** in the tray menu to restore the window.

**Quit Noullama**, available in the tray menu and sidebar, opens a centered confirmation. Confirming shuts down the API, owned runtime, and active downloads and deletes partial files from cancelled downloads before exiting; **Keep running** or Escape cancels.

Linux tray mode needs an AppIndicator library and a desktop tray host. GNOME setups without tray support may need an AppIndicator extension. If the library or host is unavailable, closing the window asks for confirmation before quitting.

Noullama releases its own ports on full exit; a port occupied by another service remains that service's responsibility.

On Linux, relaunching the same executable with the same app-data directory shuts down its older instance before starting the API. Startup also cleans up orphaned llama-server processes carrying Noullama's runtime identity and app-data key-file markers. Processes belonging to other applications or app-data directories are left running.

## Models and presets

### Hub and downloads

Hugging Face discovery uses decision-model tags and architecture metadata. Downloads use `hf download` at the inspected repository revision. noullama supports the standard Hugging Face token locations and `HF_TOKEN`; a token entered in the app is held for the session and passed to `hf` through its environment.

Author logos are fetched from each author's public Hugging Face profile the first time they appear in the Hub or a downloaded model, then cached for the session. Images are filtered at the display's pixel density for sharper small icons; authors without a profile image show their initials. No author artwork is bundled with the app.

Search results appear on the left and the picked repository's files on the right. Typing `organization/repository` offers to open that repository directly.

Hub search results show a local memory-fit estimate when a repository lists GGUF file sizes:

- Search cards use the smallest non-projector GGUF.
- Inspecting a repository shows a separate estimate beside each model weight file before downloading.
- Projector files are excluded, and missing metadata is shown as “Fit unavailable”.
- Estimates use the current context-size and GPU-layer settings; they are a rough guide, not a guarantee.

### Local models

**Models** lists the GGUF models on this machine; **Hub** is where you find and download more. Multimodal projector (`mmproj`) files are not listed as models: a model with a projector in its folder is marked **Vision available**. Set an absolute **Model download directory** in **Models** to choose where new downloads go. Changing it does not move existing files. The library scans the current download directory and the original app-data models directory.

The download panel shows one progress bar filled from both ends: network bytes received fill from the left and file reconstruction fills from the right, meeting in the middle when the file is done. Xet compression and deduplication can make the two values differ. Use **Cancel download** to stop; partial files are kept so an immediate retry resumes, and are deleted when you quit Noullama. While you are elsewhere in the app, a **Downloads** card in the bottom-right corner shows progress and can cancel the download or open its repository.

The **Models** list rescans the download directories every few seconds, so files added or removed outside the app show up without a refresh.

### Questions and presets

Questions use the `choice`, `noul`, and `score` types. The JSON editors provide syntax highlighting and inline lint diagnostics. The state editor accepts text or JSON; switching modes converts the representation while preserving the content.

Presets are in the **Run settings** panel beside the state. Enter a name under the questions editor and choose **Save preset** to save the questions, state, and state format across restarts. Select a saved preset to update it, or use the × on it to delete it. Built-in presets are starting points; calibrate decision thresholds and criteria against your own data.

## Images

Image input needs a model that supports it and its multimodal projector.

1. Put the projector `.gguf` (its name contains `mmproj`) in the same folder as the model. The Hub lists projectors next to the weights.
2. Turn on **Vision** in the model bar, or the projector switch in **Runtime → Model**, and load the model. Selecting a model from another folder turns the projector off.
3. In **Playground**, add images with **Add images**, by dropping files on the **State** box, or by pasting. The state text can be left empty to ask about the images alone.

Images are sent as base64 data URLs in the request's `images` field. PNG, JPEG, GIF, and BMP are sent unchanged; other formats the app can display, such as WebP, are converted to PNG first. Each image can be up to 20 MB and all images together up to 60 MB.

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

The adapter does not support chat history, tools, streaming, or arbitrary chat prompts. Multi-select, extraction, estimates, boxes, and video are not supported by this integration. Native clients can send images to `/v1/systemone` as an `images` array of `data:image/...;base64,...` URLs when the model is loaded with its projector. Requests through the gateway are limited to 16 MB.

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

Noullama is in early beta. Known limits:

- One loaded model, one model download, and one runtime installation at a time.
- No bundled runtime; use **Runtime → llama.cpp** to download one or point to your own `llama-server`.
- No automatic playground history. Only named custom presets are saved.
- Linux packages only (64-bit `.deb` and `.rpm`). There are no macOS or Windows builds yet.

Found a bug or missing something? [Open an issue](https://github.com/parasite6/noullama/issues/new?template=bug_report.md) or [request a feature](https://github.com/parasite6/noullama/issues/new?template=feature_request.md).
