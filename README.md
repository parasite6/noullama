<p align="center">
  <img src="logo33%25.png" alt="noullama logo">
</p>

<h1 align="center">Noullama</h1>

<p align="center"><a href="https://noullama.dev">noullama.dev</a></p>
<p align="center">A local workbench and API for decision models using llama.cpp</p>

[Quick start](#quick-start) · [What's new in v0.4.1](#whats-new-in-v041) · [Cheat sheet](#cheat-sheet) · [Requirements](#requirements) · [Runtime](#runtime) · [Memory protection](#memory-protection) · [Models and Library](#models-and-library) · [Playground](#playground) · [Images](#images) · [Multi-token prediction](#multi-token-prediction) · [Winnow models](#winnow-models) · [Local API](#local-api) · [Updates](#updates) · [Data and verification](#data-and-verification) · [Current limits](#current-limits)

## Quick start

Install the Linux x86-64 package for your distribution from the [v0.4.1 release](https://github.com/parasite6/noullama/releases/tag/v0.4.1):


## Debian / Ubuntu
```sh
curl -fLO https://github.com/parasite6/noullama/releases/download/v0.4.1/noullama_0.4.1_amd64.deb
sudo apt install ./noullama_0.4.1_amd64.deb
```
## Fedora / RPM-based distributions
```sh
curl -fLO https://github.com/parasite6/noullama/releases/download/v0.4.1/noullama-0.4.1-1.x86_64.rpm
sudo dnf install ./noullama-0.4.1-1.x86_64.rpm
```

These URLs are pinned to v0.4.1. Installed packages can check for later releases through **Updates**.

In the app: **Hub** → download a GGUF, or **Models → Browse file** → **Runtime** → choose or download a compatible llama.cpp build → **Load model** → **Playground**.

For development from source, install the [requirements](#requirements), then:

```sh
git clone https://github.com/parasite6/noullama.git
cd noullama
npm ci
npm run desktop
```

## What's new in v0.4.1

- **Several models at once**, each with its own runtime process, settings, private port, context history, API names and aliases. Eject one without stopping the others, including from the sidebar's loaded-model list.
- **Library and models.yml** for model names, aliases, Hugging Face accounts, file paths, MTP heads, projectors and sha256 values. Edit in the app or your text editor, with backups and protection against overwriting newer changes.
- **OOM prevention** with conservative load checks and automatic recovery from sustained critical memory pressure on Linux. **Allow force fit** disables this protection. A separate crowded-load prompt offers to eject another model and can be hidden without disabling protection.
- **Steadier interactions**: actions no longer dim unrelated controls across the app, unchanged polling results preserve displayed state, and MTP controls stay visible during rescans. Notifications float over the page rather than shifting its contents.
- **Concurrent downloads**, per-file cancellation, named model routing, and shared **Keep context** controls for the Playground and API.
- **Runtime download improvements**: remember the chosen backend, select the nightly after installation, and keep the official stable build as a fallback for that backend when available.
- App-drawn Library suggestions, copy controls, clearer load errors and confirmation before deleting a preset.

## Cheat sheet

| I want to… | Do this |
| --- | --- |
| Download a model | **Hub** → search or enter `organization/repository` → pick a GGUF. |
| Use an existing model | **Models → Browse file**, or **Library → Add model**. |
| Run a decision | **Playground → Run settings** → choose a loaded model and preset, edit state/questions, then **Run decision**. |
| Run several models | Select another model and choose **Load beside**. |
| Unload one model | Use its **Eject** button in **Models → Loaded** or the sidebar. |
| Configure API names | Edit the name or aliases in **Library**; aliases are also editable under **Models** and **Local API → Model routing**. |
| Use CPU inference | **Runtime → GPU layers** → `0`. |
| Ask about images | Configure the projector in **Library** or **Runtime**, enable **Vision**, then load the model and add images to State. |
| Enable MTP | Choose a model with built-in MTP or a matching head, turn on **MTP**, then load or reload it. |
| Keep earlier decisions | Enable **Keep context** in **Run settings** or **Local API**. |
| Override memory protection | **Settings → Allow force fit**. |
| Restore the crowded-load prompt | **Settings → Show crowded-load warning again**. |
| Choose model-file checks | **Settings → sha256 before loading**. |
| Diagnose a failed load | Read its error panel or open **Runtime logs**. |
| Connect a client | Copy the actual endpoint from **Local API**; the default base URL is `http://127.0.0.1:11435/v1`. |

## Requirements

Packaged installs need the runtime libraries supplied through their package dependencies. Building from source also needs:

- **Node.js 24+**, npm and a stable Rust toolchain with Cargo.
- On Linux, GTK 3, WebKitGTK 4.1 and the other [Tauri build prerequisites](https://v2.tauri.app/start/prerequisites/#linux).

For model inference and downloads:

- A compatible **llama-server**. Download builds under **Runtime → llama.cpp**, or use your own. Native decision models need a runtime that supports their architecture and `/v1/systemone`; MTP and vision also require compatible runtime support. A memory-fit badge does not establish model/runtime compatibility.
- The **Hugging Face CLI (`hf`)** on PATH, or configured under Runtime, for Hub downloads. Public ungated models do not need a login; gated models need access and authentication. Local GGUFs can be used without downloading them through the Hub.

To build release packages and a standalone executable:

```sh
npm run desktop:build
# Debug build:
npm run desktop:build -- --debug
```

Linux executables are written to `src-tauri/target/release/noullama` or `src-tauri/target/debug/noullama`. They include the frontend and do not need a Vite server. `npm run desktop` uses the separate `src-tauri/target/dev` directory. `npm run dev` alone is an interface preview without the Rust backend.

## Runtime

### Finding and downloading llama.cpp

The runtime path starts unset. **Runtime → llama.cpp** discovers managed installs, the configured location, nearby builds and PATH. Choose **Use** beside an installed build or enter a folder/executable under **Custom path**. **Check** runs device detection.

**Check official releases** previews the latest nightly and official stable release assets for the selected OS, architecture and backend. It reads metadata only. **Download nightly + fallback** downloads and verifies those archives, installs them under the app-data `runtimes/` directory, and selects the nightly for your next load. The stable build stays available to select manually. It uses the chosen backend when available, otherwise CPU, or Metal on macOS. Versioned upstream releases that link a separate binary build use that linked build. The chosen download backend is remembered.

Archives are verified against GitHub's SHA-256 digests. Linux downloads use the official Ubuntu-built archives; a custom build may be needed for your system or model.

For GPU builds, configure **Shared-library folders** if libraries cannot be found. On Linux, Noullama also checks common CUDA 12 and 13 locations and passes these paths to the child process. Set **GPU layers** to `0` for CPU inference.

### Loading and observing models

Each model loads in the background in its own `llama-server`. Select another and choose **Load beside** to keep both running, subject to available memory. Different models can use different runtime builds. A file can be loaded only once; **Reload model** ejects that file and loads it again with the current settings.

Changes to runtime path, context, GPU layers, projector or MTP apply to new loads. Already loaded models keep their original settings. **Playground → Run settings → Model** selects which loaded model answers a decision.

The sidebar runtime card shows readiness and the selected build; a different selection from the running build is marked as not applied. Hover or focus the card for model settings and live graphics-memory use. NVIDIA provides per-process usage; NVIDIA and AMD provide card-wide usage where available. These readings are fetched only while the card is open. Each sidebar loaded-model row has its own **Eject**.

A failed load or a runtime that exits after becoming ready produces an error panel with its recent llama.cpp error lines. **View logs** opens the runtime output. Each model's log lines carry its own label.

### Tray and shutdown

With an active system tray, closing the window keeps loaded models, downloads and the local API running. **Open Noullama** restores the window. **Quit Noullama** in the tray or sidebar asks for confirmation; **Keep running** or Escape cancels. Confirming stops the gateway, all owned runtimes and active downloads, and removes cancelled-download partial files.

Linux tray mode needs an AppIndicator library and a tray host. GNOME may need an AppIndicator extension. Without tray support, closing the window asks for confirmation before quitting.

On Linux, relaunching the same executable with the same app-data directory replaces its older instance. Startup also cleans up orphaned runtimes with Noullama's identity and app-data key-file markers. Other applications' processes and ports are left alone.

## Memory protection

Protection is enabled by default. Before a load is accepted, and again before its runtime starts, Noullama checks for clear oversubscription using estimated model, companion-file, context and buffer requirements against available memory. Full GPU-offload checks use live VRAM readings when available; CPU-load checks use available RAM on Linux. Estimates include a substantial uncertainty margin: borderline estimates and uncertain partial-offload allocations are not treated as definite OOMs.

If another model is loaded and the new load clearly exceeds the available budget, **“This probably won't fit”** offers **Cancel** or **Eject [model name]**. Choosing an eject target stops that model and rechecks before trying the load. With several other models, each is offered separately; ejecting one is not a promise the new load will fit.

**Don't show this again** hides only this crowded-load message. It does **not** enable force fit and does **not** bypass load checks. Restore it with **Settings → Show crowded-load warning again**.

### Automatic recovery on Linux

High utilization alone never ejects a model: **7,000 MB used out of 8,000 MB is not an OOM condition**. Recovery requires all of the following:

- Available RAM below the smaller of **1% of the memory budget** and **256 MiB**.
- Full memory-pressure stalls of at least **5% over the recent 10-second window**.
- At least **three consecutive one-second samples**, followed by a recheck before intervention.

Linux cgroup v2 memory limits and pressure are included where readable. Missing telemetry never triggers ejection. Recovery selects an owned runtime using more than 128 MiB of resident RAM, preferring the largest measured user, and unloads one model. It records an explanation, waits five seconds, then requires fresh pressure evidence before another intervention. VRAM utilization by itself never triggers automatic recovery.

**Settings → Allow force fit**, off by default, disables both pre-load prevention and automatic recovery immediately and is saved across restarts. It is independent of the prompt's suppression preference. The operating system or llama.cpp can still reject or stop a model that runs out of memory; protection cannot guarantee every allocation will succeed. Automatic pressure recovery currently requires Linux telemetry.

## Models and Library

### Hub and downloads

**Hub** discovers decision-model repositories using tags and architecture metadata. Search results show repositories on the left and the selected repository's GGUFs on the right. Entering `organization/repository` offers to open it directly.

Downloads use `hf download` at the inspected repository revision. Saved Hugging Face credentials and `HF_TOKEN` are supported; a token entered in the app is held only for the session and passed to `hf` through its environment.

Several files can download concurrently. Each has a progress panel and **Cancel download**; network progress fills from one side and file reconstruction from the other. Xet compression and deduplication can make those values differ. Cancelled partial files remain available for an immediate resume and are removed on full quit. The floating **Downloads** card provides cancellation, completion dismissal and repository navigation while you use other tabs.

Memory-fit badges use file sizes and the selected context/GPU-layer settings. Search cards estimate the smallest model GGUF; repository inspection estimates each weight file separately. Projectors and MTP heads are excluded as standalone models; missing metadata shows **Fit unavailable**. The selected local model's fit also accounts for estimated memory held by other loaded models. These are estimates, not compatibility or allocation guarantees.

Author avatars come from public Hugging Face profiles and are cached for the session; missing images fall back to initials. No author artwork is bundled.

### Local files and download directory

**Models** lists local GGUFs. Projectors and MTP heads are companions, not standalone models. **Vision available**, **MTP available** and **Winnow layout** identify detected capabilities.

Set an absolute **Model download directory** under Models to choose where new downloads go. This does not move existing files. Scans include that directory and the original app-data models directory; the Models tab rescans every few seconds. Use **Browse file** or **Library → Add model** for files elsewhere.

### Library and models.yml

The app-data `models.yml` holds one entry per model: its API **name**, **aliases**, file **path**, Hugging Face **account**, **sha256**, and optional **MTP head** and **projector**, each with its own sha256. Discovered models are registered automatically. Companion files do not get separate model entries.

```yaml
version: 1
models:
- name: d1-3B-Q8_0
  aliases:
  - triage
  path: /home/u/models/LiquidAI--d1-3B-GGUF/d1-3B-Q8_0.gguf
  account: LiquidAI
  sha256: ''
  mtp: /home/u/models/LiquidAI--d1-3B-GGUF/mtp-d1-3B.gguf
  mtp_sha256: ''
  mmproj: ''
  mmproj_sha256: ''
```

- **Library** edits names, aliases, paths, accounts, companions and hashes. Account/path suggestions are drawn inside the app. Changes stay in a draft until **Save models.yml**.
- Empty `mtp` or `mmproj` fields allow folder discovery. New entries record detected companions, and later discoveries can fill an empty field. A listed missing companion stops a load that uses it. MTP heads must still match the model.
- **Edit models.yml** opens your default editor. Noullama rereads outside changes during snapshots and relevant model/API operations. Renaming the API name never renames the file; moving a file requires updating its path.
- Backups go to `backups/models`; the newest 20 are kept. App edits are backed up except automatic hash recording, and outside changes preserve the previous valid version. **Backups** opens the folder.
- Invalid YAML, duplicate names and relative paths are rejected without overwriting the file. The last valid library remains active until corrected. Saving a draft based on an older on-disk revision is refused; discard it and edit the newer version.

### Model-file verification

**Settings → sha256 before loading** controls checks for the model and the listed companions used by a load:

| Mode | Behavior |
| --- | --- |
| **Hash once** (default) | Hash missing values in the background; before loading, hash again if size or modification time changed. A mismatch stops the load. |
| **Every load** | Read and hash the complete file before each load. |
| **Warn only** | Use the Hash once policy, but log mismatches instead of blocking. |
| **Off** | Keep recorded hashes without computing or checking them. |

Library shows missing files, hashing progress and **sha256 changed**. If you intentionally replaced a file, **Accept new sha256** records its new value. Hash writes do not overwrite a newer companion selection. Size/modification-time caching uses `sha256-cache.json`. An unrecorded folder-discovered companion has no saved hash to compare. These checks help detect replaced, truncated or incorrect files; they are not a security boundary.

## Playground

Choose a loaded model under **Run settings → Model**, then edit State and Questions. Supported question types are:

- **`choice`**: select an option.
- **`noul`**: answer yes/no with a probability.
- **`score`**: evaluate an ordered set of levels.

State accepts text or JSON. Switching representations preserves the content; JSON editors provide syntax highlighting and inline diagnostics. Named custom presets save questions, state and state format across restarts. Select one to update it, or use its × button to delete it after confirmation. Built-in presets are starting points; calibrate criteria and thresholds against your own data.

Actions show pending state on the relevant controls. Background polling preserves unchanged state and unsaved edits. Notifications appear over the page rather than moving the content.

### Keep context

Enable **Keep context** in Run settings or **Local API** to include earlier decision states and short answers. A text state gets a preceding text block; a JSON state becomes `{"earlier_decisions": [...], "current_state": ...}`. The shared switch applies to both entry points, but each loaded model has its own history.

If a request exceeds its context window, the oldest decisions are flushed one at a time until it fits. Up to 64 are kept, and images are not carried forward. Turning the switch off clears all histories. **Clear** in Playground clears the selected model's history; in Local API it clears every model's. Ejecting or reloading clears that model's history. Histories do not survive restarts.

## Images

Image input requires a model that supports images and its matching multimodal projector. Set `mmproj` in Library, choose a projector under **Runtime → Model**, or put a file containing `mmproj` in its name beside the model for discovery.

Enable **Vision**, then load or reload the model. Add images with **Add images**, drag and drop onto State, or paste them. The text state can be empty. Switching model folders can clear the previous projector selection; verify the selection before loading.

Images are sent as base64 data URLs in `images`. PNG, JPEG, GIF and BMP are sent unchanged; other browser-readable formats, such as WebP, are converted to PNG. The app allows 20 MB per image and 60 MB combined. The gateway separately caps the entire JSON request at **16 MiB**, including base64 overhead, so requests through it may need smaller images.

## Multi-token prediction

MTP can speed up models that predict several tokens per step. Noullama detects built-in MTP layers in GGUF metadata or a separate head whose metadata identifies it as MTP-only and whose width matches the model. A head can be listed in Library, chosen under Runtime or discovered in the model's folder. It is not a standalone model. Before download, the Hub labels possible heads by filename.

Enable **MTP** in the model bar or Runtime, then load/reload. Selecting a different model turns it off. The runtime receives `--spec-type draft-mtp` and, for a separate head, `--spec-draft-model` with the same GPU-layer setting. A compatible llama.cpp build is required; unsupported models or mismatched heads fail with an explanation. Decision outputs are short, so the benefit depends on the model and request.

## Winnow models

Native decision models advertise decision metadata that a compatible llama.cpp serves at `/v1/systemone`. Winnow GGUFs use a separate path: Noullama implements the `winnow-v1` layout, builds one prompt per question, reads the next-token probabilities for letter-labelled options, and converts those into the usual decision response.

Automatic detection recognizes Gemma 4 models whose name or filename contains `winnow`. For a differently named Winnow model, choose **Runtime → Model → Decision mode → Winnow**. **Native** requires native decision support; **Auto** chooses from metadata. The layout follows [Ollaya](https://github.com/ollaya-dev/ollaya), including Winnow-E4B's fitted temperature of 1.2574; requests can override it with `"winnow": {"temperature": ...}`. Each question uses a fresh evaluation rather than reusing the preceding prompt cache.

The Playground and gateway expose the same response shape for either path. Images are not supported by the Winnow layout.

## Local API

### Ports and access

The gateway prefers `http://127.0.0.1:11435`. If occupied, it finds a nearby free port, skips the configured private runtime port, saves the replacement and shows the actual endpoint in **Local API**. Copy that endpoint into your client; the examples below assume the default.

Every model has its own private localhost port: the first available port in the 64-port range starting at **Private llama-server port**, default `11436`, excluding occupied ports and the active gateway. Gateway port changes take effect after restart.

Local SDK/curl clients need no gateway API key. Requests must address `127.0.0.1` or `localhost` on the active port; foreign browser origins are rejected. Private runtimes use temporary per-load keys. Integrations should use the gateway.

### Model names and routing

Every decision request must include a case-sensitive `model`: the name in models.yml, any alias, or the filename stem for a loaded file without a library entry. Rename names in Library and edit aliases there, in Models or in **Local API → Model routing**. Names/aliases allow letters, digits and `. _ : / -`, up to 64 characters, with up to 16 aliases per model. Library names must be unique ignoring case.

| Condition | Response |
| --- | --- |
| Missing `model` | **400**, with accepted names. |
| Unknown name | **404**; case-only differences suggest the known spelling. |
| Known model unloaded or still loading | **503**. |
| Ambiguous name shared by loaded files | **409**. |

`GET /v1/models` lists names reaching ready models. Responses echo the requested name or alias. Each model serializes its own decisions; different models have independent runtime processes. The Playground selects a loaded instance directly.

### System One request

```sh
curl http://127.0.0.1:11435/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "d1-3B-Q8_0",
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

`POST /v1/chat/completions` accepts one user message containing a JSON object with `state` and `questions`. The assistant message contains the decision response as JSON. This adapter does not provide arbitrary chat, message history, tools or streaming. `GET /health` reports gateway/model readiness.

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="local")
response = client.chat.completions.create(
    model="d1-3B-Q8_0",  # exact name or alias from Library
    messages=[{"role": "user", "content": json.dumps({
        "state": "I was charged twice. Please refund the duplicate payment.",
        "questions": {
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}
        },
    })}],
    stream=False,
)
print(json.loads(response.choices[0].message.content)["answers"])
```

Native clients can supply an `images` array of `data:image/...;base64,...` URLs to `/v1/systemone` when the selected model has a projector. Gateway request bodies are limited to 16 MiB. Multi-select, extraction, estimates, boxes and video are not supported by this integration.

## Updates

**Updates** in the sidebar shows the installed version and checks GitHub. A newer eligible release changes the entry to **Update to [version]**.

- **Install and restart** downloads the matching `.deb` or `.rpm`, verifies its signature, asks for system installation authorization, then stops owned runtimes and the local API and restarts after installation.
- **Check when noullama starts** checks a few seconds after launch. It does not automatically download an update.
- **Include pre-releases** also offers prereleases; prerelease builds enable this preference by default.

Only package-installed copies can update themselves. Source/standalone builds show available updates but must be updated manually. Versions before 0.2.1 need a one-time package installation of a newer version to gain the updater.

## Data and verification

Settings, models.yml, backups, hash caches, saved presets, managed runtimes and default downloads live under the platform app-data directory: typically `~/.local/share/dev.noullama.desktop` on Linux. External GGUFs remain in place; a custom download directory affects new downloads only.

```sh
npm test
npm run build
cargo test --manifest-path src-tauri/Cargo.toml -- --test-threads=1
```

The Rust tests include localhost mock servers; serial execution avoids test-port conflicts. Real-model tests are opt-in. Set absolute `NOULLAMA_TEST_MODEL` and `NOULLAMA_TEST_RUNTIME` paths, and optionally `NOULLAMA_TEST_GPU_LAYERS=99`, then:

```sh
cargo test --manifest-path src-tauri/Cargo.toml \
  real_runtime_native_and_openai_requests -- --ignored --nocapture
```

`real_runtime_flushes_oldest_decisions_at_the_context_limit` checks Keep context on CPU with a 512-token context. `real_runtime_two_models_route_by_alias_and_eject_one` also needs a different `NOULLAMA_TEST_MODEL_2`; `NOULLAMA_TEST_GPU_LAYERS_2` configures it. It loads both, routes by alias, ejects one and checks the other still answers.

## Current limits

Noullama is in early beta:

- Published packages target **Linux x86-64** (`.deb` and `.rpm`); no published macOS or Windows app builds yet.
- No bundled llama.cpp runtime. Download one in Runtime or provide a compatible build.
- Model/runtime compatibility is independent of memory fit. Unrecognized non-native models are supported only through the Winnow layout, without images.
- Memory estimates are approximate; automatic pressure recovery currently uses Linux telemetry and cannot prevent every OOM.
- Each loaded model handles one decision request at a time. Running several models is limited by your hardware and the available private-port range.
- Only named presets persist Playground input. Keep context is temporary, per model, and does not preserve images.
- The OpenAI-compatible endpoint is a decision adapter, not a general chat API.

[Report a bug](https://github.com/parasite6/noullama/issues/new?template=bug_report.md) or [request a feature](https://github.com/parasite6/noullama/issues/new?template=feature_request.md).
