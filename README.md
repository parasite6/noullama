<p align="center">
  <img src="logo33%25.png" alt="noullama logo">
</p>

<h1 align="center">Noullama</h1>

<p align="center"><a href="https://noullama.dev">noullama.dev</a></p>
<p align="center">A local workbench and API for decision models using llama.cpp</p>

Decision models answer typed questions about a piece of content, such as "which team should handle this?" or "is this a refund request?", instead of chatting. Noullama runs them on your machine:

- downloads models from Hugging Face and llama.cpp builds from GitHub
- loads one or more models, on a GPU, split across GPUs, or on the CPU
- lets you try questions in a **Playground**
- serves them to your own code through a **local API** and a **Python client**
- protects your machine from loads that won't fit in memory

Linux x86-64 only, in early beta.

## Contents

- [Install](#install)
- [Getting started](#getting-started)
- [Using the app](#using-the-app)
  - [Playground](#playground)
  - [Models and Library](#models-and-library)
  - [Hub](#hub)
  - [Loading](#loading)
  - [Runtime](#runtime)
  - [Images and MTP](#images-and-mtp)
  - [Decision modes](#decision-modes)
- [Local API](#local-api)
- [Python client](#python-client)
- [Safety checks](#safety-checks)
  - [Memory protection](#memory-protection)
  - [Model-file checks](#model-file-checks)
- [Running in the background](#running-in-the-background)
- [Updates](#updates)
- [Where files live](#where-files-live)
- [Building from source](#building-from-source)
- [Limits](#limits)
- [License](#license)

## Install

```sh
curl -fsSL https://noullama.dev/install.sh | sh
```

The script:

- picks the `.deb` or `.rpm` for your package manager (apt, dnf or zypper)
- downloads it from the [latest release](https://github.com/Noullama/noullama/releases/latest)
- refuses to install unless it matches the SHA-256 GitHub publishes
- installs it with `sudo`, or tells you when you're already up to date

To install a package by hand, see the [download page](https://noullama.dev/download/).

## Getting started

1. **Get llama.cpp.** In **Runtime**, choose a backend, press **Check official releases**, then **Download nightly + fallback**. Press **Save settings**.
2. **Get a model.** In **Hub**, search, press **Download** on a GGUF, then **Select model**. For a file you already have, use **Models → Browse local model** instead.
3. **Load it.** In **Loading**, press **Load model**.
4. **Ask it.** In **Playground**, edit the state and questions, then press **Run decision**.

## Using the app

### Playground

Try questions against a loaded model.

- **State** is the content to evaluate, as plain text or JSON.
- **Questions** is a JSON object of named questions. There are three types:
  - `choice`: pick one option
  - `noul`: yes or no, with a probability
  - `score`: pick a level on an ordered scale
- **Run settings**, on the right:
  - **Model:** which loaded model answers
  - **Presets:** six built-in starting points, plus your own. Saved presets keep the questions and state across restarts.
  - **Keep context:** see below
- **Result** shows each answer with its confidence and probability bars, or the raw JSON, plus the time the run took.

#### Keep context

When on, a model's earlier decisions are added to each new request, so it can take them into account.

- Each loaded model has its own switch and history. You'll find it on the model's card in **Loading** and in the Playground's Run settings.
- The **Keep context** toggle in Loading only sets whether newly loaded models start with it on.
- It keeps up to 64 decisions. When a request no longer fits the context window, the oldest are dropped first.
- History lives in memory only. It's cleared when you eject or reload the model, or turn the switch off. Images are never kept.

### Models and Library

**Models** lists the GGUF files on your machine.

- Chips show what each file supports: **Vision available**, **MTP available**, **Prompt layout (Winnow)** or **Not a decision model**.
- Click a row to select it and open **Loading**.
- **Model download directory** sets where new downloads go. Existing files stay where they are.

**Library** edits `models.yml`, the list of every model Noullama knows:

| Field | What it's for |
| --- | --- |
| `name` | the name API requests use (exact case) |
| `aliases` | other names for the same model |
| `path` | the `.gguf` file |
| `account` | Hugging Face account, for the avatar |
| `sha256` | recorded hash of the file |
| `mtp`, `mmproj` | optional MTP head and image projector. Leave empty to look in the model's folder |

- Edits stay a draft until **Save models.yml**. **Edit models.yml** opens it in your text editor instead.
- Every change is backed up first, and the newest 20 backups are kept. **Backups** opens the folder.
- An invalid file is never overwritten. The last valid version stays in use until you fix it.

### Hub

Find and download decision models from Hugging Face.

- Search finds repositories tagged `decision`, `decision-model` or `system-one`. Type `organization/repository` to open one directly.
- Each GGUF has a **Download** button. Several files can download at once, and the floating **Downloads** card follows you to other tabs.
- Downloads use the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli). If it isn't on your PATH, set its location in **Runtime → Connection**.
- Public models need no login. For gated ones, Noullama uses your `hf auth login` session or `HF_TOKEN`, or a token you paste for this session only. A pasted token is never written to disk.
- **Download with** picks Auto, Xet or HTTPS.
- A cancelled download can't be resumed. Downloading it again starts over, and leftover partial files are removed when you quit.

### Loading

Load models and choose how each one runs. The **Load a model** panel has four sections:

- **Model:** the GGUF file, plus an optional **Multimodal projector** (for images) and **Multi-token prediction (MTP)**.
- **Hardware:**
  - **Device:** one GPU, **All GPUs**, **Choose GPUs…**, or **CPU**. The CPU needs a CPU runtime, set in Runtime.
  - **Split mode** and **Split ratio:** only when splitting across GPUs. Leave the ratio empty to split by free memory.
  - **GPU layers:** 99 puts every layer that fits on the GPU. 0 runs on the CPU.
  - **Context size:** maximum tokens per request, 8192 by default.
- **Behaviour:** **Decision mode** (see [Decision modes](#decision-modes)) and the default for **Keep context**.
- **Runtime:** which llama.cpp build is used. You change it in the Runtime tab.

The memory-fit badge at the top is an estimate. It doesn't tell you whether the model and runtime work together.

**Running several models:**

- Press **Load beside** to keep models already loaded, or **Load another instance** to load the same file again.
- Each instance is its own `llama-server` process with its own memory. Instances of one file can sit on different devices.
- Each one appears under **Loaded** with **Eject**, **Reload**, its API names and its Keep context switch.
- A loaded model keeps the settings it started with. Changes apply to the next load, or press **Reload**.

When a load fails, its error panel shows the recent llama.cpp errors. The **Runtime activity** log has the full output.

### Runtime

Choose the llama.cpp builds Noullama runs.

- **Installed builds:** press **Use** beside one. Noullama finds its own downloads, builds near the app and `llama-server` on your PATH.
- **Custom path:** a folder or `llama-server` executable. **Check** shows its version and devices.
- **Download a build:** choose a backend (CPU, Vulkan, CUDA 12 or CUDA 13). It installs the newest nightly plus the latest stable release as a fallback, each checked against GitHub's SHA-256.
- **CPU runtime:** a second build used only for loads on Device CPU, with its own download button.
- **Shared-library folders:** extra folders for CUDA or other libraries the runtime can't find. Common CUDA 12 and 13 locations are checked automatically.
- **Connection:** the local API port, the first private port for models, and the `hf` path.

Changes on this tab take effect after **Save settings**, and only for the next load.

Downloads are the official Ubuntu-built archives. Some systems or models may need a custom build.

#### Automatic llama.cpp updates

**Settings → llama.cpp updates → Update llama.cpp automatically** is on by default.

- It keeps the builds Noullama downloaded up to date: nightly builds move to the newest nightly, and stable builds to the newest stable.
- It checks at most once a day, starting 10 seconds after launch. A failed check is retried within the hour.
- A selected build switches to its channel's new build for your next load. An old build is removed once no loaded model uses it.
- Builds outside Noullama's `runtimes/` folder are never touched.
- **Check now** runs a check right away, even with the switch off.

### Images and MTP

**Images** need a model that supports them and its matching projector (`mmproj`).

1. Set the projector in Library, or put a file with `mmproj` in its name beside the model.
2. Turn on **Multimodal projector** in Loading, then load or reload.
3. In Playground, add images with **Add images**, drag and drop, or paste. The text state can be empty.

Limits are 20 MB per image and 60 MB in total. PNG, JPEG, GIF and BMP are sent as they are; other formats are converted to PNG.

**MTP** (multi-token prediction) can speed up models that draft several tokens per step. It uses either layers built into the model or a separate MTP head file in the model's folder.

1. Turn on **MTP** in Loading.
2. Load or reload the model.

It needs a llama.cpp build with MTP support. Decision answers are short, so the gain varies.

### Decision modes

**Decision mode** in Loading sets how answers are read:

| Mode | How it works |
| --- | --- |
| **Built-in** | The model and llama.cpp answer decisions themselves (`/v1/systemone`). Needs a decision GGUF and a compatible build. |
| **Prompt layout** | For Winnow models. Noullama builds the prompt and reads the probability of each answer label. Text only. |
| **Auto** (default) | Prompt layout for Gemma 4 models with `winnow` in the name, otherwise Built-in. |

Both modes return the same response shape. They count `score` and `confidence` differently, though: Built-in gives the most likely level, and Prompt layout a weighted average. A Winnow request can set its own temperature with `"winnow": {"temperature": 1.2}`.

## Local API

Your own code can use any loaded model over HTTP.

**Address:** `http://127.0.0.1:11435`.

- If that port is busy, Noullama picks a nearby one and shows it in the **Local API** tab.
- It accepts requests from this machine only.
- No API key is needed unless you turn on **Require API key** in Settings. Then send `Authorization: Bearer <key>`.

| Method | Path | Does |
| --- | --- | --- |
| `GET` | `/health` | whether the API is up and a model is ready |
| `GET` | `/v1/models` | the names you can send as `model` right now |
| `POST` | `/v1/systemone` | runs a decision |
| `POST` | `/v1/chat/completions` | the same, in OpenAI's chat format |

**Model names.** Every request names its model, case-sensitively. A model answers to:

- its Library name and aliases
- its file name without `.gguf`, when it has no Library entry
- for a second instance, the same names plus a suffix such as `-i-2`
- `any-available`, which goes to any free, ready model

| Problem | Status |
| --- | --- |
| No `model` given | 400, with the names you can use |
| Unknown name | 404, suggesting the right spelling when only the case is wrong |
| Same name on two loaded models | 409 |
| Model not loaded, still loading, or every model busy | 503 |

**Example request:**

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

- **Images:** add `"images": ["data:image/png;base64,..."]` for a model loaded with a projector.
- **Size limit:** a request can be at most 16 MiB.
- **One at a time:** each model handles one request at a time, and further requests wait their turn.

<details>
<summary>OpenAI-compatible adapter</summary>

`/v1/chat/completions` takes one user message whose content is the decision request as JSON. The reply's message holds the decision as JSON. It isn't a chat API: there's no history, tools or streaming.

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="local")
response = client.chat.completions.create(
    model="d1-3B-Q8_0",
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

</details>

## Python client

Ask a loaded model questions from Python. Python 3.10+, no dependencies.

```sh
python -m pip install "git+https://github.com/Noullama/noullama#subdirectory=python"
```

```python
from noullama import Client, Noul

nl = Client()
decision = nl.decide(
    model="any-available",
    state="Please refund my payment.",
    questions={"refund": Noul(instructions="Is this a refund request?")},
)

answer = decision["refund"]
print(answer.value)  # 0.97: the chance the answer is yes
```

The [Python client guide](python/README.md) covers question types, answers, API keys and errors.

## Safety checks

### Memory protection

On by default. Noullama stops loads that clearly won't fit in memory, and ejects a model if the system is about to run out.

- **Before a load:** it estimates what the model needs and compares that with the free memory on the chosen device. A clear overrun is refused, and borderline cases are let through.
- **When memory is tight:** if another model is loaded, the **"This probably won't fit"** dialog offers **Cancel** or **Eject** for each other model.
- **When the system is starving:** it ejects the largest model only when free RAM is critically low *and* the system is stalling, over several seconds. High use alone never ejects anything.
- **Allow force fit**, in Settings, turns all of this off. **Don't show this again** in the dialog hides only the dialog.

Protection can't catch everything. The operating system or llama.cpp may still stop a model that runs out of memory.

<details>
<summary>Exact rules</summary>

- **Estimate:** file size × 1.12 + context size × 24 KiB. The file size includes the projector and MTP head.
- **What gets checked:** full GPU offload (GPU layers 99 or −1) is checked against that GPU's free VRAM, and CPU loads against available RAM. Partial offload (1–98 layers) isn't checked.
- **"Clearly exceeds"** means the estimate is more than the available memory plus a margin: a fifth of the estimate, or 512 MiB if that's larger.
- **When it ejects:** available RAM is below 1% of total or 256 MiB, whichever is smaller, and memory-pressure stalls are at least 5% over 10 seconds. Both must hold for three samples in a row and on a final recheck. It then unloads the largest model using over 128 MiB of RAM, and waits 5 seconds before acting again.
- **Containers:** cgroup v2 limits are included where readable. Missing readings never trigger an ejection, and VRAM use never does.

</details>

### Model-file checks

**Settings → sha256 before loading** guards against loading a file that was replaced, truncated or swapped by mistake. It isn't a security boundary.

| Mode | What happens |
| --- | --- |
| **Hash once** (default) | Each file is hashed once, and again before a load if it changed. A mismatch stops the load. |
| **Every load** | The whole file is hashed before every load. Safest, but slower. |
| **Warn only** | Like Hash once, but a mismatch only logs a warning. |
| **Off** | No hashing or checks. |

If you replaced a file on purpose, press **Accept new sha256** in Library.

## Running in the background

- With a system tray, closing the window keeps models, downloads and the API running. **Open Noullama** in the tray brings it back.
- **Quit Noullama**, in the tray or sidebar, asks first. Quitting stops the API, every model and any downloads.
- On Linux the tray needs an AppIndicator library and a tray host. GNOME may need an AppIndicator extension. Without one, closing the window asks whether to quit.

## Updates

**Updates** in the sidebar shows your version and checks GitHub for a newer one.

- **Install and restart** downloads the new package, verifies its signature, asks for your password, then restarts.
- **Check when noullama starts** checks shortly after launch. It never installs anything on its own.
- **Include pre-releases** offers pre-release versions too.

Only copies installed from the `.deb` or `.rpm` can update themselves. Builds from source must be updated by hand.

## Where files live

Everything is under `~/.local/share/dev.noullama.desktop`:

| Path | Holds |
| --- | --- |
| `settings.json` | settings |
| `models.yml` | the model library |
| `backups/models/` | models.yml backups |
| `presets.json` | saved Playground presets |
| `runtimes/` | downloaded llama.cpp builds |
| `models/` | downloaded models, unless you set another download folder |

Model files added from elsewhere stay where they are.

## Building from source

You need:

- Node.js 24+, npm, and a stable Rust toolchain
- GTK 3, WebKitGTK 4.1 and the other [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/#linux)

```sh
git clone https://github.com/Noullama/noullama.git
cd noullama
npm ci
npm run desktop          # run in development
npm run desktop:build    # release build: src-tauri/target/release/noullama, plus .deb and .rpm
```

`npm run dev` alone opens an interface preview without the backend.

**Tests:**

```sh
npm test
cargo test --manifest-path src-tauri/Cargo.toml -- --test-threads=1
```

<details>
<summary>Tests with a real model</summary>

These are skipped by default. Set absolute paths in `NOULLAMA_TEST_MODEL` and `NOULLAMA_TEST_RUNTIME`, optionally `NOULLAMA_TEST_GPU_LAYERS=99`, then:

```sh
cargo test --manifest-path src-tauri/Cargo.toml \
  real_runtime_native_and_openai_requests -- --ignored --nocapture
```

`real_runtime_two_models_route_by_alias_and_eject_one` also needs a second model in `NOULLAMA_TEST_MODEL_2` (optionally `NOULLAMA_TEST_GPU_LAYERS_2`).

</details>

## Limits

- Linux x86-64 only (`.deb` and `.rpm`); no macOS or Windows builds.
- llama.cpp isn't bundled. Download it in Runtime or use your own build.
- A memory-fit badge is an estimate, not a promise the model will run.
- Each loaded model answers one request at a time.
- Question types are `choice`, `noul` and `score` only.
- The OpenAI-compatible endpoint is a decision adapter, not a chat API.
- Playground input is kept only in saved presets, and Keep context is lost on restart.

[Report a bug](https://github.com/Noullama/noullama/issues/new?template=bug_report.md) or [request a feature](https://github.com/Noullama/noullama/issues/new?template=feature_request.md).

## License

Noullama is licensed under the [GNU Affero General Public License v3.0 only](LICENSE) (`AGPL-3.0-only`). The Python client in `python/` is MIT-licensed. See also the [trademark policy](TRADEMARK.md). Copyright (C) 2026 parasite6.
