// The device each load runs on: one GPU that the GPU runtime (`runtime_path`)
// reports through `llama-server --list-devices`, or the CPU through the CPU
// runtime (`cpu_runtime_path`). A model is never split across devices.
use crate::runtime::{probe_output, resolve_server, runtime_command, AppState, Settings};
use serde::Serialize;
use serde_json::{json, Value};
use std::{path::PathBuf, time::Duration};

/// The `device` value that runs a model on the CPU runtime.
pub const CPU: &str = "CPU";

pub const CPU_RUNTIME_MISSING: &str = "Device CPU needs a CPU runtime. Choose or download one under CPU runtime in Runtime.";

pub fn not_found(device: &str) -> String {
    format!("Device {device} was not found: the llama.cpp build in Runtime does not report it. Choose another device in Local API.")
}

/// One device line of `llama-server --list-devices`, such as
/// `  CUDA0: NVIDIA GeForce RTX 4060 Laptop GPU (7857 MiB, 7742 MiB free)`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RuntimeDevice {
    /// What `--device` takes, such as `CUDA0` or `Vulkan1`.
    pub name: String,
    pub description: String,
    pub total_mib: Option<u64>,
    pub free_mib: Option<u64>,
}

impl RuntimeDevice {
    /// As the load panel and the instance card show it.
    pub fn display(&self) -> String {
        if self.description.is_empty() { self.name.clone() } else { format!("{} · {}", self.name, self.description) }
    }
}

/// Reads the device list from `--list-devices` stdout. Only the indented
/// lines under "Available devices:" count; "(none)" is an empty list.
/// Descriptions may contain parentheses (Vulkan names do), so only a
/// trailing "(N MiB, M MiB free)" is taken as the memory.
pub fn parse_list_devices(stdout: &str) -> Vec<RuntimeDevice> {
    let memory = regex::Regex::new(r"^(.*?)\s*\((\d+) MiB, (\d+) MiB free\)$").unwrap();
    let mut devices = Vec::new();
    let mut listing = false;
    for line in stdout.lines() {
        if line.trim() == "Available devices:" {
            listing = true;
            continue;
        }
        if !listing || line.trim().is_empty() {
            continue;
        }
        if !line.starts_with(char::is_whitespace) {
            break;
        }
        let Some((name, rest)) = line.trim().split_once(": ") else { continue };
        if name.is_empty() || name.contains(char::is_whitespace) {
            continue;
        }
        let (description, total_mib, free_mib) = match memory.captures(rest) {
            Some(c) => (c[1].to_owned(), c[2].parse().ok(), c[3].parse().ok()),
            None => (rest.trim().to_owned(), None, None),
        };
        devices.push(RuntimeDevice { name: name.to_owned(), description, total_mib, free_mib });
    }
    devices
}

/// The device a load runs on: the saved `device`, or for an empty one the
/// first GPU listed, else the CPU. `listed` is `None` when the GPU runtime
/// could not list its devices. A CPU load needs a CPU runtime; a saved GPU
/// that is not listed is refused rather than replaced.
pub fn choose_device(settings: &Settings, listed: Option<&[RuntimeDevice]>) -> Result<String, String> {
    let saved = settings.device.trim();
    let device = match saved {
        "" => listed.and_then(|l| l.first()).map_or(CPU.to_owned(), |d| d.name.clone()),
        CPU => CPU.to_owned(),
        name if listed.is_some_and(|l| l.iter().any(|d| d.name == name)) => name.to_owned(),
        name => return Err(not_found(name)),
    };
    if device == CPU && settings.cpu_runtime_path.trim().is_empty() {
        return Err(CPU_RUNTIME_MISSING.into());
    }
    Ok(device)
}

/// Whether a load runs on the CPU runtime. Takes settings whose `device` is
/// already chosen (an instance's).
pub fn on_cpu(settings: &Settings) -> bool {
    settings.device == CPU
}

/// GPU layers as launched: always 0 on the CPU.
pub fn gpu_layers(settings: &Settings) -> i32 {
    if on_cpu(settings) { 0 } else { settings.gpu_layers }
}

/// The settings a launch actually runs with: a CPU load uses the CPU runtime
/// and no GPU layers. The instance keeps its saved settings for display and
/// for comparing with the load panel.
pub fn launch_settings(settings: &Settings) -> Settings {
    let mut launch = settings.clone();
    if on_cpu(settings) {
        launch.runtime_path = settings.cpu_runtime_path.clone();
        launch.gpu_layers = 0;
    }
    launch
}

/// Restricts llama-server to the chosen device: one GPU with no splitting,
/// or none at all on the CPU.
pub fn device_args(settings: &Settings) -> Vec<String> {
    if on_cpu(settings) {
        vec!["--device".into(), "none".into()]
    } else {
        vec!["--device".into(), settings.device.clone(), "--split-mode".into(), "none".into(), "--main-gpu".into(), "0".into()]
    }
}

/// Runs `--list-devices` with `settings`' runtime (15 second limit, killed
/// on overrun).
pub async fn list_devices(settings: &Settings) -> Result<Vec<RuntimeDevice>, String> {
    resolve_server(&settings.runtime_path)?;
    let output = probe_output(runtime_command(settings).arg("--list-devices"), Duration::from_secs(15))
        .await
        .map_err(|_| "GPU detection timed out")?
        .map_err(|e| format!("Cannot detect runtime devices: {e}"))?;
    if !output.status.success() {
        return Err("Runtime device detection failed. Check the runtime path and shared libraries.".into());
    }
    Ok(parse_list_devices(&String::from_utf8_lossy(&output.stdout)))
}

/// The last device list read from the GPU runtime, by resolved llama-server
/// and shared-library folders.
pub type DeviceCache = Option<((PathBuf, String), Vec<RuntimeDevice>)>;

/// The GPU runtime's devices, from the cache while the runtime path and the
/// shared-library folders are unchanged. Failures and empty lists are not
/// kept: a CUDA build lists "(none)" when its driver failed to start, and
/// that must not stick.
pub async fn gpu_devices(state: &AppState, settings: &Settings) -> Result<Vec<RuntimeDevice>, String> {
    let key = (resolve_server(&settings.runtime_path)?, settings.library_path.clone());
    let mut cache = state.devices.lock().await;
    if let Some((cached, devices)) = cache.as_ref() {
        if *cached == key {
            return Ok(devices.clone());
        }
    }
    let devices = list_devices(settings).await?;
    *cache = (!devices.is_empty()).then(|| (key, devices.clone()));
    Ok(devices)
}

/// The GPU runtime's devices for choosing a load's device: `None` when the
/// device is "CPU" (no list needed) or the runtime could not list them.
pub async fn listed_for(state: &AppState, settings: &Settings) -> Option<Vec<RuntimeDevice>> {
    if settings.device.trim() == CPU {
        return None;
    }
    gpu_devices(state, settings).await.ok()
}

/// The load panel's "Device" options: the saved GPU runtime's devices.
#[tauri::command]
pub async fn runtime_devices(state: tauri::State<'_, AppState>) -> Result<Value, String> {
    let settings = state.settings.lock().await.clone();
    Ok(match gpu_devices(state.inner(), &settings).await {
        Ok(devices) => json!({ "runtime_path": settings.runtime_path, "devices": devices, "error": null }),
        Err(error) => json!({ "runtime_path": settings.runtime_path, "devices": [], "error": error }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Captured from `llama-b12633/llama-server --list-devices` (CUDA 13
    /// build) on an RTX 4060 Laptop GPU: stdout, then stderr empty.
    const ONE_GPU: &str = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 4060 Laptop GPU (7857 MiB, 7742 MiB free)\n";
    /// The same build with no usable GPU (CUDA_VISIBLE_DEVICES empty), and the
    /// official CPU build: exit status 0 either way. The CUDA error goes to stderr.
    const NO_GPU: &str = "Available devices:\n  (none)\n";
    const NO_GPU_STDERR: &str = "0.00.087.363 E ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected\n";

    #[test]
    fn reads_the_real_list_devices_output() {
        assert_eq!(parse_list_devices(ONE_GPU), [RuntimeDevice {
            name: "CUDA0".into(),
            description: "NVIDIA GeForce RTX 4060 Laptop GPU".into(),
            total_mib: Some(7857),
            free_mib: Some(7742),
        }]);
        assert!(parse_list_devices(NO_GPU).is_empty());
        assert!(parse_list_devices("").is_empty());
        // stderr is never read as devices, even after the list.
        assert!(parse_list_devices(&format!("{NO_GPU}{NO_GPU_STDERR}")).is_empty());
        assert!(parse_list_devices(NO_GPU_STDERR).is_empty());
    }

    #[test]
    fn reads_every_device_of_a_multi_gpu_list() {
        // Same format as the capture: two CUDA cards, a Vulkan device whose
        // description has parentheses, and an RPC device with a colon in its name.
        let text = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 4090 (24076 MiB, 23500 MiB free)\n  CUDA1: NVIDIA GeForce RTX 3060 (11910 MiB, 11000 MiB free)\n  Vulkan0: AMD Radeon Graphics (RADV GFX1103_R1) (8192 MiB, 8000 MiB free)\n  RPC[10.0.0.2:50052]: RPC[10.0.0.2:50052] (16000 MiB, 15000 MiB free)\n";
        let devices = parse_list_devices(text);
        let names: Vec<_> = devices.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, ["CUDA0", "CUDA1", "Vulkan0", "RPC[10.0.0.2:50052]"]);
        assert_eq!(devices[1].description, "NVIDIA GeForce RTX 3060");
        assert_eq!((devices[1].total_mib, devices[1].free_mib), (Some(11910), Some(11000)));
        assert_eq!(devices[2].description, "AMD Radeon Graphics (RADV GFX1103_R1)");
        assert_eq!(devices[2].total_mib, Some(8192));
        assert_eq!(devices[0].display(), "CUDA0 · NVIDIA GeForce RTX 4090");
    }

    fn settings(device: &str, cpu_runtime: &str) -> Settings {
        Settings { device: device.into(), cpu_runtime_path: cpu_runtime.into(), ..Settings::default() }
    }

    #[test]
    fn the_default_device_is_the_first_gpu_else_the_cpu() {
        let two = parse_list_devices("Available devices:\n  CUDA0: A (1 MiB, 1 MiB free)\n  CUDA1: B (1 MiB, 1 MiB free)\n");
        assert_eq!(choose_device(&settings("", ""), Some(&two)).unwrap(), "CUDA0");
        assert_eq!(choose_device(&settings("CUDA1", ""), Some(&two)).unwrap(), "CUDA1");
        assert_eq!(choose_device(&settings("", "/cpu/llama-server"), Some(&[])).unwrap(), "CPU");
        assert_eq!(choose_device(&settings("", "/cpu/llama-server"), None).unwrap(), "CPU");
        assert_eq!(choose_device(&settings("CPU", "/cpu/llama-server"), Some(&two)).unwrap(), "CPU");
    }

    #[test]
    fn a_cpu_load_without_a_cpu_runtime_is_refused() {
        let one = parse_list_devices(ONE_GPU);
        assert_eq!(choose_device(&settings("CPU", ""), Some(&one)).unwrap_err(), "Device CPU needs a CPU runtime. Choose or download one under CPU runtime in Runtime.");
        // An empty device on a machine without a GPU defaults to the CPU, so it needs one too.
        assert_eq!(choose_device(&settings("", "  "), Some(&[])).unwrap_err(), CPU_RUNTIME_MISSING);
    }

    #[test]
    fn a_saved_device_that_is_not_listed_is_refused_not_replaced() {
        let one = parse_list_devices(ONE_GPU);
        assert_eq!(choose_device(&settings("CUDA1", "/cpu/llama-server"), Some(&one)).unwrap_err(), "Device CUDA1 was not found: the llama.cpp build in Runtime does not report it. Choose another device in Local API.");
        assert_eq!(choose_device(&settings("CUDA0", "/cpu/llama-server"), None).unwrap_err(), not_found("CUDA0"));
    }
}
