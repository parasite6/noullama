// The device each load runs on: one GPU that the GPU runtime (`runtime_path`)
// reports through `llama-server --list-devices`, several of those GPUs with
// the model split across them, or the CPU through the CPU runtime
// (`cpu_runtime_path`).
//
// A split is saved in `Settings.device` as `gpus:all` (every GPU listed at
// load time) or as a comma list of names such as `CUDA0,CUDA1`. A load
// resolves either to the comma list in listed order, and that is what the
// instance keeps.
use crate::runtime::{probe_output, resolve_server, runtime_command, AppState, Settings};
use serde::Serialize;
use serde_json::{json, Value};
use std::{path::PathBuf, time::Duration};

/// The `device` value that runs a model on the CPU runtime.
pub const CPU: &str = "CPU";

/// The `device` value that splits a model across every GPU listed at load time.
pub const ALL_GPUS: &str = "gpus:all";

/// `split_mode` values: llama.cpp's `--split-mode layer|row`.
pub const SPLIT_MODES: [&str; 2] = ["layer", "row"];

pub const CPU_RUNTIME_MISSING: &str = "Device CPU needs a CPU runtime. Choose or download one under CPU runtime in Runtime.";
pub const TOO_FEW_GPUS: &str = "Choose at least 2 GPUs to split a model, or choose one device.";
pub const BAD_SPLIT_MODE: &str = "Split mode must be Layer or Row.";
pub const BAD_RATIO: &str = "Split ratio takes numbers of 0 or more separated by commas, such as 3,1.";
pub const ZERO_RATIO: &str = "Split ratio needs at least one number above 0.";

pub fn all_gpus_too_few(count: usize) -> String {
    format!("All GPUs needs at least 2 GPUs, but the llama.cpp build in Runtime reports {count}. Choose another device in Local API.")
}

pub fn ratio_count(gpus: usize, numbers: usize) -> String {
    format!("Split ratio needs one number per GPU: {gpus} GPUs are in use, but it has {numbers}.")
}

/// Whether a `device` value splits the model across GPUs (saved or chosen).
pub fn is_split(device: &str) -> bool {
    let device = device.trim();
    device == ALL_GPUS || device.contains(',')
}

/// The GPU names of a comma list, in its order, without blanks.
pub fn split_names(device: &str) -> Vec<&str> {
    device.split(',').map(str::trim).filter(|name| !name.is_empty()).collect()
}

/// The GPUs a chosen device runs on: none for the CPU, else its names.
fn gpus_of(device: &str) -> Vec<&str> {
    if device.trim() == CPU { Vec::new() } else { split_names(device) }
}

/// Whether two chosen devices have a GPU in common.
pub fn shares_gpu(a: &str, b: &str) -> bool {
    let b = gpus_of(b);
    gpus_of(a).iter().any(|name| b.contains(name))
}

/// Reads `tensor_split`: `None` when empty (llama.cpp shares by free
/// memory), else one weight per GPU. `count` is the number of GPUs in use,
/// or `None` to check only the numbers.
pub fn parse_ratio(text: &str, count: Option<usize>) -> Result<Option<Vec<f64>>, String> {
    if text.trim().is_empty() {
        return Ok(None);
    }
    let weights = text
        .split(',')
        .map(|part| part.trim().parse::<f64>().ok().filter(|w| {
            !w.is_sign_negative() && (*w == 0.0 || (f32::MIN_POSITIVE as f64..=f32::MAX as f64).contains(w))
        }))
        .collect::<Option<Vec<f64>>>()
        .ok_or(BAD_RATIO)?;
    if weights.iter().sum::<f64>() > f32::MAX as f64 {
        return Err(BAD_RATIO.into());
    }
    if let Some(count) = count.filter(|count| *count != weights.len()) {
        return Err(ratio_count(count, weights.len()));
    }
    if weights.iter().all(|w| *w == 0.0) {
        return Err(ZERO_RATIO.into());
    }
    Ok(Some(weights))
}

/// The split settings a save checks: the mode, and the ratio's numbers (its
/// count depends on the GPUs listed at load time).
pub fn check_split_settings(settings: &Settings) -> Result<(), String> {
    if !SPLIT_MODES.contains(&settings.split_mode.as_str()) {
        return Err(BAD_SPLIT_MODE.into());
    }
    parse_ratio(&settings.tensor_split, None).map(|_| ())
}

pub fn not_found(device: &str) -> String {
    format!("Device {device} was not found: the llama.cpp build in Runtime does not report it. Choose another device in Local API.")
}

/// One device line of `llama-server --list-devices`, such as
/// `  CUDA0: Example GPU (8192 MiB, 7168 MiB free)`.
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
/// that is not listed is refused rather than replaced. A split comes back
/// as a comma list in listed order, with its mode and ratio checked.
pub fn choose_device(settings: &Settings, listed: Option<&[RuntimeDevice]>) -> Result<String, String> {
    let saved = settings.device.trim();
    if is_split(saved) {
        return choose_split(settings, listed.unwrap_or(&[]));
    }
    let device = match saved {
        "" => listed.and_then(|l| l.first()).map_or(CPU.to_owned(), |d| d.name.clone()),
        CPU => CPU.to_owned(),
        name if listed.is_some_and(|l| l.iter().any(|d| d.name == name)) => name.to_owned(),
        name => return Err(not_found(name)),
    };
    if device == CPU && settings.cpu_runtime_path.trim().is_empty()
        && (!saved.is_empty() || settings.runtime_path.trim().is_empty())
    {
        return Err(CPU_RUNTIME_MISSING.into());
    }
    Ok(device)
}

/// `choose_device` for a saved split: every listed GPU for `gpus:all`, else
/// the saved names, each of which must still be listed.
fn choose_split(settings: &Settings, listed: &[RuntimeDevice]) -> Result<String, String> {
    let saved = settings.device.trim();
    let names: Vec<&str> = if saved == ALL_GPUS {
        if listed.len() < 2 {
            return Err(all_gpus_too_few(listed.len()));
        }
        listed.iter().map(|d| d.name.as_str()).collect()
    } else {
        let wanted = split_names(saved);
        if let Some(name) = wanted.iter().find(|name| !listed.iter().any(|d| d.name == **name)) {
            return Err(not_found(name));
        }
        listed.iter().map(|d| d.name.as_str()).filter(|name| wanted.contains(name)).collect()
    };
    if names.len() < 2 {
        return Err(TOO_FEW_GPUS.into());
    }
    if !SPLIT_MODES.contains(&settings.split_mode.as_str()) {
        return Err(BAD_SPLIT_MODE.into());
    }
    parse_ratio(&settings.tensor_split, Some(names.len()))?;
    Ok(names.join(","))
}

/// How the load panel and the instance card show a chosen device: a GPU
/// with its description, "CUDA0 + CUDA1" or "All GPUs (2)" for a split
/// with the GPUs' summed memory, or "CPU". `None` when a GPU is not listed.
pub fn display(chosen: &str, all_gpus: bool, listed: &[RuntimeDevice]) -> Option<String> {
    if chosen == CPU {
        return Some(CPU.to_owned());
    }
    let names = split_names(chosen);
    let gpus = names.iter().map(|name| listed.iter().find(|d| d.name == *name)).collect::<Option<Vec<_>>>()?;
    if gpus.len() == 1 {
        return Some(gpus[0].display());
    }
    let label = if all_gpus { format!("All GPUs ({})", gpus.len()) } else { names.join(" + ") };
    Some(match gpus.iter().map(|d| d.total_mib).sum::<Option<u64>>() {
        Some(mib) => format!("{label} · {} GB", (mib as f64 / 1024.0).round()),
        None => label,
    })
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
        if !settings.cpu_runtime_path.trim().is_empty() {
            launch.runtime_path = settings.cpu_runtime_path.clone();
        }
        launch.gpu_layers = 0;
    }
    launch
}

/// Restricts llama-server to the chosen device: one GPU with no splitting,
/// several GPUs with the saved split mode and ratio (none means llama.cpp
/// shares by free memory), or none at all on the CPU.
pub fn device_args(settings: &Settings) -> Vec<String> {
    if on_cpu(settings) {
        vec!["--device".into(), "none".into()]
    } else if is_split(&settings.device) {
        let mut args = vec!["--device".into(), split_names(&settings.device).join(","), "--split-mode".into(), settings.split_mode.clone()];
        if let Ok(Some(weights)) = parse_ratio(&settings.tensor_split, None) {
            let ratio: Vec<String> = weights.iter().map(ToString::to_string).collect();
            args.extend(["--tensor-split".into(), ratio.join(",")]);
        }
        args
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

    /// Example output in the format `llama-server --list-devices` (CUDA build) prints for one
    /// GPU: stdout, then stderr empty.
    const ONE_GPU: &str = "Available devices:\n  CUDA0: Example GPU (8192 MiB, 7168 MiB free)\n";
    /// The same build with no usable GPU (CUDA_VISIBLE_DEVICES empty), and the
    /// official CPU build: exit status 0 either way. The CUDA error goes to stderr.
    const NO_GPU: &str = "Available devices:\n  (none)\n";
    const NO_GPU_STDERR: &str = "0.00.087.363 E ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected\n";

    #[test]
    fn automatic_cpu_uses_the_main_runtime_until_a_dedicated_build_is_saved() {
        let mut settings = Settings { runtime_path: "/example/main".into(), ..Settings::default() };
        let mut launch = settings.clone();
        launch.device = choose_device(&settings, Some(&[])).unwrap();
        assert_eq!(launch.device, CPU);
        assert_eq!(launch_settings(&launch).runtime_path, settings.runtime_path);
        assert_eq!(launch_settings(&launch).gpu_layers, 0);
        assert!(settings.cpu_runtime_path.is_empty());

        settings.cpu_runtime_path = "/example/cpu".into();
        settings.device = choose_device(&settings, Some(&[])).unwrap();
        assert_eq!(launch_settings(&settings).runtime_path, "/example/cpu");
    }

    #[test]
    fn reads_example_list_devices_output() {
        assert_eq!(parse_list_devices(ONE_GPU), [RuntimeDevice {
            name: "CUDA0".into(),
            description: "Example GPU".into(),
            total_mib: Some(8192),
            free_mib: Some(7168),
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
        let text = "Available devices:\n  CUDA0: Example GPU A (24076 MiB, 23500 MiB free)\n  CUDA1: Example GPU B (11910 MiB, 11000 MiB free)\n  Vulkan0: Example GPU (Vulkan) (8192 MiB, 8000 MiB free)\n  RPC[192.0.2.20:50052]: RPC[192.0.2.20:50052] (16000 MiB, 15000 MiB free)\n";
        let devices = parse_list_devices(text);
        let names: Vec<_> = devices.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, ["CUDA0", "CUDA1", "Vulkan0", "RPC[192.0.2.20:50052]"]);
        assert_eq!(devices[1].description, "Example GPU B");
        assert_eq!((devices[1].total_mib, devices[1].free_mib), (Some(11910), Some(11000)));
        assert_eq!(devices[2].description, "Example GPU (Vulkan)");
        assert_eq!(devices[2].total_mib, Some(8192));
        assert_eq!(devices[0].display(), "CUDA0 · Example GPU A");
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

    /// Two made-up GPUs, as a CUDA build lists them.
    const TWO_GPUS: &str = "Available devices:\n  CUDA0: Example GPU A (10240 MiB, 9000 MiB free)\n  CUDA1: Example GPU B (24576 MiB, 23000 MiB free)\n";

    fn split(device: &str, mode: &str, ratio: &str) -> Settings {
        Settings { device: device.into(), split_mode: mode.into(), tensor_split: ratio.into(), ..Settings::default() }
    }

    #[test]
    fn a_split_resolves_to_the_listed_gpus_in_listed_order() {
        let two = parse_list_devices(TWO_GPUS);
        assert_eq!(choose_device(&split(ALL_GPUS, "layer", ""), Some(&two)).unwrap(), "CUDA0,CUDA1");
        assert_eq!(choose_device(&split("CUDA1, CUDA0", "row", "3,1"), Some(&two)).unwrap(), "CUDA0,CUDA1");
        // A split needs no CPU runtime.
        assert!(split(ALL_GPUS, "layer", "").cpu_runtime_path.is_empty());
    }

    #[test]
    fn a_split_is_refused_rather_than_changed() {
        let two = parse_list_devices(TWO_GPUS);
        let one = parse_list_devices(ONE_GPU);
        // Every saved name must still be listed; none is swapped for another.
        assert_eq!(choose_device(&split("CUDA0,CUDA2", "layer", ""), Some(&two)).unwrap_err(), not_found("CUDA2"));
        assert_eq!(choose_device(&split("CUDA0,CUDA1", "layer", ""), None).unwrap_err(), not_found("CUDA0"));
        // Fewer than 2 GPUs chosen, or listed for All GPUs.
        for few in ["CUDA0,", ",", "CUDA0,CUDA0"] {
            assert_eq!(choose_device(&split(few, "layer", ""), Some(&two)).unwrap_err(), TOO_FEW_GPUS, "{few}");
        }
        assert_eq!(choose_device(&split(ALL_GPUS, "layer", ""), Some(&one)).unwrap_err(), all_gpus_too_few(1));
        assert_eq!(choose_device(&split(ALL_GPUS, "layer", ""), None).unwrap_err(), all_gpus_too_few(0));
        // The ratio needs one number per GPU, all 0 or more, not all 0.
        assert_eq!(choose_device(&split(ALL_GPUS, "layer", "3,1,1"), Some(&two)).unwrap_err(), "Split ratio needs one number per GPU: 2 GPUs are in use, but it has 3.");
        assert_eq!(choose_device(&split(ALL_GPUS, "layer", "3"), Some(&two)).unwrap_err(), ratio_count(2, 1));
        for bad in ["3,-1", "3,x", "3,,1", "inf,1", "1e39,1", "3e38,3e38", "1e-40,1"] {
            assert_eq!(choose_device(&split(ALL_GPUS, "layer", bad), Some(&two)).unwrap_err(), BAD_RATIO, "{bad}");
        }
        assert_eq!(choose_device(&split(ALL_GPUS, "layer", "0,0"), Some(&two)).unwrap_err(), ZERO_RATIO);
        assert_eq!(choose_device(&split(ALL_GPUS, "tensor", ""), Some(&two)).unwrap_err(), BAD_SPLIT_MODE);
        // A single GPU ignores the split settings.
        assert_eq!(choose_device(&split("CUDA1", "bogus", "1,2,3"), Some(&two)).unwrap(), "CUDA1");
    }

    #[test]
    fn split_arguments_name_every_gpu_and_never_a_main_gpu() {
        let args = device_args(&split("CUDA0,CUDA1", "layer", ""));
        assert_eq!(args, ["--device", "CUDA0,CUDA1", "--split-mode", "layer"]);
        let args = device_args(&split("CUDA0,CUDA1", "row", " 3, 1 "));
        assert_eq!(args, ["--device", "CUDA0,CUDA1", "--split-mode", "row", "--tensor-split", "3,1"]);
        let args = device_args(&split("CUDA0,CUDA1", "row", "1e-999,1"));
        assert_eq!(args, ["--device", "CUDA0,CUDA1", "--split-mode", "row", "--tensor-split", "0,1"]);
        // One GPU and the CPU launch exactly as before, whatever the split settings say.
        assert_eq!(device_args(&split("CUDA1", "row", "3,1")), ["--device", "CUDA1", "--split-mode", "none", "--main-gpu", "0"]);
        assert_eq!(device_args(&split(CPU, "row", "3,1")), ["--device", "none"]);
        let old: Settings = serde_json::from_str(r#"{"device":"CUDA1"}"#).unwrap();
        assert_eq!(old.split_mode, "layer");
        assert!(old.tensor_split.is_empty());
        assert_eq!(device_args(&old), device_args(&split("CUDA1", "row", "3,1")));
    }

    #[test]
    fn a_split_is_shown_by_its_gpus_with_their_summed_memory() {
        let two = parse_list_devices(TWO_GPUS);
        assert_eq!(display("CUDA0,CUDA1", false, &two).unwrap(), "CUDA0 + CUDA1 · 34 GB");
        assert_eq!(display("CUDA0,CUDA1", true, &two).unwrap(), "All GPUs (2) · 34 GB");
        assert_eq!(display("CUDA1", false, &two).unwrap(), "CUDA1 · Example GPU B");
        assert_eq!(display(CPU, false, &[]).unwrap(), "CPU");
        assert!(display("CUDA0,CUDA2", false, &two).is_none());
        let unread = parse_list_devices("Available devices:\n  CUDA0: A\n  CUDA1: B (1024 MiB, 1000 MiB free)\n");
        assert_eq!(display("CUDA0,CUDA1", false, &unread).unwrap(), "CUDA0 + CUDA1");
    }

    #[test]
    fn devices_share_a_gpu_when_any_name_is_in_both() {
        assert!(shares_gpu("CUDA0", "CUDA0"));
        assert!(shares_gpu("CUDA0,CUDA1", "CUDA1"));
        assert!(!shares_gpu("CUDA0", "CUDA1"));
        assert!(!shares_gpu(CPU, CPU));
        assert!(!shares_gpu("CUDA0,CUDA1", CPU));
    }
}
