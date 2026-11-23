// Rough, deliberately simple model/hardware fit estimate. The numbers come from
// the file size plus an allowance for KV cache and compute buffers, compared
// against the memory the model can actually use on this machine.
use serde::Serialize;
use std::sync::OnceLock;
use std::time::Duration;

#[derive(Clone, Serialize, Default)]
pub struct Hardware {
    pub ram_bytes: u64,
    pub vram_bytes: Option<u64>,
    pub gpu: Option<String>,
    pub cpu_cores: usize,
}

#[derive(Clone, Serialize)]
pub struct Fit {
    pub level: u8,
    pub label: &'static str,
    pub detail: String,
    pub model_bytes: u64,
    pub required_bytes: u64,
}

static CACHE: OnceLock<Hardware> = OnceLock::new();

pub fn hardware() -> &'static Hardware {
    CACHE.get_or_init(probe)
}

/// The probe result if it has finished; never waits on nvidia-smi.
pub fn cached() -> Option<&'static Hardware> {
    CACHE.get()
}

fn file_bytes<S: AsRef<str>>(path: &str, extras: &[S]) -> Option<u64> {
    let file = std::fs::metadata(path).ok()?;
    if !file.is_file() || file.len() == 0 {
        return None;
    }
    let extras: u64 = extras
        .iter()
        .map(AsRef::as_ref)
        .filter(|path| !path.is_empty())
        .filter_map(|path| std::fs::metadata(path).ok())
        .filter(std::fs::Metadata::is_file)
        .map(|file| file.len())
        .sum();
    Some(file.len() + extras)
}

/// The memory a model is estimated to need once loaded.
pub fn estimate_bytes<S: AsRef<str>>(path: &str, extras: &[S], context_size: u32) -> Option<u64> {
    file_bytes(path, extras).map(|bytes| required_bytes(bytes, context_size))
}

fn required_bytes(model_bytes: u64, context_size: u32) -> u64 {
    // Weight overhead plus a KV-cache allowance for the configured context.
    model_bytes + (model_bytes as f64 * 0.12) as u64 + context_size as u64 * 24 * 1024
}

pub const MODEL_UNREADABLE: &str = "The selected model file could not be read.";
pub const NO_SYSTEM_MEMORY: &str = "System memory could not be detected, so there is no estimate.";

/// How a model file fits `hardware`, with the memory other loaded models
/// are estimated to hold set aside: each `(bytes, on_gpu)` in `taken` comes
/// out of graphics or system memory. `extras` are files loaded alongside the
/// model, such as a multimodal projector or an MTP head; they sit in the
/// same memory as the weights, so they count toward the estimate. Empty
/// paths are skipped. `hardware.vram_bytes` is the memory of the GPU the
/// load runs on.
pub fn fit_beside(path: &str, extras: &[&str], context_size: u32, gpu_layers: i32, hardware: &Hardware, taken: &[(u64, bool)]) -> Result<Fit, &'static str> {
    let bytes = file_bytes(path, extras).ok_or(MODEL_UNREADABLE)?;
    let mut hardware = hardware.clone();
    for (used, on_gpu) in taken {
        match (on_gpu, hardware.vram_bytes.as_mut()) {
            (true, Some(vram)) => *vram = vram.saturating_sub(*used).max(1),
            _ => hardware.ram_bytes = hardware.ram_bytes.saturating_sub(*used).max(1),
        }
    }
    fit_model_bytes(bytes, context_size, gpu_layers, &hardware).ok_or(NO_SYSTEM_MEMORY)
}

#[derive(Serialize, Default, Debug, PartialEq)]
pub struct MemoryUse {
    /// Graphics memory held by all the runtime processes (NVIDIA only).
    pub process_bytes: Option<u64>,
    /// Graphics memory of each process asked about, in the same order.
    pub processes: Vec<Option<u64>>,
    pub used_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
}

/// Live graphics memory, read on demand (it runs nvidia-smi, so never per snapshot).
pub fn memory_use(pids: &[u32]) -> MemoryUse {
    if let Some(text) = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
    ) {
        let apps = (!pids.is_empty()).then(|| {
            command_output(
                "nvidia-smi",
                &[
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
            )
        }).flatten();
        return parse_nvidia_memory(&text, apps.as_deref(), pids);
    }
    amd_memory()
}

/// One GPU as nvidia-smi lists it, in its index order.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuReading {
    pub name: String,
    pub used_bytes: u64,
    pub total_bytes: u64,
}

/// Every NVIDIA GPU's memory, in nvidia-smi index order (PCI bus order);
/// `None` without nvidia-smi.
pub fn nvidia_gpus() -> Option<Vec<GpuReading>> {
    command_output("nvidia-smi", &["--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"])
        .map(|text| parse_nvidia_gpus(&text))
}

/// `index, name, used, total` lines. The index comes off the front and the
/// two numbers off the back, so a comma inside a name cannot shift them.
fn parse_nvidia_gpus(text: &str) -> Vec<GpuReading> {
    text.lines()
        .filter_map(|line| {
            let (_, rest) = line.split_once(',')?;
            let (rest, total) = rest.rsplit_once(',')?;
            let (name, used) = rest.rsplit_once(',')?;
            Some(GpuReading { name: name.trim().to_owned(), used_bytes: mib(used)?, total_bytes: mib(total)? })
        })
        .collect()
}

/// Which nvidia-smi GPU (an index into `gpus`) is the llama.cpp device
/// `device` among `devices`, matched by name:
/// - a name that only one listed device and only one nvidia-smi GPU have
///   is that GPU;
/// - several GPUs with the same name are matched in order only when both
///   lists count the same number of them and `pci_order` holds (CUDA set to
///   enumerate in PCI bus order, as nvidia-smi does, with no device remapping);
/// - otherwise, including a device nvidia-smi does not list, `None`.
pub fn match_gpu(device: &str, devices: &[crate::devices::RuntimeDevice], gpus: &[GpuReading], pci_order: bool) -> Option<usize> {
    let chosen = devices.iter().find(|d| d.name == device)?;
    let twins: Vec<_> = devices.iter().filter(|d| d.description == chosen.description).collect();
    let rank = twins.iter().position(|d| d.name == device)?;
    let cards: Vec<usize> = gpus.iter().enumerate().filter(|(_, g)| g.name == chosen.description).map(|(i, _)| i).collect();
    match (twins.len(), cards.len()) {
        (1, 1) => Some(cards[0]),
        (a, b) if a == b && a > 1 && pci_order => Some(cards[rank]),
        _ => None,
    }
}

/// Whether CUDA numbers its devices in the order nvidia-smi does: only when
/// `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set and `CUDA_VISIBLE_DEVICES` is not.
/// CUDA's default order puts the fastest card first.
pub fn cuda_in_pci_order(device: &str) -> bool {
    device.starts_with("CUDA")
        && std::env::var("CUDA_DEVICE_ORDER").is_ok_and(|v| v == "PCI_BUS_ID")
        && std::env::var_os("CUDA_VISIBLE_DEVICES").is_none()
}

/// Live NVIDIA memory for the selected device when it maps unambiguously.
pub fn nvidia_device_memory(device: &str, devices: &[crate::devices::RuntimeDevice]) -> Option<(u64, u64)> {
    if !device.starts_with("CUDA") {
        return None;
    }
    let gpus = nvidia_gpus()?;
    let index = match_gpu(device, devices, &gpus, cuda_in_pci_order(device))?;
    Some((gpus[index].total_bytes, gpus[index].used_bytes))
}

/// Memory from a fresh runtime probe of the selected device.
pub fn listed_device_memory(device: &str, devices: &[crate::devices::RuntimeDevice]) -> Option<(u64, u64)> {
    let reading = devices.iter().find(|entry| entry.name == device)?;
    let total = reading.total_mib?.saturating_mul(1024 * 1024);
    let free = reading.free_mib?.saturating_mul(1024 * 1024);
    (total > 0 && free <= total).then(|| (total, total - free))
}

fn mib(text: &str) -> Option<u64> {
    text.trim().parse::<u64>().ok().map(|mib| mib * 1024 * 1024)
}

fn parse_nvidia_memory(gpus: &str, apps: Option<&str>, pids: &[u32]) -> MemoryUse {
    let (used_bytes, total_bytes) = gpus
        .lines()
        .next()
        .and_then(|line| line.split_once(','))
        .map_or((None, None), |(used, total)| (mib(used), mib(total)));
    let processes: Vec<Option<u64>> = pids
        .iter()
        .map(|pid| {
            apps.and_then(|apps| {
                apps.lines()
                    .filter_map(|line| line.split_once(','))
                    .filter(|(id, _)| id.trim().parse::<u32>().ok() == Some(*pid))
                    .filter_map(|(_, used)| mib(used))
                    .reduce(|a, b| a + b)
            })
        })
        .collect();
    let process_bytes = processes.iter().flatten().copied().reduce(|a, b| a + b);
    MemoryUse {
        process_bytes,
        processes,
        used_bytes,
        total_bytes,
    }
}

#[cfg(target_os = "linux")]
fn amd_memory() -> MemoryUse {
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return MemoryUse::default();
    };
    for entry in entries.flatten() {
        let device = entry.path().join("device");
        let read = |name: &str| {
            std::fs::read_to_string(device.join(name))
                .ok()
                .and_then(|text| text.trim().parse::<u64>().ok())
        };
        if let (Some(used), Some(total)) = (read("mem_info_vram_used"), read("mem_info_vram_total"))
        {
            if total > 0 {
                return MemoryUse {
                    process_bytes: None,
                    processes: Vec::new(),
                    used_bytes: Some(used),
                    total_bytes: Some(total),
                };
            }
        }
    }
    MemoryUse::default()
}

#[cfg(not(target_os = "linux"))]
fn amd_memory() -> MemoryUse {
    MemoryUse::default()
}

pub fn fit_model_bytes(
    model_bytes: u64,
    context_size: u32,
    gpu_layers: i32,
    hardware: &Hardware,
) -> Option<Fit> {
    if model_bytes == 0 || hardware.ram_bytes == 0 {
        return None;
    }
    let required_bytes = required_bytes(model_bytes, context_size);
    let (level, label, detail) = evaluate(required_bytes, hardware, gpu_layers != 0);
    Some(Fit {
        level,
        label,
        detail,
        model_bytes,
        required_bytes,
    })
}

fn evaluate(required: u64, hardware: &Hardware, offload: bool) -> (u8, &'static str, String) {
    // Leave room for the desktop, the runtime and the page cache.
    let ram = (hardware.ram_bytes as f64 * 0.7) as u64;
    if offload {
        if let Some(vram) = hardware.vram_bytes {
            let gpu = (vram as f64 * 0.9) as u64;
            if required * 100 <= gpu * 60 {
                return (
                    1,
                    "Comfortable",
                    format!(
                        "About {} needed. Well inside the {} of graphics memory available.",
                        gb(required),
                        gb(gpu)
                    ),
                );
            }
            if required <= gpu {
                return (
                    2,
                    "Capable",
                    format!(
                        "About {} needed of the {} of graphics memory available.",
                        gb(required),
                        gb(gpu)
                    ),
                );
            }
            if required <= ram {
                return (
                    3,
                    "Strained",
                    format!(
                        "About {} needed — more than the {} of graphics memory. It would run from system memory.",
                        gb(required),
                        gb(gpu)
                    ),
                );
            }
            return (
                4,
                "Too large",
                format!(
                    "About {} needed. This machine offers {} of graphics memory and {} of system memory.",
                    gb(required),
                    gb(gpu),
                    gb(ram)
                ),
            );
        }
    }
    if required * 100 <= ram * 50 {
        (
            1,
            "Comfortable",
            format!(
                "About {} needed. Well inside the {} of system memory available.",
                gb(required),
                gb(ram)
            ),
        )
    } else if required * 100 <= ram * 80 {
        (
            2,
            "Capable",
            format!(
                "About {} needed of the {} of system memory available.",
                gb(required),
                gb(ram)
            ),
        )
    } else if required <= ram {
        (
            3,
            "Strained",
            format!(
                "About {} needed of the {} of system memory available. Expect slow responses.",
                gb(required),
                gb(ram)
            ),
        )
    } else {
        (
            4,
            "Too large",
            format!(
                "About {} needed. This machine offers about {} of system memory.",
                gb(required),
                gb(ram)
            ),
        )
    }
}

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

fn probe() -> Hardware {
    let vram = vram();
    Hardware {
        ram_bytes: ram_bytes(),
        vram_bytes: vram.as_ref().map(|(bytes, _)| *bytes),
        gpu: vram.map(|(_, name)| name),
        cpu_cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
    }
}

#[cfg(target_os = "linux")]
fn ram_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("MemTotal:"))
                .and_then(|line| line.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

#[cfg(target_os = "macos")]
fn ram_bytes() -> u64 {
    command_output("sysctl", &["-n", "hw.memsize"])
        .and_then(|text| text.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn ram_bytes() -> u64 {
    0
}

fn vram() -> Option<(u64, String)> {
    nvidia().or_else(amd)
}

fn nvidia() -> Option<(u64, String)> {
    let text = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let line = text.lines().next()?.trim();
    let (name, memory) = line.rsplit_once(',')?;
    let mib: u64 = memory.trim().parse().ok()?;
    if mib == 0 {
        return None;
    }
    Some((mib * 1024 * 1024, name.trim().to_string()))
}

#[cfg(target_os = "linux")]
fn amd() -> Option<(u64, String)> {
    let entries = std::fs::read_dir("/sys/class/drm").ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(card) = name.strip_prefix("card") else {
            continue;
        };
        if card.is_empty() || !card.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let device = entry.path().join("device");
        let bytes = std::fs::read_to_string(device.join("mem_info_vram_total"))
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .filter(|bytes| *bytes > 0);
        if let Some(bytes) = bytes {
            let label = match std::fs::read_to_string(device.join("uevent"))
                .ok()
                .and_then(|text| {
                    text.lines()
                        .find(|line| line.starts_with("PCI_ID="))
                        .map(|line| line.to_string())
                }) {
                Some(id) if id.to_ascii_lowercase().contains("1002") => "AMD Radeon".to_string(),
                Some(id) if id.to_ascii_lowercase().contains("8086") => {
                    "Intel graphics".to_string()
                }
                _ => "Graphics card".to_string(),
            };
            return Some((bytes, label));
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn amd() -> Option<(u64, String)> {
    None
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    use std::process::{Command, Stdio};
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .stdout(Stdio::piped())
        .spawn()
        .ok()?;
    let start = std::time::Instant::now();
    while child.try_wait().ok()?.is_none() {
        if start.elapsed() > Duration::from_secs(2) {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    #[test]
    fn runtime_memory_uses_only_the_selected_device() {
        let devices = crate::devices::parse_list_devices(
            "Available devices:\n  CUDA0: Example GPU (8192 MiB, 1024 MiB free)\n  CUDA1: Example GPU (8192 MiB, 7168 MiB free)\n  CUDA2: Example GPU\n",
        );
        let mib = 1024 * 1024;
        assert_eq!(super::listed_device_memory("CUDA1", &devices), Some((8192 * mib, 1024 * mib)));
        assert_eq!(super::listed_device_memory("CUDA2", &devices), None);
        assert_eq!(super::listed_device_memory("CUDA3", &devices), None);
    }

    use super::*;

    #[test]
    fn nvidia_memory_reads_the_runtime_process_and_the_card() {
        let usage = parse_nvidia_memory("3120, 10240\n", Some("911, 300\n4242, 2500\n77, 100\n"), &[4242, 77, 5]);
        assert_eq!(usage.used_bytes, Some(3120 * 1024 * 1024));
        assert_eq!(usage.total_bytes, Some(10240 * 1024 * 1024));
        assert_eq!(usage.process_bytes, Some(2600 * 1024 * 1024));
        assert_eq!(usage.processes, [Some(2500 * 1024 * 1024), Some(100 * 1024 * 1024), None]);
        assert_eq!(
            parse_nvidia_memory("3120, 10240", Some("911, 300"), &[4242]).process_bytes,
            None
        );
        assert_eq!(
            parse_nvidia_memory("garbage", None, &[]),
            MemoryUse::default()
        );
    }

    #[test]
    fn llama_devices_map_to_their_own_nvidia_gpu() {
        use crate::devices::parse_list_devices;
        // The format nvidia-smi prints for `index,name,memory.used,memory.total`
        // (for example "0, Example GPU, 15, 10240").
        let gpus = parse_nvidia_gpus("0, Example GPU A, 100, 12288\n1, Example GPU B, 2000, 24564\n2, Odd, Name, 5, 10\n");
        assert_eq!(gpus.len(), 3);
        assert_eq!(gpus[2].name, "Odd, Name");
        assert_eq!((gpus[1].used_bytes, gpus[1].total_bytes), (2000 << 20, 24564 << 20));
        // CUDA lists the faster card first; names put each on its own GPU.
        let devices = parse_list_devices("Available devices:\n  CUDA0: Example GPU B (24076 MiB, 22000 MiB free)\n  CUDA1: Example GPU A (11910 MiB, 11800 MiB free)\n");
        assert_eq!(match_gpu("CUDA0", &devices, &gpus, false), Some(1));
        assert_eq!(match_gpu("CUDA1", &devices, &gpus, false), Some(0));
        assert_eq!(match_gpu("CUDA7", &devices, &gpus, false), None);
        // Two identical cards: their order is only known in PCI bus order.
        let twins = parse_nvidia_gpus("0, Example GPU C, 1, 23028\n1, Example GPU C, 2, 23028\n");
        let devices = parse_list_devices("Available devices:\n  CUDA0: Example GPU C (22000 MiB, 21000 MiB free)\n  CUDA1: Example GPU C (22000 MiB, 21000 MiB free)\n");
        assert_eq!(match_gpu("CUDA1", &devices, &twins, false), None);
        assert_eq!(match_gpu("CUDA1", &devices, &twins, true), Some(1));
        // A device nvidia-smi does not list (AMD over Vulkan) has no reading of its own.
        let amd = parse_list_devices("Available devices:\n  Vulkan0: Example GPU (Vulkan) (24560 MiB, 24000 MiB free)\n");
        assert_eq!(match_gpu("Vulkan0", &amd, &gpus, true), None);
    }

    fn machine(ram_gb: u64, vram_gb: Option<u64>) -> Hardware {
        Hardware {
            ram_bytes: ram_gb * 1_000_000_000,
            vram_bytes: vram_gb.map(|gb| gb * 1_000_000_000),
            gpu: vram_gb.map(|_| "Test GPU".to_string()),
            cpu_cores: 8,
        }
    }

    #[test]
    fn gpu_fit_levels_follow_available_graphics_memory() {
        let hw = machine(32, Some(12));
        assert_eq!(evaluate(2_000_000_000, &hw, true).0, 1);
        assert_eq!(evaluate(9_000_000_000, &hw, true).0, 2);
        assert_eq!(evaluate(15_000_000_000, &hw, true).0, 3);
        assert_eq!(evaluate(40_000_000_000, &hw, true).0, 4);
    }

    #[test]
    fn cpu_fit_levels_follow_system_memory() {
        let hw = machine(16, None);
        assert_eq!(evaluate(3_000_000_000, &hw, true).0, 1);
        assert_eq!(evaluate(8_000_000_000, &hw, true).0, 2);
        assert_eq!(evaluate(11_000_000_000, &hw, true).0, 3);
        assert_eq!(evaluate(20_000_000_000, &hw, true).0, 4);
    }

    #[test]
    fn disabling_offload_ignores_graphics_memory() {
        let hw = machine(32, Some(24));
        assert_eq!(evaluate(2_000_000_000, &hw, false).0, 1);
        assert_eq!(evaluate(20_000_000_000, &hw, false).0, 3);
        assert_eq!(evaluate(30_000_000_000, &hw, false).0, 4);
    }

    #[test]
    fn detail_mentions_the_budget_that_applies() {
        let gpu = machine(32, Some(12));
        assert!(evaluate(15_000_000_000, &gpu, true)
            .2
            .contains("system memory"));
        let ram = machine(16, None);
        assert!(evaluate(11_000_000_000, &ram, true)
            .2
            .contains("system memory"));
        assert!(evaluate(2_000_000_000, &gpu, true)
            .2
            .contains("graphics memory"));
    }

    #[test]
    fn missing_or_empty_files_have_no_rating() {
        assert!(estimate_bytes::<&str>("/definitely/not/a/model.gguf", &[], 8192).is_none());
        let empty = std::env::temp_dir().join(format!("noullama-empty-{}.gguf", uuid::Uuid::new_v4()));
        std::fs::write(&empty, b"").unwrap();
        assert!(estimate_bytes::<&str>(empty.to_str().unwrap(), &[], 8192).is_none());
        std::fs::remove_file(empty).unwrap();
    }

    #[test]
    fn fit_can_be_estimated_from_a_hub_file_size() {
        let hw = machine(32, Some(12));
        let fit = fit_model_bytes(2_000_000_000, 8192, 99, &hw).unwrap();
        assert_eq!(fit.model_bytes, 2_000_000_000);
        assert_eq!(fit.level, 1);
        assert_eq!(fit.label, "Comfortable");
    }
}
