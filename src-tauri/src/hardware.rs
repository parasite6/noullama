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

pub fn hardware() -> &'static Hardware {
    static CACHE: OnceLock<Hardware> = OnceLock::new();
    CACHE.get_or_init(probe)
}

pub fn fit(path: &str, context_size: u32, gpu_layers: i32) -> Option<Fit> {
    let file = std::fs::metadata(path).ok()?;
    if !file.is_file() || file.len() == 0 {
        return None;
    }
    fit_model_bytes(file.len(), context_size, gpu_layers, hardware())
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
    // Weight overhead plus a KV-cache allowance for the configured context.
    let required_bytes =
        model_bytes + (model_bytes as f64 * 0.12) as u64 + context_size as u64 * 24 * 1024;
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
    use super::*;

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
        assert!(fit("/definitely/not/a/model.gguf", 8192, 99).is_none());
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
