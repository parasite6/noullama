//! OOM policy: utilization is not pressure, and fit estimates are not measurements.
const MIB: u64 = 1024 * 1024;

#[derive(Clone, Copy)]
pub struct Memory {
    pub total: u64,
    pub available: u64,
    critical: bool,
}

#[cfg(any(target_os = "linux", test))]
struct MemorySource {
    total: u64,
    available: u64,
    stalled: bool,
}

#[cfg(any(target_os = "linux", test))]
impl MemorySource {
    fn sample(self) -> Memory {
        Memory {
            total: self.total,
            available: self.available,
            critical: self.total > 0 && self.available < (self.total / 100).min(256 * MIB) && self.stalled,
        }
    }
}

#[cfg(any(target_os = "linux", test))]
fn merge(memory: Memory, source: MemorySource) -> Memory {
    let source = source.sample();
    Memory {
        total: memory.total.min(source.total),
        available: memory.available.min(source.available),
        critical: memory.critical || source.critical,
    }
}

pub fn clearly_exceeds(required: u64, available: u64) -> bool {
    required > available.saturating_add((required / 5).max(512 * MIB))
}

#[derive(Default)]
pub struct Pressure {
    consecutive: u8,
}
impl Pressure {
    pub fn sample(&mut self, memory: Option<Memory>) -> bool {
        let critical = memory.is_some_and(|m| m.critical);
        self.consecutive = if critical {
            self.consecutive.saturating_add(1)
        } else {
            0
        };
        self.consecutive >= 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn critical_host_survives_a_tighter_non_stalled_cgroup() {
        let host = MemorySource { total: 8000 * MIB, available: 32 * MIB, stalled: true };
        let cgroup = MemorySource { total: 1000 * MIB, available: 16 * MIB, stalled: false };
        let sample = merge(host.sample(), cgroup);
        assert_eq!(sample.available, 16 * MIB);
        assert_eq!(sample.total, 1000 * MIB);
        assert!(sample.critical);
    }

    #[test]
    fn high_utilization_is_not_oom() {
        let mut pressure = Pressure::default();
        for _ in 0..10 {
            assert!(!pressure.sample(Some(MemorySource {
                total: 8000 * MIB,
                available: 1000 * MIB,
                stalled: true
            }.sample())));
        }
        assert!(!clearly_exceeds(7000 * MIB, 8000 * MIB));
        assert!(!clearly_exceeds(8500 * MIB, 8000 * MIB));
        assert!(clearly_exceeds(12000 * MIB, 8000 * MIB));
    }
    #[test]
    fn requires_sustained_pressure_and_resets_on_unknown() {
        let source = MemorySource {
            total: 8000 * MIB,
            available: 32 * MIB,
            stalled: true,
        };
        let critical = Some(MemorySource { ..source }.sample());
        let mut pressure = Pressure::default();
        assert!(!pressure.sample(critical));
        assert!(!pressure.sample(None));
        assert!(!pressure.sample(critical));
        assert!(!pressure.sample(critical));
        assert!(pressure.sample(critical));
        assert!(!pressure.sample(Some(MemorySource {
            stalled: false,
            ..source
        }.sample())));
    }
}

#[cfg(target_os = "linux")]
fn psi_stalled(text: &str) -> bool {
    text.lines()
        .find(|line| line.starts_with("full "))
        .and_then(|line| {
            line.split_whitespace()
                .find_map(|v| v.strip_prefix("avg10="))
        })
        .and_then(|v| v.parse::<f64>().ok())
        .is_some_and(|v| v >= 5.0)
}

#[cfg(target_os = "linux")]
pub fn memory() -> Option<Memory> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| {
        text.lines()
            .find_map(|line| {
                line.strip_prefix(name)?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()
            })
            .map(|v| v.saturating_mul(1024))
    };
    let mut result = MemorySource {
        total: field("MemTotal:")?,
        available: field("MemAvailable:")?,
        stalled: std::fs::read_to_string("/proc/pressure/memory")
            .ok()
            .is_some_and(|v| psi_stalled(&v)),
    }.sample();
    // The process may be constrained even when the host has plenty of RAM.
    // Inspect ancestors too: limits are hierarchical in cgroup v2.
    if let Ok(groups) = std::fs::read_to_string("/proc/self/cgroup") {
        if let Some(relative) = groups.lines().find_map(|l| l.strip_prefix("0::")) {
            let root = std::path::Path::new("/sys/fs/cgroup");
            let mut path = root.join(relative.trim_start_matches('/'));
            while path.starts_with(root) {
                let read = |name| {
                    std::fs::read_to_string(path.join(name))
                        .ok()?
                        .trim()
                        .parse::<u64>()
                        .ok()
                };
                if let (Some(limit), Some(used)) = (read("memory.max"), read("memory.current")) {
                    result = merge(result, MemorySource {
                        total: limit,
                        available: limit.saturating_sub(used),
                        stalled: std::fs::read_to_string(path.join("memory.pressure"))
                            .ok()
                            .is_some_and(|v| psi_stalled(&v)),
                    });
                }
                if path == root || !path.pop() {
                    break;
                }
            }
        }
    }
    Some(result)
}

#[cfg(not(target_os = "linux"))]
pub fn memory() -> Option<Memory> {
    None
}

#[cfg(target_os = "linux")]
pub fn resident_bytes(pid: u32) -> Option<u64> {
    std::fs::read_to_string(format!("/proc/{pid}/status"))
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmRSS:")?
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
        })
        .map(|kb| kb.saturating_mul(1024))
}
#[cfg(not(target_os = "linux"))]
pub fn resident_bytes(_pid: u32) -> Option<u64> {
    None
}

#[cfg(all(test, target_os = "linux"))]
mod linux_tests {
    use super::*;
    #[test]
    fn pressure_requires_full_stalls_and_valid_measurement() {
        assert!(!psi_stalled(
            "some avg10=90.0 avg60=3.0\nfull avg10=0.0 avg60=0.0"
        ));
        assert!(psi_stalled("full avg10=5.00 avg60=0.1"));
        assert!(!psi_stalled("full avg10=garbage"));
        assert!(!psi_stalled(""));
    }
    #[test]
    fn live_probe_is_bounded_and_owned_process_is_readable() {
        let memory = memory().expect("Linux MemAvailable");
        assert!(memory.total > 0);
        assert!(memory.available <= memory.total);
        assert!(resident_bytes(std::process::id()).unwrap() > 0);
    }
}
