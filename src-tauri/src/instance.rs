use std::{fs::File, path::Path, time::Duration};

// Keep the lock file itself in place: unlinking a locked file can let two
// launches lock different inodes and both believe they own the application.
pub struct InstanceGuard {
    _lock: File,
}

pub fn acquire(data_dir: &Path) -> Result<InstanceGuard, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(data_dir)?;
    let root = std::fs::canonicalize(data_dir)?;
    let mut options = std::fs::OpenOptions::new();
    options.create(true).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(root.join(".instance.lock"))?;
    #[cfg(target_os = "linux")]
    linux::clean_older_instances(&root)?;
    for _ in 0..100 {
        match fs2::FileExt::try_lock_exclusive(&file) {
            Ok(()) => {
                #[cfg(target_os = "linux")]
                linux::clean_owned_runtimes(&root)?;
                return Ok(InstanceGuard { _lock: file });
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(50))
            }
            Err(error) => return Err(error.into()),
        }
    }
    Err("Another Noullama instance is still shutting down. Its processes were not replaced blindly; try launching again after it exits.".into())
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::{
        os::{
            fd::{AsRawFd, FromRawFd, OwnedFd},
            unix::fs::MetadataExt,
        },
        path::PathBuf,
    };

    struct Process {
        pid: u32,
        start: u64,
        exe: PathBuf,
        data_dir: Option<PathBuf>,
        args: Vec<String>,
    }

    fn start_time(stat: &str) -> Option<u64> {
        stat.get(stat.rfind(')')? + 1..)?
            .split_whitespace()
            .nth(19)?
            .parse()
            .ok()
    }

    fn read(pid: u32) -> Option<Process> {
        let root = PathBuf::from(format!("/proc/{pid}"));
        // Only the current user's processes are eligible, even if a similarly
        // named system service happens to hold the configured port.
        if std::fs::metadata(&root).ok()?.uid() != unsafe { libc::getuid() } {
            return None;
        }
        let exe = std::fs::read_link(root.join("exe")).ok()?;
        let exe = PathBuf::from(
            exe.to_string_lossy()
                .strip_suffix(" (deleted)")
                .unwrap_or(&exe.to_string_lossy()),
        );
        let this_exe = std::env::current_exe().ok()?;
        if pid != std::process::id()
            && exe != this_exe
            && exe.file_name().is_none_or(|name| name != "llama-server")
        {
            return None;
        }
        let start = start_time(&std::fs::read_to_string(root.join("stat")).ok()?)?;
        let env = std::fs::read(root.join("environ")).ok()?;
        let value = |key: &str| {
            env.split(|b| *b == 0)
                .find_map(|entry| {
                    entry
                        .strip_prefix(key.as_bytes())
                        .and_then(|v| std::str::from_utf8(v).ok())
                })
                .map(PathBuf::from)
        };
        let data = value("XDG_DATA_HOME=")
            .filter(|p| p.is_absolute())
            .or_else(|| value("HOME=").map(|home| home.join(".local/share")))
            .map(|root| root.join("dev.noullama.desktop"))
            .and_then(|path| std::fs::canonicalize(path).ok());
        let args = std::fs::read(root.join("cmdline"))
            .ok()?
            .split(|b| *b == 0)
            .filter_map(|s| std::str::from_utf8(s).ok().map(str::to_owned))
            .collect();
        Some(Process {
            pid,
            start,
            exe,
            data_dir: data,
            args,
        })
    }

    fn processes() -> Vec<Process> {
        std::fs::read_dir("/proc")
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| entry.file_name().to_str()?.parse().ok())
            .filter_map(read)
            .collect()
    }

    fn owned_runtime(process: &Process, root: &Path) -> bool {
        if process
            .exe
            .file_name()
            .is_none_or(|name| name != "llama-server")
        {
            return false;
        }
        let flag = |name: &str| {
            process
                .args
                .windows(2)
                .find(|args| args[0] == name)
                .map(|args| args[1].as_str())
        };
        let Some(alias) = flag("--alias").and_then(|a| a.strip_prefix("noullama-")) else {
            return false;
        };
        if uuid::Uuid::parse_str(alias).is_err() {
            return false;
        }
        let Some(key) = flag("--api-key-file").map(PathBuf::from) else {
            return false;
        };
        let Some(name) = key
            .file_name()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("runtime-"))
            .and_then(|s| s.strip_suffix(".key"))
        else {
            return false;
        };
        uuid::Uuid::parse_str(name).is_ok()
            && key
                .parent()
                .and_then(|path| std::fs::canonicalize(path).ok())
                .as_deref()
                == Some(root)
    }

    fn terminate(process: &Process, root: &Path, app_exe: Option<&Path>) -> std::io::Result<()> {
        // The pidfd pins the process identity, so a reused PID cannot receive
        // either signal. Recheck start time, executable, and scope after opening.
        let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, process.pid, 0) } as i32;
        if fd < 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::ESRCH) {
                return Ok(());
            }
            return Err(error);
        }
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let Some(current) = read(process.pid) else {
            return Ok(());
        };
        if current.start != process.start || current.exe != process.exe {
            return Ok(());
        }
        let owned = app_exe
            .map(|exe| current.exe == exe && current.data_dir.as_deref() == Some(root))
            .unwrap_or_else(|| owned_runtime(&current, root));
        if !owned {
            return Ok(());
        }
        let signal = |signal: libc::c_int| {
            if unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    fd.as_raw_fd(),
                    signal,
                    std::ptr::null::<libc::siginfo_t>(),
                    0,
                )
            } < 0
            {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ESRCH) {
                    return Err(error);
                }
            }
            Ok(())
        };
        let exited = |timeout| {
            let mut poll = libc::pollfd {
                fd: fd.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            (unsafe { libc::poll(&mut poll, 1, timeout) }) > 0
        };
        signal(libc::SIGTERM)?;
        if !exited(5000) {
            signal(libc::SIGKILL)?;
            if !exited(2000) {
                return Err(std::io::Error::other(
                    "A verified Noullama process did not exit",
                ));
            }
        }
        Ok(())
    }

    pub fn clean_older_instances(root: &Path) -> std::io::Result<()> {
        let me = read(std::process::id())
            .ok_or_else(|| std::io::Error::other("Cannot verify this Noullama process identity"))?;
        let candidates: Vec<_> = processes()
            .into_iter()
            .filter(|p| p.pid != me.pid && p.exe == me.exe && p.data_dir.as_deref() == Some(root))
            .collect();
        if candidates
            .iter()
            .any(|p| p.start > me.start || (p.start == me.start && p.pid > me.pid))
        {
            return Err(std::io::Error::other(
                "A newer Noullama launch is already starting",
            ));
        }
        for process in candidates {
            terminate(&process, root, Some(&me.exe))?;
        }
        Ok(())
    }

    pub fn clean_owned_runtimes(root: &Path) -> std::io::Result<()> {
        for process in processes()
            .into_iter()
            .filter(|p| p.pid != std::process::id() && owned_runtime(p, root))
        {
            terminate(&process, root, None)?;
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn only_runtime_with_both_unique_ownership_markers_is_eligible() {
            let root =
                std::env::temp_dir().join(format!("noullama-owner-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&root).unwrap();
            let root = std::fs::canonicalize(root).unwrap();
            let mut process = Process {
                pid: 1,
                start: 1,
                exe: PathBuf::from("/custom/llama-server"),
                data_dir: None,
                args: vec!["llama-server".into(), "--port".into(), "11435".into()],
            };
            assert!(!owned_runtime(&process, &root));
            process.args.extend([
                "--alias".into(),
                format!("noullama-{}", uuid::Uuid::new_v4()),
                "--api-key-file".into(),
                root.join(format!("runtime-{}.key", uuid::Uuid::new_v4()))
                    .to_string_lossy()
                    .into_owned(),
            ]);
            assert!(owned_runtime(&process, &root));
            process.exe = PathBuf::from("/usr/bin/ollaya");
            assert!(!owned_runtime(&process, &root));
            process.exe = PathBuf::from("/custom/llama-server");
            process.args.pop();
            process.args.push("/tmp/someone-elses.key".into());
            assert!(!owned_runtime(&process, &root));
            std::fs::remove_dir_all(root).unwrap();
        }
        #[test]
        fn process_identity_parser_handles_spaces_and_parentheses_in_names() {
            let stat = format!("123 (noullama (debug)) S {} 987 0", vec!["0"; 18].join(" "));
            assert_eq!(start_time(&stat), Some(987));
            assert_eq!(start_time("bad data"), None);
        }
    }
}
