import React, { useEffect, useRef, useState } from 'react';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { Select } from './Select';
import { CopyButton } from './CopyButton';
import { ArrowDownToLine, Check, Cpu, RefreshCw, X } from 'lucide-react';

export type LocalRuntime = { path: string; label: string; backend: string | null; source: string };
type Scan = { platform: string; backends: string[]; local: LocalRuntime[]; warnings: string[] };
type Asset = { name: string; size: number };
type Build = { release: string; tag: string; backend: string; assets: Asset[] };
type Plan = { id: string; platform: string; nightly: Build; fallback: Build; total_bytes: number };
export type RuntimeDownload = { active: boolean; message: string; downloaded: number; total: number; error: string | null; installed: LocalRuntime[] };
// The backend picked for downloads is remembered on this machine.
const BACKEND_KEY = 'noullama.runtimeBackend';
const savedBackend = () => { try { return localStorage.getItem(BACKEND_KEY); } catch { return null; } };
const saveBackend = (value: string) => { try { localStorage.setItem(BACKEND_KEY, value); } catch { /* not kept; the default is used next time */ } };
const backendNames: Record<string, string> = { cpu: 'CPU', metal: 'Metal', vulkan: 'Vulkan', cuda12: 'NVIDIA CUDA 12', cuda13: 'NVIDIA CUDA 13' };
const size = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`;
// A build row is the chosen runtime when the path names it or its folder.
const isChosen = (runtime: LocalRuntime, path: string) => runtime.path === path || runtime.path.replace(/[\\/]llama-server(?:\.exe)?$/, '') === path.replace(/[\\/]$/, '');

// A runtime download's status line and progress bar.
function InstallStatus({ progress, label }: { progress?: RuntimeDownload; label: string }) {
  if (!progress?.message) return null;
  return <div className="runtime-install-status" role="status"><p>{progress.message}</p>{progress.active && <><div className="download-track" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress.total ? Math.min(100, Math.round(progress.downloaded / progress.total * 100)) : undefined}><i style={{ width: `${progress.total ? Math.min(100, progress.downloaded / progress.total * 100) : 0}%` }}/></div><small>{size(progress.downloaded)} / {size(progress.total)}</small></>}</div>;
}

// The Runtime tab's "llama.cpp" panel (the GPU runtime, `runtime_path`) and
// its "CPU runtime" panel (`cpu_runtime_path`, for loads on Device CPU).
// Both share the build scan; each download has its own progress, and only
// one runtime download runs at a time.
export function RuntimeManager({ currentPath, running, progress, onSelect, onRefresh, children, cpuPath, cpuName, cpuProgress, onSelectCpu, cpuChildren }: { currentPath: string; running: boolean; progress?: RuntimeDownload; onSelect: (runtime: LocalRuntime) => void; onRefresh: () => Promise<void>; children?: React.ReactNode; cpuPath: string; cpuName: string; cpuProgress?: RuntimeDownload; onSelectCpu: (runtime: LocalRuntime) => void; cpuChildren?: React.ReactNode }) {
  const desktop = isTauri();
  const [scan, setScan] = useState<Scan | null>(null);
  const [backend, setBackendState] = useState(() => savedBackend() ?? 'cpu');
  const setBackend = (value: string | ((current: string) => string)) => setBackendState(current => {
    const next = typeof value === 'function' ? value(current) : value;
    if (next !== current) saveBackend(next);
    return next;
  });
  const [plan, setPlan] = useState<Plan | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dismissedError, setDismissedError] = useState<string | null>(null);
  const [cpuError, setCpuError] = useState<string | null>(null);
  const [dismissedCpuError, setDismissedCpuError] = useState<string | null>(null);
  const downloading = !!progress?.active || !!cpuProgress?.active;
  async function discover() {
    setPending('scan'); setError(null);
    try {
      const next = await invoke<Scan>('discover_runtimes'); setScan(next);
      setBackendState(current => next.backends.includes(current) ? current : next.backends[0]);
    } catch (error) { setError(String(error)); } finally { setPending(null); }
  }
  useEffect(() => { if (desktop) void discover(); }, []);
  useEffect(() => { if (desktop && progress?.installed.length && !progress.active) void discover(); }, [progress?.installed.length, progress?.active]);
  useEffect(() => { if (desktop && cpuProgress?.installed.length && !cpuProgress.active) void discover(); }, [cpuProgress?.installed.length, cpuProgress?.active]);
  // When the CPU download finishes, fill in its nightly as the CPU runtime: an unsaved edit, like "Use for CPU".
  const cpuWasActive = useRef(false);
  useEffect(() => {
    if (cpuWasActive.current && cpuProgress && !cpuProgress.active && !cpuProgress.error && cpuProgress.installed.length) onSelectCpu(cpuProgress.installed[0]);
    cpuWasActive.current = !!cpuProgress?.active;
  }, [cpuProgress?.active]);
  // When a download finishes, select the nightly (installed first); the official build stays as the fallback.
  const wasActive = useRef(false);
  useEffect(() => {
    if (wasActive.current && progress && !progress.active && !progress.error && progress.installed.length) onSelect(progress.installed[0]);
    wasActive.current = !!progress?.active;
  }, [progress?.active]);
  async function checkReleases() {
    setPending('releases'); setError(null); setPlan(null);
    try { setPlan(await invoke<Plan>('runtime_releases', { backend })); }
    catch (error) { setError(String(error)); } finally { setPending(null); }
  }
  async function download() {
    if (!plan) return;
    setPending('download'); setError(null); setDismissedError(null);
    try { await invoke('download_runtimes', { planId: plan.id }); await onRefresh(); }
    catch (error) { setError(String(error)); } finally { setPending(null); }
  }
  async function downloadCpu() {
    setPending('cpu-download'); setCpuError(null); setDismissedCpuError(null);
    try { await invoke('download_cpu_runtime'); await onRefresh(); }
    catch (error) { setCpuError(String(error)); } finally { setPending(null); }
  }
  const installed = [...(scan?.local ?? []), ...(progress?.installed ?? []), ...(cpuProgress?.installed ?? [])].filter((item, index, all) => all.findIndex(other => other.path === item.path) === index);
  const message = error ?? (progress?.error !== dismissedError ? progress?.error : null);
  const cpuMessage = cpuError ?? (cpuProgress?.error !== dismissedCpuError ? cpuProgress?.error : null);
  const cpuBuilds = installed.filter(runtime => runtime.backend === 'cpu');
  return <><section className="panel runtime-manager">
    <div className="panel-heading"><span><Cpu size={16}/> llama.cpp</span><span className="muted">{scan?.platform ?? 'Local runtimes'}</span></div>
    <div className="settings-body">
      <div className="runtime-manager-heading"><div><h3>Installed builds</h3><p className="hint">Found in Noullama downloads, in the folders of the selected GPU and CPU builds, in the folder above the GPU build's folder, and on PATH.</p></div><button className="secondary compact" disabled={!desktop || !!pending} onClick={() => void discover()}><RefreshCw size={14} className={pending === 'scan' ? 'spin' : ''}/>{pending === 'scan' ? 'Scanning…' : 'Scan local builds'}</button></div>
      {installed.length > 0 ? <div className="runtime-build-list">{installed.map(runtime => {
        const selected = isChosen(runtime, currentPath);
        return <div className="runtime-build-row" key={runtime.path}><div><strong>{runtime.label}</strong><span className="identity-line"><code title={runtime.path}>{runtime.path}</code><CopyButton value={runtime.path} label="Copy runtime path" size={12} onError={setError}/></span><small>{runtime.source}{runtime.backend ? ` / ${backendNames[runtime.backend] ?? runtime.backend}` : ''}</small></div><button className="secondary compact" disabled={selected} onClick={() => onSelect(runtime)}>{selected ? <><Check size={13}/>Selected</> : 'Use'}</button></div>;
      })}</div> : <p className="hint">{pending === 'scan' ? 'Checking local runtime locations…' : 'No builds found. Download one below or enter a custom path.'}</p>}
      {scan?.warnings.map(warning => <p className="hint" key={warning}>{warning}</p>)}
      {running && <p className="hint">Loaded models keep the build they started with. A new choice applies to the next model you load.</p>}
      <div className="subsection"><h3>Custom path</h3>{children}</div>
      <div className="subsection"><h3>Download a build</h3><p className="hint">Downloads the newest nightly for the backend you pick, which becomes your runtime, plus the latest official stable release for the same backend as a fallback. Checking only reads release details from GitHub.</p></div>
      <div className="runtime-release-controls field"><span>Nightly backend</span><div className="file-input-row"><Select label="Nightly runtime backend" value={backend} disabled={!!pending || downloading} onChange={value => { setBackend(value); setPlan(null); }} options={(scan?.backends ?? ['cpu']).map(backend => ({ value: backend, label: backendNames[backend] ?? backend }))}/><button className="secondary" disabled={!desktop || !scan || !!pending || downloading} onClick={() => void checkReleases()}><RefreshCw size={14} className={pending === 'releases' ? 'spin' : ''}/>{pending === 'releases' ? 'Checking GitHub…' : 'Check official releases'}</button></div></div>
      {plan && <div className="runtime-download-plan">{[['Nightly (default)', plan.nightly], ['Official stable (fallback)', plan.fallback]].map(([label, value]) => { const build = value as Build; return <div className="runtime-plan-build" key={String(label)}><strong>{String(label)} <span className="pill">{build.release} / {build.tag} / {backendNames[build.backend]}</span></strong>{build.assets.map(asset => <div className="runtime-asset" key={asset.name}><code>{asset.name}</code><span>{size(asset.size)}</span></div>)}</div>; })}<div className="runtime-download-confirm"><span className="hint">{size(plan.total_bytes)} in total. Checked against SHA-256 and extracted locally.</span><button className="primary compact" disabled={!!pending || downloading} onClick={() => void download()}><ArrowDownToLine size={14}/>Download nightly + fallback</button></div></div>}
      {cpuProgress?.active && <p className="hint">One runtime download runs at a time. This one can start when the CPU runtime download below has finished.</p>}
      <InstallStatus progress={progress} label="Runtime download progress"/>
      {message && <div className="notice error" role="alert"><span>{message}</span><button aria-label="Dismiss runtime error" onClick={() => { setError(null); setDismissedError(progress?.error ?? null); }}><X size={16}/></button></div>}
    </div>
  </section>
  <section className="panel runtime-manager cpu-runtime" aria-label="CPU runtime">
    <div className="panel-heading"><span><Cpu size={16}/> CPU runtime</span><span className="muted">For Device CPU</span></div>
    <div className="settings-body">
      <p className="hint">Loads whose Device is CPU in Local API run this llama-server, with GPU layers 0. Loads on a GPU use the llama.cpp build above.</p>
      <div className="runtime-build-row cpu-runtime-current"><div><strong>{cpuName}</strong>{cpuPath ? <span className="identity-line"><code title={cpuPath}>{cpuPath}</code><CopyButton value={cpuPath} label="Copy CPU runtime path" size={12} onError={setCpuError}/></span> : <small>Choose a CPU build below, enter its path, or download the official one.</small>}</div></div>
      <div className="subsection"><h3>CPU builds</h3></div>
      {cpuBuilds.length > 0 ? <div className="runtime-build-list">{cpuBuilds.map(runtime => {
        const selected = isChosen(runtime, cpuPath);
        return <div className="runtime-build-row" key={runtime.path}><div><strong>{runtime.label}</strong><span className="identity-line"><code title={runtime.path}>{runtime.path}</code><CopyButton value={runtime.path} label="Copy runtime path" size={12} onError={setCpuError}/></span><small>{runtime.source} / {backendNames.cpu}</small></div><button className="secondary compact" disabled={selected} onClick={() => onSelectCpu(runtime)}>{selected ? <><Check size={13}/>Selected</> : 'Use for CPU'}</button></div>;
      })}</div> : <p className="hint">{pending === 'scan' ? 'Checking local runtime locations…' : 'No CPU builds found. Download the official one below or enter a path.'}</p>}
      <div className="subsection"><h3>Path</h3>{cpuChildren}</div>
      <div className="subsection"><h3>Download the official CPU build</h3><p className="hint">Downloads the newest nightly CPU build plus the latest official stable CPU build as a fallback, from the same official releases and with the same checks as above. The nightly is filled in as the CPU runtime; save to use it.</p></div>
      <div className="runtime-download-confirm"><span className="hint">{progress?.active ? 'One runtime download runs at a time. This one can start when the download above has finished.' : 'Checked against SHA-256 and extracted locally.'}</span><button className="primary compact" disabled={!desktop || !!pending || downloading} onClick={() => void downloadCpu()}><ArrowDownToLine size={14}/>{pending === 'cpu-download' ? 'Starting…' : 'Download the official CPU build'}</button></div>
      <InstallStatus progress={cpuProgress} label="CPU runtime download progress"/>
      {cpuMessage && <div className="notice error" role="alert"><span>{cpuMessage}</span><button aria-label="Dismiss CPU runtime error" onClick={() => { setCpuError(null); setDismissedCpuError(cpuProgress?.error ?? null); }}><X size={16}/></button></div>}
    </div>
  </section></>;
}
