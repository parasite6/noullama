import { Reveal } from './Reveal';
import { busyButton, useDelayedPending } from './busy-button';
import React, { useEffect, useRef, useState } from 'react';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { Select } from './Select';
import { CopyButton } from './CopyButton';
import { DownloadSimpleIcon as ArrowDownToLine } from '@phosphor-icons/react/dist/icons/DownloadSimple';
import { CheckIcon as Check } from '@phosphor-icons/react/dist/icons/Check';
import { CpuIcon as Cpu } from '@phosphor-icons/react/dist/icons/Cpu';
import { ArrowsClockwiseIcon as RefreshCw } from '@phosphor-icons/react/dist/icons/ArrowsClockwise';
import { XIcon as X } from '@phosphor-icons/react/dist/icons/X';
import { Help, SettingRow, SettingSection, StatusNote } from './SettingsLayout';

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
export function RuntimeManager({ currentPath, currentName, running, progress, onSelect, onRefresh, children, cpuPath, cpuName, cpuProgress, onSelectCpu, cpuChildren }: { currentPath: string; currentName: string; running: boolean; progress?: RuntimeDownload; onSelect: (runtime: LocalRuntime) => void; onRefresh: () => Promise<void>; children?: React.ReactNode; cpuPath: string; cpuName: string; cpuProgress?: RuntimeDownload; onSelectCpu: (runtime: LocalRuntime) => void; cpuChildren?: React.ReactNode }) {
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
  const visiblePending = useDelayedPending(pending);
  const scanning = visiblePending === 'scan';
  const buildRow = (runtime: LocalRuntime, chosen: string, use: string, onUse: (runtime: LocalRuntime) => void, onCopyError: (error: string) => void) => {
    const selected = isChosen(runtime, chosen);
    return <div className={`runtime-build-row ${selected ? 'selected' : ''}`} key={runtime.path}>
      <div className="runtime-build-text"><strong>{runtime.label}</strong><span className="identity-line"><code title={runtime.path}>{runtime.path}</code><CopyButton value={runtime.path} label="Copy runtime path" size={12} onError={onCopyError}/></span><small>{runtime.source}{runtime.backend ? ` / ${backendNames[runtime.backend] ?? runtime.backend}` : ''}</small></div>
      <div className="setting-control">{selected ? <span className="selected-badge"><Check size={15} weight="bold"/>Selected</span> : <button className="secondary compact" onClick={() => onUse(runtime)}>{use}</button>}</div>
    </div>;
  };
  return <><section className="panel runtime-manager" aria-label="GPU build">
    <div className="panel-heading"><span><Cpu size={16}/> llama.cpp</span><span className="muted">{scan?.platform ?? 'Local runtimes'}</span></div>
    <div className="settings-body">
      <Current kicker="Selected build" name={currentName} path={currentPath} copyLabel="Copy runtime path" onCopyError={setError}
        description="Used by loads on a GPU device." more="Automatic CPU selection uses this build when no CPU runtime is set."
        status={(!currentPath || running) && <>{!currentPath && <StatusNote tone="warn">No build selected. Choose one below, enter its path, or download one.</StatusNote>}{running && <StatusNote tone="info">Loaded models keep the build they started with. A new choice applies to the next model you load.</StatusNote>}</>}/>
      <SettingSection title="Installed builds" description="Builds found on this computer." more="Found in Noullama downloads, in the folders of the selected GPU and CPU builds, in the folder above the GPU build's folder, and on PATH."
        aside={<button className="secondary compact" {...busyButton(!!pending, !desktop, pending === 'scan')} onClick={() => void discover()}><RefreshCw size={14} className={scanning ? 'spin' : ''}/>{scanning ? 'Scanning…' : 'Scan local builds'}</button>}>
        {installed.length > 0 ? <div className="runtime-build-list">{installed.map(runtime => buildRow(runtime, currentPath, 'Use', onSelect, setError))}</div>
          : <StatusNote tone={scanning ? 'pending' : 'info'}>{scanning ? 'Checking local runtime locations…' : 'No builds found. Download one below or enter a custom path.'}</StatusNote>}
        {scan?.warnings.map(warning => <StatusNote tone="warn" key={warning}>{warning}</StatusNote>)}
      </SettingSection>
      <SettingSection title="Custom path">{children}</SettingSection>
      <SettingSection title="Download a build" description="The newest nightly for a backend, plus the latest stable release as a fallback." more="Downloads the newest nightly for the backend you pick, which becomes your runtime, plus the latest official stable release for the same backend as a fallback.">
        <SettingRow label="Nightly backend" description="Checking only reads release details from GitHub." wide
          control={<div className="file-input-row"><Select label="Nightly runtime backend" value={backend} disabled={downloading} busy={!!pending} onChange={value => { setBackend(value); setPlan(null); }} options={(scan?.backends ?? ['cpu']).map(backend => ({ value: backend, label: backendNames[backend] ?? backend }))}/><button className="secondary" {...busyButton(!!pending, !desktop || !scan || downloading, pending === 'releases')} onClick={() => void checkReleases()}><RefreshCw size={14} className={visiblePending === 'releases' ? 'spin' : ''}/>{visiblePending === 'releases' ? 'Checking GitHub…' : 'Check official releases'}</button></div>}/>
        <Reveal open={plan}>{plan && <div className="runtime-download-plan">{[['Nightly (default)', plan.nightly], ['Official stable (fallback)', plan.fallback]].map(([label, value]) => { const build = value as Build; return <div className="runtime-plan-build" key={String(label)}><strong>{String(label)} <span className="pill">{build.release} / {build.tag} / {backendNames[build.backend]}</span></strong>{build.assets.map(asset => <div className="runtime-asset" key={asset.name}><code>{asset.name}</code><span>{size(asset.size)}</span></div>)}</div>; })}<div className="runtime-download-confirm"><span className="hint">{size(plan.total_bytes)} in total. Checked against SHA-256 and extracted locally.</span><button className="primary compact" {...busyButton(!!pending, downloading, pending === 'download')} onClick={() => void download()}><ArrowDownToLine size={14}/>Download nightly + fallback</button></div></div>}</Reveal>
        <Reveal open={cpuProgress?.active}>{cpuProgress?.active && <StatusNote tone="info">One runtime download runs at a time. This one can start when the CPU runtime download below has finished.</StatusNote>}</Reveal>
        <InstallStatus progress={progress} label="Runtime download progress"/>
        <Reveal open={message}>{message && <div className="notice error" role="alert"><span>{message}</span><button aria-label="Dismiss runtime error" onClick={() => { setError(null); setDismissedError(progress?.error ?? null); }}><X size={16}/></button></div>}</Reveal>
      </SettingSection>
    </div>
  </section>
  <section className="panel runtime-manager cpu-runtime" aria-label="CPU runtime">
    <div className="panel-heading"><span><Cpu size={16}/> CPU runtime</span><span className="muted">For Device CPU</span></div>
    <div className="settings-body">
      <Current kicker="Selected CPU build" name={cpuName} path={cpuPath} copyLabel="Copy CPU runtime path" onCopyError={setCpuError}
        description="Used by loads on Device CPU, with GPU layers 0." more="Loads whose Device is CPU in Loading run this llama-server, with GPU layers 0. Loads on a GPU use the llama.cpp build above."
        status={!cpuPath && <StatusNote tone="warn">Choose a CPU build below, enter its path, or download the official one.</StatusNote>}/>
      <SettingSection title="CPU builds" description="CPU builds among the installed builds above.">
        {cpuBuilds.length > 0 ? <div className="runtime-build-list">{cpuBuilds.map(runtime => buildRow(runtime, cpuPath, 'Use for CPU', onSelectCpu, setCpuError))}</div>
          : <StatusNote tone={scanning ? 'pending' : 'info'}>{scanning ? 'Checking local runtime locations…' : 'No CPU builds found. Download the official one below or enter a path.'}</StatusNote>}
      </SettingSection>
      <SettingSection title="Custom path">{cpuChildren}</SettingSection>
      <SettingSection title="Download a build" description="The newest nightly CPU build, plus the latest stable one as a fallback." more="Downloads the newest nightly CPU build plus the latest official stable CPU build as a fallback, from the same official releases and with the same checks as above. Choose Use for CPU on the build you want, then save settings.">
        <SettingRow label="Official CPU build" description="Checked against SHA-256 and extracted locally." wide
          control={<button className="primary" {...busyButton(!!pending, !desktop || downloading, pending === 'cpu-download')} onClick={() => void downloadCpu()}><ArrowDownToLine size={14}/>{visiblePending === 'cpu-download' ? 'Starting…' : 'Download the official CPU build'}</button>}/>
        <Reveal open={progress?.active}>{progress?.active && <StatusNote tone="info">One runtime download runs at a time. This one can start when the download above has finished.</StatusNote>}</Reveal>
        <InstallStatus progress={cpuProgress} label="CPU runtime download progress"/>
        <Reveal open={cpuMessage}>{cpuMessage && <div className="notice error" role="alert"><span>{cpuMessage}</span><button aria-label="Dismiss CPU runtime error" onClick={() => { setCpuError(null); setDismissedCpuError(cpuProgress?.error ?? null); }}><X size={16}/></button></div>}</Reveal>
      </SettingSection>
    </div>
  </section></>;
}

// The build a panel uses now, at the top of the panel.
function Current({ kicker, name, path, copyLabel, onCopyError, description, more, status }: { kicker: string; name: string; path: string; copyLabel: string; onCopyError: (error: string) => void; description: string; more: string; status?: React.ReactNode }) {
  return <div className="runtime-current">
    <div className="load-identity">
      <div className="load-identity-icon"><Cpu size={20}/></div>
      <div className="model-identity"><small>{kicker}</small><span className="identity-line"><strong title={path || undefined}>{name}</strong></span>{path && <span className="identity-line"><code title={path}>{path}</code><CopyButton value={path} label={copyLabel} size={12} onError={onCopyError}/></span>}</div>
    </div>
    <Help text={description} more={more} topic={kicker}/>
    <Reveal open={status}>{status}</Reveal>
  </div>;
}
