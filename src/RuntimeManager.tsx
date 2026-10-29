import React, { useEffect, useState } from 'react';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { Select } from './Select';
import { ArrowDownToLine, Check, Cpu, RefreshCw, X } from 'lucide-react';

export type LocalRuntime = { path: string; label: string; backend: string | null; source: string };
type Scan = { platform: string; backends: string[]; local: LocalRuntime[]; warnings: string[] };
type Asset = { name: string; size: number };
type Build = { release: string; tag: string; backend: string; assets: Asset[] };
type Plan = { id: string; platform: string; nightly: Build; fallback: Build; total_bytes: number };
export type RuntimeDownload = { active: boolean; message: string; downloaded: number; total: number; error: string | null; installed: LocalRuntime[] };
const backendNames: Record<string, string> = { cpu: 'CPU', metal: 'Metal', vulkan: 'Vulkan', cuda12: 'NVIDIA CUDA 12', cuda13: 'NVIDIA CUDA 13' };
const size = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`;

export function RuntimeManager({ currentPath, running, progress, onSelect, onRefresh, children }: { currentPath: string; running: boolean; progress?: RuntimeDownload; onSelect: (runtime: LocalRuntime) => void; onRefresh: () => Promise<void>; children?: React.ReactNode }) {
  const desktop = isTauri();
  const [scan, setScan] = useState<Scan | null>(null);
  const [backend, setBackend] = useState('cpu');
  const [plan, setPlan] = useState<Plan | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dismissedError, setDismissedError] = useState<string | null>(null);
  async function discover() {
    setPending('scan'); setError(null);
    try {
      const next = await invoke<Scan>('discover_runtimes'); setScan(next);
      setBackend(current => next.backends.includes(current) ? current : next.backends[0]);
    } catch (error) { setError(String(error)); } finally { setPending(null); }
  }
  useEffect(() => { if (desktop) void discover(); }, []);
  useEffect(() => { if (desktop && progress?.installed.length && !progress.active) void discover(); }, [progress?.installed.length, progress?.active]);
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
  const installed = [...(scan?.local ?? []), ...(progress?.installed ?? [])].filter((item, index, all) => all.findIndex(other => other.path === item.path) === index);
  const message = error ?? (progress?.error !== dismissedError ? progress?.error : null);
  return <section className="panel runtime-manager">
    <div className="panel-heading"><span><Cpu size={16}/> llama.cpp</span><span className="muted">{scan?.platform ?? 'Local runtimes'}</span></div>
    <div className="settings-body">
      <div className="runtime-manager-heading"><div><h3>Installed builds</h3><p className="hint">Found in the configured path, next to the app, on PATH, and in Noullama downloads.</p></div><button className="secondary compact" disabled={!desktop || !!pending} onClick={() => void discover()}><RefreshCw size={14} className={pending === 'scan' ? 'spin' : ''}/>{pending === 'scan' ? 'Scanning…' : 'Scan local builds'}</button></div>
      {installed.length > 0 ? <div className="runtime-build-list">{installed.map(runtime => {
        const selected = runtime.path === currentPath || runtime.path.replace(/[\\/]llama-server(?:\.exe)?$/, '') === currentPath.replace(/[\\/]$/, '');
        return <div className="runtime-build-row" key={runtime.path}><div><strong>{runtime.label}</strong><code title={runtime.path}>{runtime.path}</code><small>{runtime.source}{runtime.backend ? ` / ${backendNames[runtime.backend] ?? runtime.backend}` : ''}</small></div><button className="secondary compact" disabled={running || selected} onClick={() => onSelect(runtime)}>{selected ? <><Check size={13}/>Selected</> : 'Use'}</button></div>;
      })}</div> : <p className="hint">{pending === 'scan' ? 'Checking local runtime locations…' : 'No builds found. Download one below or enter a custom path.'}</p>}
      {scan?.warnings.map(warning => <p className="hint" key={warning}>{warning}</p>)}
      {running && <p className="hint">Unload the model before switching builds.</p>}
      <div className="subsection"><h3>Custom path</h3>{children}</div>
      <div className="subsection"><h3>Download a build</h3><p className="hint">Downloads the newest nightly for the backend you pick, plus an official CPU build as a fallback. Checking only reads release details from GitHub.</p></div>
      <div className="runtime-release-controls"><div className="field"><span>Nightly backend</span><Select label="Nightly runtime backend" value={backend} disabled={!!pending || progress?.active} onChange={value => { setBackend(value); setPlan(null); }} options={(scan?.backends ?? ['cpu']).map(backend => ({ value: backend, label: backendNames[backend] ?? backend }))}/></div><button className="secondary compact" disabled={!desktop || !scan || !!pending || progress?.active} onClick={() => void checkReleases()}><RefreshCw size={14} className={pending === 'releases' ? 'spin' : ''}/>{pending === 'releases' ? 'Checking GitHub…' : 'Check official releases'}</button></div>
      {plan && <div className="runtime-download-plan">{[['Nightly', plan.nightly], ['Official fallback', plan.fallback]].map(([label, value]) => { const build = value as Build; return <div className="runtime-plan-build" key={String(label)}><strong>{String(label)} <span className="pill">{build.release} / {build.tag} / {backendNames[build.backend]}</span></strong>{build.assets.map(asset => <div className="runtime-asset" key={asset.name}><code>{asset.name}</code><span>{size(asset.size)}</span></div>)}</div>; })}<div className="runtime-download-confirm"><span className="hint">{size(plan.total_bytes)} in total. Checked against SHA-256 and extracted locally.</span><button className="primary compact" disabled={!!pending || progress?.active} onClick={() => void download()}><ArrowDownToLine size={14}/>Download nightly + fallback</button></div></div>}
      {progress?.message && <div className="runtime-install-status" role="status"><p>{progress.message}</p>{progress.active && <><div className="download-track" role="progressbar" aria-label="Runtime download progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress.total ? Math.min(100, Math.round(progress.downloaded / progress.total * 100)) : undefined}><i style={{ width: `${progress.total ? Math.min(100, progress.downloaded / progress.total * 100) : 0}%` }}/></div><small>{size(progress.downloaded)} / {size(progress.total)}</small></>}</div>}
      {message && <div className="notice error" role="alert"><span>{message}</span><button aria-label="Dismiss runtime error" onClick={() => { setError(null); setDismissedError(progress?.error ?? null); }}><X size={16}/></button></div>}
    </div>
  </section>;
}
