import React, { useEffect, useState } from 'react';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { ArrowDownToLine, Check, Cpu, RefreshCw, X } from 'lucide-react';

export type LocalRuntime = { path: string; label: string; backend: string | null; source: string };
type Scan = { platform: string; backends: string[]; local: LocalRuntime[]; warnings: string[] };
type Asset = { name: string; size: number };
type Build = { release: string; tag: string; backend: string; assets: Asset[] };
type Plan = { id: string; platform: string; nightly: Build; fallback: Build; total_bytes: number };
export type RuntimeDownload = { active: boolean; message: string; downloaded: number; total: number; error: string | null; installed: LocalRuntime[] };
const backendNames: Record<string, string> = { cpu: 'CPU', metal: 'Metal', vulkan: 'Vulkan', cuda12: 'NVIDIA CUDA 12', cuda13: 'NVIDIA CUDA 13' };
const size = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`;

export function RuntimeManager({ currentPath, running, progress, onSelect, onRefresh }: { currentPath: string; running: boolean; progress?: RuntimeDownload; onSelect: (runtime: LocalRuntime) => void; onRefresh: () => Promise<void> }) {
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
    <div className="panel-heading"><span><Cpu size={16}/> llama.cpp builds</span><span className="muted">{scan?.platform ?? 'Local runtimes'}</span></div>
    <div className="settings-body">
      <div className="runtime-manager-heading"><div><strong>Available locally</strong><p className="hint">Configured path, nearby builds, PATH, and Noullama downloads. Existing builds are kept in place.</p></div><button className="secondary compact" disabled={!desktop || !!pending} onClick={() => void discover()}><RefreshCw size={14} className={pending === 'scan' ? 'spin' : ''}/>{pending === 'scan' ? 'Scanning…' : 'Scan local builds'}</button></div>
      {installed.length > 0 ? <div className="runtime-build-list">{installed.map(runtime => {
        const selected = runtime.path === currentPath || runtime.path.replace(/[\\/]llama-server(?:\.exe)?$/, '') === currentPath.replace(/[\\/]$/, '');
        return <div className="runtime-build-row" key={runtime.path}><div><strong>{runtime.label}</strong><code title={runtime.path}>{runtime.path}</code><small>{runtime.source}{runtime.backend ? ` · ${backendNames[runtime.backend] ?? runtime.backend}` : ''}</small></div><button className="secondary compact" disabled={running || selected} onClick={() => onSelect(runtime)}>{selected ? <><Check size={13}/>Selected</> : 'Use'}</button></div>;
      })}</div> : <p className="hint">{pending === 'scan' ? 'Checking local runtime locations…' : 'No local builds found in the checked locations. You can still enter a custom path below.'}</p>}
      {scan?.warnings.map(warning => <p className="hint" key={warning}>{warning}</p>)}
      <div className="runtime-release-controls"><label>Nightly backend<select aria-label="Nightly runtime backend" value={backend} disabled={!!pending || progress?.active} onChange={event => { setBackend(event.target.value); setPlan(null); }}>{(scan?.backends ?? ['cpu']).map(backend => <option key={backend} value={backend}>{backendNames[backend] ?? backend}</option>)}</select></label><button className="secondary compact" disabled={!desktop || !scan || !!pending || progress?.active} onClick={() => void checkReleases()}><RefreshCw size={14} className={pending === 'releases' ? 'spin' : ''}/>{pending === 'releases' ? 'Checking GitHub…' : 'Check official releases'}</button></div>
      <p className="hint">Release checks fetch metadata only. Downloads start when you click the download button. The official fallback uses CPU (Metal on macOS) so it can also work without GPU drivers.</p>
      {plan && <div className="runtime-download-plan">{[['Nightly', plan.nightly], ['Official fallback', plan.fallback]].map(([label, value]) => { const build = value as Build; return <div className="runtime-plan-build" key={String(label)}><strong>{String(label)} <span className="pill">{build.release} · {build.tag} · {backendNames[build.backend]}</span></strong>{build.assets.map(asset => <div className="runtime-asset" key={asset.name}><code>{asset.name}</code><span>{size(asset.size)}</span></div>)}</div>; })}<div className="runtime-download-confirm"><span className="hint">{size(plan.total_bytes)} total · SHA-256 verified · extracted locally</span><button className="primary compact" disabled={!!pending || progress?.active} onClick={() => void download()}><ArrowDownToLine size={14}/>Download nightly + fallback</button></div></div>}
      {progress?.message && <div className="runtime-install-status" role="status"><p>{progress.message}</p>{progress.active && <><div className="download-track" role="progressbar" aria-label="Runtime download progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress.total ? Math.min(100, Math.round(progress.downloaded / progress.total * 100)) : undefined}><i style={{ width: `${progress.total ? Math.min(100, progress.downloaded / progress.total * 100) : 0}%` }}/></div><small>{size(progress.downloaded)} / {size(progress.total)}</small></>}</div>}
      {message && <div className="notice error" role="alert"><span>{message}</span><button aria-label="Dismiss runtime error" onClick={() => { setError(null); setDismissedError(progress?.error ?? null); }}><X size={16}/></button></div>}
      {running && <p className="hint">Stop the loaded model before selecting another runtime.</p>}
    </div>
  </section>;
}
