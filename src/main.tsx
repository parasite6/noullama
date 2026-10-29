import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { Toggle } from './Toggle';
import { ModelDetails, type MemoryUse } from './ModelDetails';
import { ImageAttachments, MAX_TOTAL_BYTES, readImages, type Attachment } from './ImageAttachments';
import { Activity, AlertCircle, ArrowDownToLine, ArrowRight, BookOpen, Check, ChevronDown, ChevronRight, ChevronUp, CircleCheck, Code2, Copy, Cpu, Download, ExternalLink, FileJson, FlaskConical, FolderOpen, Gauge, Layers3, Mail, MemoryStick, Microchip, Network, Play, Plus, RefreshCw, Route, Search, Settings2, Shield, ShieldCheck, Square, Terminal, Trash2, Save, Power, X } from 'lucide-react';
import { buildRequest, convertState, presets } from './presets';
import { QuitDialog } from './QuitDialog';
import { ContextMenu } from './ContextMenu';
import { NumberField } from './NumberField';
import { runtimeStatus } from './runtime-state';
import { modelStatus } from './model-state';
import { JsonEditor } from './JsonEditor';
import { DownloadPanel, type DownloadState } from './DownloadPanel';
import { AuthorLogo, authorFromPath } from './AuthorLogo';
import { HuggingFaceLogo } from './HuggingFaceLogo';
import { RuntimeManager, type RuntimeDownload } from './RuntimeManager';
import { visibleBanner } from './ui-state';
import { containScrollChaining } from './scroll-contain';
import '@fontsource-variable/geist';
import '@fontsource-variable/geist-mono';
import './styles.css';

const logoUrl = new URL('../logo.png', import.meta.url).href;

type Settings = { runtime_path: string; model_path: string; mmproj_path: string; context_size: number; gpu_layers: number; api_port: number; runtime_port: number; hf_path: string; download_dir: string; library_path: string };

type Snapshot = { settings: Settings; running: boolean; ready: boolean; api_port: number; api_error: string | null; api_notice: string | null; load_error: string | null; runtime_download: RuntimeDownload; quit_confirmation: boolean; startup_warning: string | null; logs: string[]; download: DownloadState; data_dir: string; download_dir: string; loaded_model: { path: string; device: string; runtime_path: string; projector: boolean } | null; hardware: Hardware | null; fit: Fit | null };
type Hardware = { ram_bytes: number; vram_bytes: number | null; gpu: string | null; cpu_cores: number };
type Fit = { level: number; label: string; detail: string; model_bytes: number; required_bytes: number };
type SavedPreset = { id: string; name: string; questions: unknown; state: string; state_mode: 'text' | 'json' };
type Model = { id: string; downloads: number; likes: number; tags: string[]; fit: Fit | null };
type ModelFile = { name: string; size?: number; projector: boolean; fit: Fit | null };
type Inspection = { repo: string; files: ModelFile[]; sha: string };
type Session = { authenticated: boolean; account: string | null; cli_available: boolean; token_present: boolean };
type Result = { answers?: Record<string, { type?: string; choice?: string; score?: number; noul?: number; confidence?: number; probabilities?: Record<string, number> }>; usage?: { input_tokens: number; output_tokens: number }; [key: string]: unknown };
const defaultSettings: Settings = { runtime_path: '', model_path: '', mmproj_path: '', context_size: 8192, gpu_layers: 99, api_port: 11435, runtime_port: 11436, hf_path: 'hf', download_dir: '', library_path: '' };
const icons = [Route, ShieldCheck, Shield, Mail, Network, Plus];
const pages: Record<string, [string, string]> = {
  playground: ['Playground', 'Ask typed questions about a piece of content.'],
  models: ['Models', 'GGUF files on this machine.'],
  hub: ['Hub', 'Find and download decision models from Hugging Face.'],
  runtime: ['Runtime', 'llama.cpp builds, model settings and ports.'],
  api: ['Local API', 'Call the loaded model from your own code.'],
};
const pretty = (v: unknown) => JSON.stringify(v, null, 2);
const errorText = (e: unknown) => e instanceof Error ? e.message : String(e);
const gb = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(1)} GB`;
const basename = (path: string) => path.split(/[\\/]/).pop() || 'No model loaded';

function FileFit({ fit }: { fit: Fit | null }) {
  if (!fit) return <span className="fit-unavailable" title="A file size and detected system memory are needed for an estimate.">Fit unavailable</span>;
  return <span className={`fit-badge fit-level-${fit.level}`} title={`Estimated memory fit, not model/runtime compatibility. ${fit.detail}`}>
    <span className="fit-bars" aria-hidden="true">{[1, 2, 3, 4].map(n => <i key={n} className={n <= fit.level ? 'on' : ''}/>)}</span>{fit.label}
  </span>;
}

function App() {
  const [tab, setTab] = useState('playground');
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [dirty, setDirty] = useState(false);
  const [preset, setPreset] = useState('triage');
  const [savedPresets, setSavedPresets] = useState<SavedPreset[]>([]);
  const [presetName, setPresetName] = useState('');
  const [dismissedApi, setDismissedApi] = useState<string | null>(null);
  const [dismissedApiNotice, setDismissedApiNotice] = useState(false);
  const [dismissedLoadError, setDismissedLoadError] = useState<string | null>(null);
  const [dismissedStartup, setDismissedStartup] = useState(false);
  const [previewDismissed, setPreviewDismissed] = useState(false);
  const [downloadAnchor, setDownloadAnchor] = useState<{repo:string;file:string} | null>(null);
  const [questions, setQuestions] = useState(pretty(presets[0].questions));
  const [content, setContent] = useState(presets[0].state);
  const [stateMode, setStateMode] = useState<'text' | 'json'>('text');
  const [images, setImages] = useState<Attachment[]>([]);
  const [dropping, setDropping] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [memory, setMemory] = useState<MemoryUse | null>(null);
  const [stateMemory, setStateMemory] = useState<{ json: string; text: string } | null>(null);
  const [downloadDir, setDownloadDir] = useState('');
  const [result, setResult] = useState<Result | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [resultTab, setResultTab] = useState('decisions');
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ text: string; error: boolean } | null>(null);
  const [showLogs, setShowLogs] = useState(false);
  const [query, setQuery] = useState('');
  const [searched, setSearched] = useState('');
  const [toastDismissed, setToastDismissed] = useState('');
  const [toastCollapsed, setToastCollapsed] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [scanTick, setScanTick] = useState(0);
  const [localModels, setLocalModels] = useState<{ path: string; name: string; size: number }[]>([]);
  const [scanWarnings, setScanWarnings] = useState<string[]>([]);
  const [session, setSession] = useState<Session | null>(null);
  const [token, setToken] = useState('');
  const [version, setVersion] = useState('');
  const [copied, setCopied] = useState('');
  const desktop = isTauri();

  useEffect(() => { window.scrollTo({ top: 0, behavior: 'instant' }); }, [tab]);
  useEffect(containScrollChaining, []);
  useEffect(() => {
    const stay = (event: DragEvent) => { if (event.dataTransfer?.types.includes('Files')) event.preventDefault(); };
    window.addEventListener('dragover', stay);
    window.addEventListener('drop', stay);
    return () => { window.removeEventListener('dragover', stay); window.removeEventListener('drop', stay); };
  }, []);
  useEffect(() => {
    if (!detailsOpen || !desktop) return;
    let live = true;
    const read = () => invoke<MemoryUse>('model_memory').then(value => { if (live) setMemory(value); }).catch(() => {});
    read();
    const timer = setInterval(read, 2000);
    return () => { live = false; clearInterval(timer); };
  }, [detailsOpen]);

  useEffect(() => { if (!desktop || tab !== 'models') return; const timer = setInterval(() => setScanTick(t => t + 1), 5000); return () => clearInterval(timer); }, [tab]);
  async function refresh() { if (desktop) setSnapshot(await invoke<Snapshot>('snapshot')); }
  useEffect(() => {
    if (!desktop) return;
    invoke<SavedPreset[]>('list_presets').then(setSavedPresets).catch(e => setNotice({ text: errorText(e), error: true }));
    refresh().catch(e => setNotice({ text: errorText(e), error: true }));
    const timer = setInterval(() => { refresh().catch(() => {}); }, 1200);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => { if (snapshot && !dirty) setSettings(snapshot.settings); }, [snapshot?.settings.runtime_path, snapshot?.settings.model_path, snapshot?.settings.mmproj_path, snapshot?.settings.context_size, snapshot?.settings.gpu_layers, snapshot?.settings.api_port, snapshot?.settings.runtime_port, snapshot?.settings.hf_path, snapshot?.settings.download_dir, snapshot?.settings.library_path, dirty]);
  useEffect(() => { if (snapshot) setDownloadDir(snapshot.settings.download_dir); }, [snapshot?.settings.download_dir]);
  useEffect(() => { if (desktop && tab === 'hub' && !session) invoke<Session>('hf_session').then(setSession).catch(e => setNotice({ text: errorText(e), error: true })); }, [tab]);
  useEffect(() => { if (desktop && tab !== 'hub' && tab !== 'api') invoke<{ models: typeof localModels; warnings: string[] }>('local_models').then(scan => { setLocalModels(scan.models); setScanWarnings(scan.warnings); }).catch(e => setNotice({ text: errorText(e), error: true })); }, [tab, snapshot?.download.path, scanTick, snapshot?.download_dir, settings.model_path]);
  async function action(name: string, task: () => Promise<void>) {
    if (!desktop) { setNotice({ text: 'Launch the desktop app with npm run desktop to use the local backend.', error: true }); return; }
    setBusy(name); setNotice(null);
    try { await task(); await refresh(); } catch (e) { setNotice({ text: errorText(e), error: true }); } finally { setBusy(null); }
  }
  function edit<K extends keyof Settings>(key: K, value: Settings[K]) { setSettings(s => ({ ...s, [key]: value })); setDirty(true); }
  const isProjector = (path: string) => basename(path).toLowerCase().includes('mmproj');
  const folder = (path: string) => path.replace(/[\\/][^\\/]*$/, '');
  // A projector belongs to the model in the same folder; switching models drops one that does not.
  function selectModel(path: string) {
    edit('model_path', path);
    if (settings.mmproj_path && folder(settings.mmproj_path) !== folder(path)) edit('mmproj_path', '');
  }
  function addImages(files: File[]) {
    if (!files.length) return;
    readImages(files).then(added => {
      const total = [...images, ...added].reduce((sum, image) => sum + image.bytes, 0);
      if (total > MAX_TOTAL_BYTES) throw new Error('These images add up to more than 60 MB. Remove some or use smaller files.');
      setImages(current => [...current, ...added]); setResult(null);
    }).catch(e => setNotice({ text: errorText(e), error: true }));
  }
  async function save() { await invoke('save_settings', { settings }); setDirty(false); }
  async function run() {
    await action('decide', async () => {
      const payload = { ...buildRequest(questions, content, stateMode), ...(visionLoaded && images.length ? { images: images.map(image => image.url) } : {}) };
      const start = performance.now();
      const response = await invoke<Result>('decide', { payload });
      setElapsed(performance.now() - start); setResult(response);
    });
  }
  function selectPreset(id: string) {
    const saved = savedPresets.find(p => p.id === id);
    if (saved) {
      setPreset(saved.id); setQuestions(pretty(saved.questions)); setContent(saved.state); setStateMode(saved.state_mode); setPresetName(saved.name);
    } else {
      const p = presets.find(p => p.id === id)!;
      setPreset(id); setQuestions(pretty(p.questions)); setContent(p.state); setStateMode('text'); setPresetName('');
    }
    setStateMemory(null); setResult(null);
  }
  async function saveCustom() {
    await action('preset', async () => {
      const request = buildRequest(questions, content, stateMode);
      const next = await invoke<SavedPreset[]>('save_preset', { preset: { id: preset.startsWith('saved-') ? preset : '', name: presetName, questions: request.questions, state: content, state_mode: stateMode } });
      setSavedPresets(next);
      const saved = next.find(p => p.name.toLowerCase() === presetName.trim().toLowerCase());
      if (saved) { setPreset(saved.id); setPresetName(saved.name); }
      setNotice({ text: 'Custom preset saved.', error: false });
    });
  }
  async function trashPreset(id: string) {
    await action('preset', async () => {
      setSavedPresets(await invoke<SavedPreset[]>('delete_preset', { id }));
      if (preset === id) { setPreset('custom'); setPresetName(''); }
    });
  }
  async function browseFile(field: 'model_path' | 'mmproj_path') {
    await action('browse', async () => {
      const path = await invoke<string | null>('pick_model_file');
      if (path && field === 'model_path') selectModel(path);
      else if (path) edit(field, path);
    });
  }
  async function revealFile(path: string) {
    await action('reveal', async () => { await invoke('reveal_model_file', { path }); });
  }
  // Side by side, the detail pane scrolls on its own; stacked, bring it into view.
  useEffect(() => {
    const detail = document.getElementById('hub-detail');
    if (!inspection || !detail) return;
    if (getComputedStyle(detail).overflowY === 'auto') detail.scrollTo({ top: 0, behavior: 'smooth' });
    else detail.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [inspection]);
  async function openRepo(id: string) {
    await action('inspect', async () => { setInspection(await invoke('inspect_model', { contextSize: settings.context_size, gpuLayers: settings.gpu_layers, repo: id })); });
  }
  async function cancelDownload() { await action('cancel', async () => { await invoke('cancel_download'); }); }
  async function startDownload(repo: string, file: string) {
    setDownloadAnchor({ repo, file }); setToastDismissed(''); setToastCollapsed(false);
    await action('download', async () => {
      try { await invoke('download_model', { repo, filename: file }); }
      catch (error) { setDownloadAnchor(null); throw error; }
    });
  }
  function switchStateMode(next: 'text' | 'json') {
    try {
      const converted = convertState(content, stateMode, next, stateMemory?.json, stateMemory?.text);
      if (stateMode === 'json' && next === 'text') setStateMemory({ json: content, text: converted });
      setContent(converted); setStateMode(next);
    } catch (e) { setNotice({ text: errorText(e), error: true }); }
  }
  async function loadSelected() {
    setDismissedLoadError(null);
    await action('start', async () => {
      if (snapshot?.running) { await invoke('stop_runtime'); await refresh(); }
      await save();
      await invoke('start_runtime');
    });
  }
  async function saveDownloadDirectory() {
    await action('directory', async () => {
      if (!snapshot) return;
      await invoke('save_settings', { settings: { ...snapshot.settings, download_dir: downloadDir } });
      setSettings(current => ({ ...current, download_dir: downloadDir }));
      setNotice({ text: 'Download directory saved. Existing files stay in place.', error: false });
    });
  }
  async function copy(value: string, id: string) {
    try { await navigator.clipboard.writeText(value); setCopied(id); setTimeout(() => setCopied(''), 1800); }
    catch (e) { setNotice({ text: `Could not copy: ${errorText(e)}`, error: true }); }
  }
  const banner = visibleBanner(notice ?? (!dismissedApiNotice && snapshot?.api_notice ? { text: snapshot.api_notice, error: false } : null) ?? (!dismissedStartup && snapshot?.startup_warning ? { text: snapshot.startup_warning, error: true } : null), snapshot?.api_error, dismissedApi, desktop, previewDismissed);
  const ready = snapshot?.ready ?? false;
  const running = snapshot?.running ?? false;
  const model = modelStatus(settings.model_path, snapshot?.loaded_model ?? null, running, ready);
  const runtimeDirty = snapshot ? (['runtime_path', 'model_path', 'mmproj_path', 'context_size', 'gpu_layers', 'runtime_port', 'library_path'] as const).some(key => settings[key] !== snapshot.settings[key]) : false;
  const fit = snapshot?.fit ?? null;
  const visionLoaded = ready && !!snapshot?.loaded_model?.projector;
  const projector = settings.mmproj_path || localModels.find(file => isProjector(file.path) && folder(file.path) === folder(settings.model_path))?.path || '';
  const modelFiles = localModels.filter(file => !isProjector(file.path));
  const loadError = !running && snapshot?.load_error && snapshot.load_error !== dismissedLoadError ? snapshot.load_error : null;
  const hardware = snapshot?.hardware ?? null;
  const chosenRuntime = runtimeStatus(settings.runtime_path, running ? snapshot?.loaded_model?.runtime_path ?? snapshot?.settings.runtime_path ?? null : null, running, snapshot?.data_dir ?? '');
  const apiBase = `http://127.0.0.1:${snapshot?.api_port ?? 11435}/v1`;
  let questionError = '';
  try { buildRequest(questions, '', 'text'); } catch (e) { questionError = errorText(e); }
  let jsonError = '';
  try { buildRequest(questions, content, stateMode); } catch (e) { jsonError = errorText(e); }
  const requestJson = (() => { try { return pretty(buildRequest(questions, content, stateMode)); } catch { return '{}'; } })();

  const [pageTitle, pageSubtitle] = pages[tab];
  const repoLike = /^[\w.-]+\/[\w.-]+$/.test(query.trim()) && !models.some(m => m.id === query.trim());
  const download = snapshot?.download;
  const showDownloadToast = !!download && (download.active || (!!download.path && !download.error && !download.cancelled)) && download.file !== '' && !(tab === 'hub' && inspection?.repo === download.repo) && toastDismissed !== `${download.repo}/${download.file}` && !showLogs;
  const hubRow = (group: string, id: string, meta: React.ReactNode, fitInfo: Fit | null, fitTitle?: string) => <button key={`${group}:${id}`} className={`hub-row ${inspection?.repo === id ? 'selected' : ''}`} onClick={() => openRepo(id)}><AuthorLogo author={id.split('/')[0]} size={34}/><span className="hub-row-name"><strong>{id.split('/').slice(1).join('/') || id}</strong><small>{id.split('/')[0]}</small></span><span className="hub-row-meta">{meta}</span>{fitInfo && <span className={`fit-dot fit-level-${fitInfo.level}`} title={fitTitle ?? fitInfo.detail}><span className="fit-bars" aria-hidden="true">{[1, 2, 3, 4].map(n => <i key={n} className={n <= fitInfo.level ? 'on' : ''}/>)}</span><span className="sr-only">{fitInfo.label}</span></span>}</button>;

  return <div className={`app-shell ${showDownloadToast ? toastCollapsed ? 'has-toast collapsed-toast' : 'has-toast' : ''}`}>
    <aside className="sidebar">
      <div className="brand"><img className="brand-logo" src={logoUrl} width={28} height={28} alt=""/><span>noullama</span><span className="brand-tag">Beta</span></div>
      <nav>{[
        ['playground', FlaskConical, 'Playground'], ['models', Layers3, 'Models'], ['hub', Network, 'Hub'], ['runtime', Cpu, 'Runtime'], ['api', Code2, 'Local API'],
      ].map(([id, Icon, title]) => { const I = Icon as typeof Cpu; return <button key={String(id)} className={`nav-item ${tab === id ? 'active' : ''}`} aria-current={tab === id ? 'page' : undefined} title={String(title)} onClick={() => setTab(String(id))}><I size={18}/><span>{String(title)}</span></button>; })}</nav>
      <div className="sidebar-bottom">
        <button className="nav-item" title="Runtime logs" aria-label="Runtime logs" aria-pressed={showLogs} onClick={() => setShowLogs(v => !v)}><Terminal size={18}/><span>Runtime logs</span></button>
        <button className="nav-item" title="Quit Noullama" aria-label="Quit Noullama" disabled={!desktop || busy === 'quit'} onClick={() => action('quit-prompt', async () => { await invoke('request_quit'); })}><Power size={18}/><span>Quit Noullama</span></button>
        <div className="runtime-card-wrap" onMouseEnter={() => setDetailsOpen(true)} onMouseLeave={() => setDetailsOpen(false)} onFocus={() => setDetailsOpen(true)} onBlur={() => setDetailsOpen(false)}>
          <button className="runtime-card" onClick={() => setTab('runtime')} aria-label={`Selected llama.cpp: ${chosenRuntime.source} ${chosenRuntime.name}${chosenRuntime.inUse ? ', in use' : ', not running'}`}><span className={`runtime-card-icon ${ready ? 'ready' : running ? 'loading' : ''}`}><Cpu size={16}/></span><span><strong>{ready ? 'Model ready' : running ? 'Loading model' : 'Runtime offline'}</strong><small>llama.cpp {chosenRuntime.source} · {chosenRuntime.name}</small>{chosenRuntime.pending && <small className="runtime-pending">Next selection not applied</small>}</span></button>
          {detailsOpen && <ModelDetails
            name={basename(snapshot?.loaded_model?.path || settings.model_path || '') }
            status={ready ? 'Ready' : running ? 'Loading' : model.selected ? 'Not loaded' : 'No model selected'}
            path={snapshot?.loaded_model?.path || settings.model_path}
            device={running ? snapshot?.loaded_model?.device ?? null : null}
            projector={running ? (snapshot?.loaded_model?.projector ? basename(snapshot.settings.mmproj_path) : null) : (settings.mmproj_path ? basename(settings.mmproj_path) : null)}
            contextSize={(running ? snapshot?.settings.context_size : settings.context_size) ?? settings.context_size}
            gpuLayers={(running ? snapshot?.settings.gpu_layers : settings.gpu_layers) ?? settings.gpu_layers}
            runtime={`${chosenRuntime.source} / ${chosenRuntime.name}${chosenRuntime.pending ? ' (next selection not applied)' : ''}`}
            endpoint={ready ? apiBase : null}
            memory={desktop ? memory : null}
            estimate={fit?.required_bytes ?? null}
            gpuName={hardware?.gpu ?? null}/>}
        </div>
      </div>
    </aside>
    <main>
      <div className={`page-content page-${tab}`}>
        <header className="page-heading">
          <div><h1>{pageTitle}</h1><p>{pageSubtitle}</p></div>
          {hardware && <div className="stat-chips" aria-label="This machine">{hardware.vram_bytes ? <span className="stat-chip" title={hardware.gpu ?? 'Graphics memory'}><Gauge size={14}/><strong>{gb(hardware.vram_bytes)}</strong>VRAM</span> : <span className="stat-chip"><Gauge size={14}/>No GPU memory</span>}{hardware.ram_bytes > 0 && <span className="stat-chip"><MemoryStick size={14}/><strong>{gb(hardware.ram_bytes)}</strong>RAM</span>}{hardware.cpu_cores > 0 && <span className="stat-chip"><Microchip size={14}/><strong>{hardware.cpu_cores}</strong>CPU threads</span>}</div>}
        </header>
        {banner && <div className={`notice ${banner.error ? 'error' : ''}`} role={banner.error ? 'alert' : 'status'}><span className="notice-text">{banner.error && <AlertCircle size={16}/>}<span>{banner.text}</span></span><button aria-label="Dismiss notification" onClick={() => { setNotice(null); setDismissedStartup(true); setDismissedApiNotice(true); setDismissedApi(snapshot?.api_error ?? null); setPreviewDismissed(true); }}><X size={16}/></button></div>}
        <section className="model-strip" aria-label="Model status">
          <div className="model-strip-icon">{authorFromPath(model.loaded || model.selected || '') ? <AuthorLogo author={authorFromPath(model.loaded || model.selected || '')} size={40}/> : <Cpu size={20}/>}</div>
          <div className="model-identity"><small>{running ? ready ? 'Loaded model' : 'Loading model' : 'Selected model'}</small><strong title={model.loaded || model.selected}>{model.loaded ? basename(model.loaded) : model.selected ? basename(model.selected) : 'No model selected'}</strong>{model.device && <small>{model.device}</small>}{model.switching && <small className="pending-model" title={model.selected}>Next: {basename(model.selected)}</small>}</div>
          <span className={`chip ${ready ? 'success' : ''}`}><span className={`status-dot ${ready ? 'green' : running ? 'amber' : ''}`}/>{busy === 'start' ? 'Starting…' : model.label}</span>
          {projector && <Toggle label="Vision" title={settings.mmproj_path ? `Loads ${basename(settings.mmproj_path)}` : `Load ${basename(projector)} with this model`} checked={!!settings.mmproj_path} onChange={on => edit('mmproj_path', on ? projector : '')}/>}
          {fit && <span className={`fit-badge fit-level-${fit.level}`} title={fit.detail}><span className="fit-bars" aria-hidden="true">{[1, 2, 3, 4].map(n => <i key={n} className={n <= fit.level ? 'on' : ''}/>)}</span>{fit.label}</span>}
          <div className="model-controls"><button className="text-button" disabled={!!busy} onClick={() => tab === 'models' ? browseFile('model_path') : setTab('models')}>{tab === 'models' ? 'Browse file' : 'Select model'}</button>
          {running && <button className="secondary" disabled={!!busy} onClick={() => action('stop', async () => { await invoke('stop_runtime'); })}><Square size={13}/>{ready ? 'Unload' : 'Cancel loading'}</button>}
          {(!running || model.switching || runtimeDirty) && <button className="primary" disabled={!!busy || !model.selected || (running && !ready)} onClick={loadSelected}>{busy === 'start' ? <RefreshCw className="spin" size={14}/> : <Play size={14}/>} {busy === 'start' ? 'Starting…' : running ? model.switching ? 'Switch model' : 'Reload model' : 'Load model'}</button>}</div>
        </section>
        {loadError && <div className="load-error" role="alert"><AlertCircle size={18}/><div><strong>{loadError.split('\n')[0]}</strong>{loadError.includes('\n') && <pre>{loadError.split('\n').slice(1).join('\n')}</pre>}</div><div className="load-error-actions"><button className="text-button" onClick={() => setShowLogs(true)}>View logs</button><button className="icon-button" aria-label="Dismiss error" onClick={() => setDismissedLoadError(snapshot?.load_error ?? null)}><X size={15}/></button></div></div>}

        {tab === 'playground' && <div className="playground">
          <div className="playground-main">
            <section className={`panel state-panel ${dropping ? 'dropping' : ''}`} onDragOver={event => { if (!visionLoaded || !event.dataTransfer.types.includes('Files')) return; event.preventDefault(); setDropping(true); }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setDropping(false); }} onDrop={event => { if (!visionLoaded) return; event.preventDefault(); setDropping(false); addImages([...event.dataTransfer.files]); }}><div className="panel-heading"><span><Terminal size={16}/> State <span className="muted normal-weight">The content to evaluate</span></span><div className="segmented"><button className={stateMode === 'text' ? 'on' : ''} onClick={() => switchStateMode('text')}>Text</button><button className={stateMode === 'json' ? 'on' : ''} onClick={() => switchStateMode('json')}>JSON</button></div></div>{stateMode === 'json' ? <div className="state-json"><JsonEditor label="State JSON" value={content} onChange={value => { setContent(value); setResult(null); }}/></div> : <textarea aria-label="State content" value={content} spellCheck={false} placeholder={visionLoaded ? 'Content to evaluate, or leave empty to ask about the images only…' : 'Content to evaluate…'} onPaste={event => { const files = [...event.clipboardData.files].filter(file => file.type.startsWith('image/')); if (visionLoaded && files.length) { if (!event.clipboardData.types.includes('text/plain')) event.preventDefault(); addImages(files); } }} onChange={e => { setContent(e.target.value); setResult(null); }}/>}{visionLoaded && <ImageAttachments images={images} onFiles={addImages} onRemove={index => { setImages(current => current.filter((_, i) => i !== index)); setResult(null); }}/>}<div className="state-footer"><span>{content.length.toLocaleString()} characters<span className="separator">/</span>{stateMode === 'text' ? 'Plain text' : 'Structured JSON'}{visionLoaded && images.length > 0 && <><span className="separator">/</span>{images.length} {images.length === 1 ? 'image' : 'images'}</>}</span><div className="state-run"><button className="text-button" onClick={() => { setContent(''); setImages([]); setResult(null); }}>Clear</button><button className="primary" disabled={!!busy || !!jsonError || !ready} onClick={run}>{busy === 'decide' ? <RefreshCw className="spin" size={15}/> : <Play size={15}/>} {busy === 'decide' ? 'Deciding…' : 'Run decision'}</button></div></div></section>
            {(jsonError || !ready) && <p className="run-hint">{jsonError ? <span className="invalid">{jsonError}</span> : 'Load a model to run a decision.'}</p>}
            <section className="panel results-panel"><div className="panel-heading"><span><Activity size={16}/> Result {result && <span className="chip success">Complete</span>}</span><div className="heading-actions">{result && <span className="timing">{elapsed.toFixed(1)} ms</span>}<div className="segmented"><button className={resultTab === 'decisions' ? 'on' : ''} onClick={() => setResultTab('decisions')}>Decisions</button><button className={resultTab === 'json' ? 'on' : ''} onClick={() => setResultTab('json')}>JSON</button></div>{result && <button className="icon-button" aria-label="Copy result" onClick={() => copy(pretty(result), 'result')}>{copied === 'result' ? <Check size={14}/> : <Copy size={14}/>}</button>}</div></div>{!result ? <div className="empty-result"><Activity size={17}/><p>{ready ? 'Run a decision to view results.' : 'Results appear here after a run.'}</p></div> : resultTab === 'json' ? <div className="result-code"><JsonEditor label="Result JSON" value={pretty(result)} readOnly/></div> : <div className="decision-results">{Object.entries(result.answers ?? {}).map(([name, answer]) => <div className="answer" key={name}><div className="answer-heading"><span>{name}</span><strong>{answer.choice ?? (answer.noul !== undefined ? `${(answer.noul * 100).toFixed(1)}% yes` : answer.score?.toFixed(2))}</strong>{answer.confidence !== undefined && <small>{(answer.confidence * 100).toFixed(1)}% confidence</small>}</div>{Object.entries(answer.probabilities ?? (answer.noul !== undefined ? { yes: answer.noul, no: 1 - answer.noul } : {})).map(([key, value]) => <div className="probability" key={key}><span>{key}</span><div><i style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}/></div><code>{(value * 100).toFixed(1)}%</code></div>)}</div>)}<div className="usage">{result.usage?.input_tokens.toLocaleString()} input tokens <span>/</span> {result.usage?.output_tokens ?? 0} generated tokens <span>/</span> {elapsed.toFixed(1)} ms round trip</div></div>}</section>
          </div>
          <aside className="run-settings" aria-label="Run settings">
            <h2>Run settings</h2>
            <div className="rail-section"><div className="rail-label">Preset</div><div className="preset-chips">{presets.map((p, i) => { const Icon = icons[i]; return <button key={p.id} className={`preset-chip ${p.id === preset ? 'selected' : ''}`} aria-pressed={p.id === preset} title={p.subtitle} onClick={() => selectPreset(p.id)}><Icon size={14}/>{p.title}</button>; })}{savedPresets.map(p => <span className={`preset-chip saved ${p.id === preset ? 'selected' : ''}`} key={p.id}><button aria-pressed={p.id === preset} onClick={() => selectPreset(p.id)}><FileJson size={14}/>{p.name}</button><button className="chip-remove" aria-label={`Delete preset ${p.name}`} title={`Delete ${p.name}`} disabled={!!busy} onClick={() => trashPreset(p.id)}><X size={12}/></button></span>)}</div><p className="rail-note">{presets.find(p => p.id === preset)?.subtitle ?? (preset.startsWith('saved-') ? 'Custom preset' : 'Custom questions')}</p></div>
            <section className="question-panel"><div className="rail-heading"><span className="rail-label">Questions <span className="code-badge">JSON</span></span><div className="heading-actions"><button className="text-button" onClick={() => { try { setQuestions(pretty(JSON.parse(questions))); } catch (e) { setNotice({ text: errorText(e), error: true }); } }}>Format</button><button className="icon-button" aria-label="Copy questions" onClick={() => copy(questions, 'questions')}>{copied === 'questions' ? <Check size={14}/> : <Copy size={14}/>}</button></div></div><JsonEditor label="Questions JSON" value={questions} onChange={value => { setQuestions(value); setResult(null); }}/><div className="editor-footer"><span className={questionError ? 'invalid' : 'valid'}><span className={`status-dot ${questionError ? 'red' : 'green'}`}/>{questionError ? 'Check JSON' : `${Object.keys(JSON.parse(questions)).length} questions`}</span><span>choice, noul or score</span></div></section>
            <form className="preset-save" onSubmit={e => { e.preventDefault(); saveCustom(); }}><input aria-label="Custom preset name" placeholder="Preset name" maxLength={80} value={presetName} onChange={e => setPresetName(e.target.value)}/><button className="secondary compact" disabled={!!busy || !!jsonError || !presetName.trim()}><Save size={13}/>{preset.startsWith('saved-') ? 'Save changes' : 'Save preset'}</button></form>
          </aside>
        </div>}
        {tab === 'models' && <>
          {modelFiles.length > 0 && <><div className="section-heading"><h2>On your machine</h2><span>{modelFiles.length} {modelFiles.length === 1 ? 'model' : 'models'}</span></div><div className="model-list">{modelFiles.map(model => { const hasProjector = localModels.some(file => isProjector(file.path) && folder(file.path) === folder(model.path)); return <div key={model.path} className="local-model-entry"><button className="model-row" onClick={() => { selectModel(model.path); setTab('runtime'); }}><AuthorLogo author={authorFromPath(model.path)}/><span><strong>{model.name}</strong><small>{(model.size / 1e9).toFixed(2)} GB / {model.path}</small></span>{hasProjector && <span className="chip" title="A multimodal projector is in the same folder. Turn it on from the model bar or Runtime.">Vision available</span>}<span className="pill">{model.path === snapshot?.loaded_model?.path ? 'Loaded' : model.path === settings.model_path ? 'Selected' : 'Select'}</span></button><button className="icon-button reveal-file" aria-label={`Show ${model.name} in folder`} title="Show in folder" disabled={!!busy} onClick={() => revealFile(model.path)}><FolderOpen size={16}/></button></div>; })}</div></>}
          {modelFiles.length === 0 && <div className="local-empty"><FolderOpen size={18}/><p>No models on this machine yet. Download one from the <button className="text-button" onClick={() => setTab('hub')}>Hub</button> or point Noullama at a GGUF file.</p></div>}
          {scanWarnings.length > 0 && <div className="hint" role="status">{scanWarnings.map(warning => <p key={warning}>{warning}</p>)}</div>}
          <div className="import-note"><FolderOpen size={18}/><div><strong>Local GGUF</strong><p>Already have a GGUF file? Pick it here instead of downloading.</p></div><button className="text-button" onClick={() => browseFile('model_path')}>Browse local model<ArrowRight size={14}/></button></div>
          <section className="panel directory-panel"><div className="directory-field"><label>Model download directory<input aria-label="Model download directory" value={downloadDir} placeholder={snapshot ? `${snapshot.data_dir}/models` : 'Default app model directory'} onChange={e => setDownloadDir(e.target.value)}/></label><small>New downloads use this directory. Existing files stay in place.</small></div><button className="secondary" disabled={!!busy || !!snapshot?.download.active || downloadDir === snapshot?.settings.download_dir} onClick={saveDownloadDirectory}>Save directory</button></section>
        </>}
        {tab === 'hub' && <>
          <form className="hub-search" onSubmit={e => { e.preventDefault(); const q = query.trim(); action('search', async () => { setModels(await invoke('search_models', { query: q, contextSize: settings.context_size, gpuLayers: settings.gpu_layers })); setSearched(q); }); }}><div className="search-input"><Search size={17}/><input aria-label="Search decision models" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search decision models, or enter organization/repository"/>{query && <button type="button" className="icon-button" aria-label="Clear search" onClick={() => { setQuery(''); setModels([]); setSearched(''); }}><X size={15}/></button>}</div><button className="secondary" disabled={!!busy}>{busy === 'search' ? 'Searching…' : 'Search'}</button></form>
          <div className="hub-panes">
            <section className="hub-list" aria-label="Repositories">
              {repoLike && <><div className="pane-label">Repository</div>{hubRow('open', query.trim(), <>Open repository</>, null)}</>}
              {searched && <div className="pane-label">{models.length ? `Results for “${searched}”` : `No decision models match “${searched}”`}</div>}
              {models.map(m => hubRow('result', m.id, <><Download size={12}/>{m.downloads?.toLocaleString() ?? 0}</>, m.fit, m.fit ? `Estimate for the smallest listed GGUF. ${m.fit.detail}` : undefined))}
              <div className="pane-label">Recommended</div>
              {hubRow('recommended', 'LiquidAI/d1-2B-GGUF', <>1.92B</>, null)}
              <div className="hf-card">
                <strong className="auth-heading"><HuggingFaceLogo size={16}/>{session?.authenticated ? `Connected as ${session.account}` : 'Hugging Face'}</strong>
                <p>{session && !session.cli_available ? 'hf CLI not found. Install it or set the executable path in Runtime.' : session?.authenticated ? 'Downloads use your Hugging Face session.' : 'Public models work without signing in. Uses HF_TOKEN or your hf auth login session.'}</p>
                <div className="hf-card-actions"><button className="secondary compact" disabled={!!busy} onClick={() => action('session', async () => setSession(await invoke('hf_session')))}><RefreshCw size={13}/>Check session</button></div>
                <details className="token-details"><summary><ChevronRight className="disclosure" size={14}/>Use a token for this session</summary><div className="inline-form"><input type="password" autoComplete="off" aria-label="Hugging Face token" placeholder="hf_… (kept in memory only)" value={token} onChange={e => setToken(e.target.value)}/><button className="secondary compact" onClick={() => action('token', async () => { await invoke('set_hf_token', { token }); setToken(''); setSession(await invoke('hf_session')); })}>Apply</button></div><button className="text-button" onClick={() => action('token', async () => { await invoke('set_hf_token', { token: '' }); setSession(await invoke('hf_session')); })}>Use saved login instead</button></details>
              </div>
            </section>
            <section className="hub-detail" id="hub-detail" aria-label="Repository details">
              {!inspection ? <div className="hub-empty">{busy === 'inspect' ? <><RefreshCw className="spin" size={20}/><p>Reading repository…</p></> : <><Network size={22}/><p>Pick a repository to see its GGUF files.</p></>}</div> : <>
                <div className="detail-head"><AuthorLogo author={inspection.repo.split('/')[0]} size={56}/><div><h2>{inspection.repo.split('/').slice(1).join('/')}</h2><p>{inspection.repo.split('/')[0]}</p></div></div>
                <div className="detail-chips"><span className="chip">GGUF</span><span className="chip">{inspection.files.length} {inspection.files.length === 1 ? 'file' : 'files'}</span>{inspection.sha && <span className="chip" title={inspection.sha}>Revision <code>{inspection.sha.slice(0, 10)}</code></span>}</div>
                <div className="files" id="hub-files">{inspection.files.length === 0 ? <p className="hint">No GGUF files in this repository. Open a GGUF conversion of this model.</p> : inspection.files.map(file => <div className="file-entry" key={file.name}><div className="file-row"><FileJson size={17}/><div><strong>{file.name}</strong><small>{file.projector ? 'Multimodal projector' : 'Model weights'}{file.size ? ` / ${(file.size / 1e9).toFixed(2)} GB` : ''}</small></div>{!file.projector && <FileFit fit={file.fit}/>}<button className="secondary compact" disabled={snapshot?.download.active || !!busy} onClick={() => startDownload(inspection.repo, file.name)}><ArrowDownToLine size={14}/>Download</button></div>{snapshot?.download.repo === inspection.repo && snapshot.download.file === file.name ? <DownloadPanel download={snapshot.download} busy={!!busy} onCancel={cancelDownload} onReveal={() => revealFile(snapshot.download.path!)} onSelect={() => { if (file.projector) edit('mmproj_path', snapshot.download.path!); else selectModel(snapshot.download.path!); setTab('runtime'); }}/> : downloadAnchor?.repo === inspection.repo && downloadAnchor.file === file.name && busy === 'download' ? <div className="download-preparing" role="status"><RefreshCw className="spin" size={14}/>Preparing download…</div> : null}</div>)}</div>
              </>}
            </section>
          </div>
        </>}
        {tab === 'runtime' && <>
          <RuntimeManager currentPath={settings.runtime_path} running={running} progress={snapshot?.runtime_download} onRefresh={refresh} onSelect={runtime => { edit('runtime_path', runtime.path); if (runtime.backend) edit('gpu_layers', runtime.backend === 'cpu' ? 0 : 99); }}>
            <label>Path to llama-server<div className="file-input-row"><input placeholder="/path/to/llama.cpp/build/bin" value={settings.runtime_path} onChange={e => edit('runtime_path', e.target.value)}/><button className="secondary" type="button" disabled={!!busy} onClick={() => action('probe', async () => setVersion(await invoke('probe_runtime', { path: settings.runtime_path, libraryPath: settings.library_path })))}><RefreshCw size={14} className={busy === 'probe' ? 'spin' : ''}/>Check</button></div><small>A runtime folder or the llama-server executable. Choosing a build above fills this in.</small></label>
            {version && <pre className="runtime-version">{version}</pre>}
            <label>Shared-library folders (optional)<input placeholder="Found automatically, or /usr/local/cuda/lib64" value={settings.library_path} onChange={e => edit('library_path', e.target.value)}/><small>For CUDA or other libraries the runtime cannot find. Separate folders with a colon. Only the runtime process sees them.</small></label>
          </RuntimeManager>
          <section className="panel settings-panel"><div className="panel-heading"><span><Layers3 size={16}/> Model</span>{fit && <span className={`fit-badge fit-level-${fit.level}`}><span className="fit-bars" aria-hidden="true">{[1, 2, 3, 4].map(n => <i key={n} className={n <= fit.level ? 'on' : ''}/>)}</span>{fit.label}</span>}</div><div className="settings-body">
            <p className="hint fit-note">{fit ? fit.detail : model.selected ? 'The selected model file could not be read.' : 'Select a model to see how it would run on this machine.'}</p>
            <label>Model file (GGUF)<div className="file-input-row"><input placeholder="/path/to/model.gguf" value={settings.model_path} onChange={e => edit('model_path', e.target.value)}/><button className="secondary" type="button" disabled={!!busy} onClick={() => browseFile('model_path')}><FolderOpen size={14}/>Browse</button></div></label>
            <div className="field"><div className="field-row"><span>Multimodal projector</span><Toggle label={settings.mmproj_path ? 'On' : 'Off'} ariaLabel="Multimodal projector" checked={!!settings.mmproj_path} disabled={!settings.mmproj_path && !projector} onChange={on => edit('mmproj_path', on ? projector : '')}/></div>{settings.mmproj_path ? <div className="file-input-row"><input aria-label="Multimodal projector path" placeholder="/path/to/mmproj.gguf" value={settings.mmproj_path} onChange={e => edit('mmproj_path', e.target.value)}/><button className="secondary" type="button" disabled={!!busy} onClick={() => browseFile('mmproj_path')}><FolderOpen size={14}/>Browse</button></div> : <small>{projector ? `Found ${basename(projector)} next to the selected model.` : <>No projector next to the selected model. <button type="button" className="text-button inline" disabled={!!busy} onClick={() => browseFile('mmproj_path')}>Choose a file</button></>}</small>}<small>Lets image-capable models read images in the Playground.</small></div>
            <div className="form-grid"><label>Context size<NumberField min={512} max={131072} step={1024} value={settings.context_size} onChange={value => edit('context_size', value)}/><small>Maximum tokens per request. The default is 8192.</small></label><label>GPU layers<NumberField min={-1} value={settings.gpu_layers} onChange={value => edit('gpu_layers', value)}/><small>0 runs on the CPU. 99 puts every layer that fits on the GPU.</small></label></div>
          </div></section>
          <section className="panel settings-panel"><div className="panel-heading"><span><Settings2 size={16}/> Connection</span></div><div className="settings-body">
            <div className="form-grid"><label>Local API port<NumberField min={1} max={65535} value={settings.api_port} onChange={value => edit('api_port', value)}/><small>Used after a restart. If it is taken, Noullama picks a free nearby port and shows it in Local API.</small></label><label>Private llama-server port<NumberField min={1} max={65535} value={settings.runtime_port} onChange={value => edit('runtime_port', value)}/><small>Internal only. Both ports listen on 127.0.0.1.</small></label></div>
            <label>Hugging Face CLI<input value={settings.hf_path} onChange={e => edit('hf_path', e.target.value)} placeholder="hf"/><small>Use the full path if hf is not on the desktop app's PATH.</small></label>
          </div></section>
          <div className={`save-bar ${dirty ? 'dirty' : ''}`}><span className="hint">{running ? 'Unload the model to save changes. Reload model applies model changes right away.' : dirty ? 'You have unsaved changes.' : 'All changes saved.'}</span><button className={dirty && !running ? 'primary' : 'secondary'} disabled={!dirty || !!busy || running} onClick={() => action('save', async () => { await save(); setNotice({ text: 'Settings saved.', error: false }); })}>Save settings</button></div>
        </>}
        {tab === 'api' && <>
          <section className="endpoint-card"><div><span className={`status-dot ${!snapshot?.api_error && desktop ? 'green' : ''}`}/><strong>{apiBase}</strong><button className="icon-button" aria-label="Copy API URL" onClick={() => copy(apiBase, 'api')}>{copied === 'api' ? <Check size={16}/> : <Copy size={16}/>}</button></div><p>{snapshot?.api_error ?? 'Localhost only. No API key needed. Available while Noullama is open.'}</p></section>
          <div className="endpoint-list"><div><span className="method get">GET</span><code>/v1/models</code><span>List the loaded decision model</span></div><div><span className="method">POST</span><code>/v1/systemone</code><span>Native state + typed questions</span></div><div><span className="method">POST</span><code>/v1/chat/completions</code><span>OpenAI chat response envelope</span></div><div><span className="method get">GET</span><code>/health</code><span>Gateway health and model readiness (outside /v1)</span></div></div>
          <section className="panel code-panel"><div className="panel-heading"><span>Native decision request</span><button className="icon-button" aria-label="Copy native API example" onClick={() => copy(`curl ${apiBase}/systemone -H 'Content-Type: application/json' -d '${requestJson.replace(/'/g, "'\\''")}'`, 'curl')}>{copied === 'curl' ? <Check size={14}/> : <Copy size={14}/>}</button></div><pre>{`curl ${apiBase}/systemone \\\n  -H 'Content-Type: application/json' \\\n  -d '${requestJson.replace(/'/g, "'\\''")}'`}</pre></section>
          <section className="panel code-panel"><div className="panel-heading"><span>Python · OpenAI SDK</span><span className="chip">Non-streaming</span></div><pre>{`import json\nfrom openai import OpenAI\n\nclient = OpenAI(base_url="${apiBase}", api_key="local")\n\nrequest = {\n    "state": "I was charged twice. Please refund the duplicate.",\n    "questions": {\n        "refund": {\n            "type": "noul",\n            "instructions": "Is the customer requesting a refund?"\n        }\n    }\n}\n\nresponse = client.chat.completions.create(\n    model="noullama",\n    messages=[{"role": "user", "content": json.dumps(request)}],\n    stream=False,\n)\ndecision = json.loads(response.choices[0].message.content)\nprint(decision["answers"])`}</pre></section>
          <div className="api-note"><BookOpen size={18}/><div><strong>Request format</strong><p>Send one user message containing a JSON request with state and questions. The assistant content contains the native result as JSON. Chat history, tools, and streaming are unsupported. Questions use choice, noul, or score; this runtime does not support extraction or multi-select.</p><a href="https://github.com/ggml-org/llama.cpp/blob/b11938/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api" target="_blank" rel="noreferrer">llama.cpp decision API reference<ExternalLink size={12}/></a></div></div>
        </>}
      </div>
    </main>
    {showDownloadToast && download && <aside className="download-toast" aria-label="Downloads">
      <div className="toast-head"><strong>Downloads</strong><span>{!download.active && <button className="icon-button" aria-label="Dismiss" onClick={() => setToastDismissed(`${download.repo}/${download.file}`)}><X size={15}/></button>}<button className="icon-button" aria-label={toastCollapsed ? 'Show downloads' : 'Collapse downloads'} aria-expanded={!toastCollapsed} onClick={() => setToastCollapsed(v => !v)}>{toastCollapsed ? <ChevronUp size={15}/> : <ChevronDown size={15}/>}</button></span></div>
      {!toastCollapsed && <><div className="toast-row"><button className="toast-open" onClick={() => { setTab('hub'); openRepo(download.repo); }}><strong>{download.repo}</strong><small>{download.active ? `${download.file} / ${download.progress === null ? 'Connecting' : `${download.progress.toFixed(0)}%`}` : `${download.file} / Downloaded`}</small></button>{download.active ? <button className="icon-button" aria-label="Cancel download" title="Cancel download" disabled={busy === 'cancel'} onClick={cancelDownload}><X size={15}/></button> : <CircleCheck className="toast-done" size={17}/>}</div>
      {download.active && <div className="download-track" aria-hidden="true"><i style={{ width: `${Math.max(0, Math.min(100, download.progress ?? 0))}%` }}/></div>}</>}
    </aside>}
    <ContextMenu/>
    <QuitDialog open={snapshot?.quit_confirmation ?? false} pending={busy === 'quit' || busy === 'cancel-quit'} onCancel={() => { void action('cancel-quit', async () => { await invoke('cancel_quit'); }); }} onConfirm={() => { void action('quit', async () => { await invoke('quit_app'); }); }}/>
    {showLogs && <div className="log-drawer"><div className="panel-heading"><span><Terminal size={16}/> Runtime activity</span><button className="icon-button" aria-label="Close runtime logs" onClick={() => setShowLogs(false)}><X size={17}/></button></div><pre>{snapshot?.logs.join('\n') || 'No runtime activity yet.'}</pre></div>}
  </div>;
}

createRoot(document.getElementById('root')!).render(<App/>);
