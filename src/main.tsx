import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { Activity, ArrowDownToLine, ArrowRight, BookOpen, Check, ChevronRight, Code2, Copy, Cpu, Download, ExternalLink, FileJson, FlaskConical, FolderOpen, Layers3, Mail, Network, Play, Plus, RefreshCw, Route, Search, Settings2, Shield, ShieldCheck, Square, Terminal, Trash2, Save, X } from 'lucide-react';
import { buildRequest, convertState, presets } from './presets';
import { modelStatus } from './model-state';
import { JsonEditor } from './JsonEditor';
import { DownloadPanel, type DownloadState } from './DownloadPanel';
import { visibleBanner } from './ui-state';
import './styles.css';

const logoUrl = new URL('../logo.png', import.meta.url).href;

type Settings = { runtime_path: string; model_path: string; mmproj_path: string; context_size: number; gpu_layers: number; api_port: number; runtime_port: number; hf_path: string; download_dir: string; library_path: string };

type Snapshot = { settings: Settings; running: boolean; ready: boolean; api_port: number; api_error: string | null; logs: string[]; download: DownloadState; data_dir: string; download_dir: string; loaded_model: { path: string; device: string } | null };
type SavedPreset = { id: string; name: string; questions: unknown; state: string; state_mode: 'text' | 'json' };
type Model = { id: string; downloads: number; likes: number; tags: string[] };
type ModelFile = { name: string; size?: number; projector: boolean };
type Inspection = { repo: string; files: ModelFile[]; sha: string };
type Session = { authenticated: boolean; account: string | null; cli_available: boolean; token_present: boolean };
type Result = { answers?: Record<string, { type?: string; choice?: string; score?: number; noul?: number; confidence?: number; probabilities?: Record<string, number> }>; usage?: { input_tokens: number; output_tokens: number }; [key: string]: unknown };
const defaultSettings: Settings = { runtime_path: '/home/andrewunknown/llamacpp-new-patch/llama-b11954-bin-ubuntu-cuda-13.4-x64/llama-b11954', model_path: '', mmproj_path: '', context_size: 8192, gpu_layers: 99, api_port: 11435, runtime_port: 11436, hf_path: 'hf', download_dir: '', library_path: '' };
const icons = [Route, ShieldCheck, Shield, Mail, Network, Plus];
const pretty = (v: unknown) => JSON.stringify(v, null, 2);
const errorText = (e: unknown) => e instanceof Error ? e.message : String(e);
const basename = (path: string) => path.split(/[\\/]/).pop() || 'No model loaded';

function App() {
  const [tab, setTab] = useState('playground');
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [dirty, setDirty] = useState(false);
  const [preset, setPreset] = useState('triage');
  const [savedPresets, setSavedPresets] = useState<SavedPreset[]>([]);
  const [presetName, setPresetName] = useState('');
  const [dismissedApi, setDismissedApi] = useState<string | null>(null);
  const [previewDismissed, setPreviewDismissed] = useState(false);
  const [downloadAnchor, setDownloadAnchor] = useState<{repo:string;file:string} | null>(null);
  const [questions, setQuestions] = useState(pretty(presets[0].questions));
  const [content, setContent] = useState(presets[0].state);
  const [stateMode, setStateMode] = useState<'text' | 'json'>('text');
  const [stateMemory, setStateMemory] = useState<{ json: string; text: string } | null>(null);
  const [downloadDir, setDownloadDir] = useState('');
  const [result, setResult] = useState<Result | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [resultTab, setResultTab] = useState('decisions');
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ text: string; error: boolean } | null>(null);
  const [showLogs, setShowLogs] = useState(false);
  const [query, setQuery] = useState('');
  const [models, setModels] = useState<Model[]>([]);
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [localModels, setLocalModels] = useState<{ path: string; name: string; size: number }[]>([]);
  const [repo, setRepo] = useState('LiquidAI/d1-2B-GGUF');
  const [session, setSession] = useState<Session | null>(null);
  const [token, setToken] = useState('');
  const [version, setVersion] = useState('');
  const [copied, setCopied] = useState('');
  const desktop = isTauri();

  useEffect(() => { window.scrollTo({ top: 0, behavior: 'instant' }); }, [tab]);

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
  useEffect(() => { if (desktop && tab === 'models' && !session) invoke<Session>('hf_session').then(setSession).catch(e => setNotice({ text: errorText(e), error: true })); }, [tab]);
  useEffect(() => { if (desktop && tab === 'models') invoke<typeof localModels>('local_models').then(setLocalModels).catch(() => {}); }, [tab, snapshot?.download.path, snapshot?.download_dir]);
  async function action(name: string, task: () => Promise<void>) {
    if (!desktop) { setNotice({ text: 'Launch the desktop app with npm run desktop to use the local backend.', error: true }); return; }
    setBusy(name); setNotice(null);
    try { await task(); await refresh(); } catch (e) { setNotice({ text: errorText(e), error: true }); } finally { setBusy(null); }
  }
  function edit<K extends keyof Settings>(key: K, value: Settings[K]) { setSettings(s => ({ ...s, [key]: value })); setDirty(true); }
  async function save() { await invoke('save_settings', { settings }); setDirty(false); }
  async function run() {
    await action('decide', async () => {
      const payload = buildRequest(questions, content, stateMode);
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
      if (path) edit(field, path);
    });
  }
  async function revealFile(path: string) {
    await action('reveal', async () => { await invoke('reveal_model_file', { path }); });
  }
  async function startDownload(repo: string, file: string) {
    setDownloadAnchor({ repo, file });
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
  const banner = visibleBanner(notice, snapshot?.api_error, dismissedApi, desktop, previewDismissed);
  const ready = snapshot?.ready ?? false;
  const running = snapshot?.running ?? false;
  const model = modelStatus(settings.model_path, snapshot?.loaded_model ?? null, running, ready);
  const runtimeDirty = snapshot ? (['runtime_path', 'model_path', 'mmproj_path', 'context_size', 'gpu_layers', 'runtime_port', 'library_path'] as const).some(key => settings[key] !== snapshot.settings[key]) : false;
  const apiBase = `http://127.0.0.1:${snapshot?.api_port ?? 11435}/v1`;
  let questionError = '';
  try { buildRequest(questions, '', 'text'); } catch (e) { questionError = errorText(e); }
  let jsonError = '';
  try { buildRequest(questions, content, stateMode); } catch (e) { jsonError = errorText(e); }
  const requestJson = (() => { try { return pretty(buildRequest(questions, content, stateMode)); } catch { return '{}'; } })();

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><img className="brand-logo" src={logoUrl} width={30} height={30} alt=""/><span>noullama<span className="brand-dot">.</span></span></div>
      <div className="workspace-label">LOCAL WORKSPACE <span>v0.1</span></div>
      <nav>{[
        ['playground', FlaskConical, 'Playground'], ['models', Layers3, 'Models'], ['runtime', Cpu, 'Runtime'], ['api', Code2, 'Local API'],
      ].map(([id, Icon, title]) => { const I = Icon as typeof Cpu; return <button key={String(id)} className={`nav-item ${tab === id ? 'active' : ''}`} onClick={() => setTab(String(id))}><I size={18}/><span>{String(title)}</span>{tab === id && <span className="nav-marker"/>}</button>; })}</nav>
      <div className="sidebar-bottom">
        <button className="activity-toggle" onClick={() => setShowLogs(v => !v)}><Terminal size={16}/> Runtime logs <ChevronRight size={14}/></button>
        <div className="sidebar-footer"><span className={`status-dot ${ready ? 'green' : ''}`}/>{ready ? 'Model ready' : running ? 'Loading model' : 'Runtime offline'}<span className="shortcut">LOCAL</span></div>
      </div>
    </aside>
    <main>
      <header className="topbar"><div className="breadcrumb">Workspace <ChevronRight size={13}/><span>{({ playground: 'Playground', models: 'Models', runtime: 'Runtime', api: 'Local API' } as Record<string,string>)[tab]}</span></div><div className="topbar-right"><span className="pill"><span className={`status-dot ${ready ? 'green' : ''}`}/>{ready ? 'Runtime ready' : running ? 'Loading…' : 'Not running'}</span></div></header>
      {banner && <div className={`notice ${banner.error ? 'error' : ''}`} role={banner.error ? 'alert' : 'status'}><span>{banner.text}</span><button aria-label="Dismiss notification" onClick={() => { setNotice(null); setDismissedApi(snapshot?.api_error ?? null); setPreviewDismissed(true); }}><X size={16}/></button></div>}
      <div className="page-content">
        <div className="page-heading"><h1>{({ playground: 'Playground', models: 'Models', runtime: 'Runtime', api: 'Local API' } as Record<string,string>)[tab]}</h1></div>
        <section className="model-strip" aria-label="Model status">
          <div className="model-strip-icon"><Cpu size={20}/></div>
          <div className="model-identity"><div className="small-label">{running ? ready ? 'LOADED MODEL' : 'LOADING MODEL' : 'SELECTED MODEL'}</div><strong title={model.loaded || model.selected}>{model.loaded ? basename(model.loaded) : model.selected ? basename(model.selected) : 'No model selected'}</strong>{model.device && <small>{model.device}</small>}{model.switching && <small className="pending-model" title={model.selected}>Next: {basename(model.selected)}</small>}</div>
          <span className={`pill ${ready ? 'success' : ''}`}>{busy === 'start' ? 'Starting…' : model.label}</span>
          <div className="model-controls"><button className="text-button" disabled={!!busy} onClick={() => tab === 'models' ? browseFile('model_path') : setTab('models')}>{tab === 'models' ? 'Browse file' : 'Select model'}</button>
          {running && <button className="secondary" disabled={!!busy} onClick={() => action('stop', async () => { await invoke('stop_runtime'); })}><Square size={13}/>{ready ? 'Unload' : 'Cancel loading'}</button>}
          {(!running || model.switching || runtimeDirty) && <button className="primary" disabled={!!busy || !model.selected || (running && !ready)} onClick={loadSelected}>{busy === 'start' ? <RefreshCw className="spin" size={14}/> : <Play size={14}/>} {busy === 'start' ? 'Starting…' : running ? model.switching ? 'Switch model' : 'Reload model' : 'Load selected'}</button>}</div>
        </section>

        {tab === 'playground' && <>
          

          <div className="workbench">
            <section className="panel preset-panel"><div className="panel-heading"><span><Layers3 size={15}/> Presets</span><span className="muted">{presets.length + savedPresets.length}</span></div><div className="presets-list">{presets.map((p, i) => { const Icon = icons[i]; return <button key={p.id} className={`preset ${p.id === preset ? 'selected' : ''}`} onClick={() => selectPreset(p.id)}><span className="preset-icon"><Icon size={18}/></span><span><strong>{p.title}</strong><small>{p.subtitle}</small></span>{p.id === preset && <ChevronRight size={15}/>}</button>; })}{savedPresets.length > 0 && <div className="saved-label">SAVED</div>}{savedPresets.map(p => <div className="saved-preset-row" key={p.id}><button className={`preset ${p.id === preset ? 'selected' : ''}`} onClick={() => selectPreset(p.id)}><span className="preset-icon"><FileJson size={17}/></span><span><strong>{p.name}</strong><small>Custom preset</small></span></button><button className="icon-button trash-preset" aria-label={`Delete preset ${p.name}`} title={`Delete ${p.name}`} disabled={!!busy} onClick={() => trashPreset(p.id)}><Trash2 size={14}/></button></div>)}</div></section>
            <section className="panel question-panel"><div className="panel-heading"><span><FileJson size={16}/> Questions <span className="code-badge">JSON</span></span><div className="heading-actions"><button className="text-button" onClick={() => { try { setQuestions(pretty(JSON.parse(questions))); } catch (e) { setNotice({ text: errorText(e), error: true }); } }}>Format</button><button className="icon-button" aria-label="Copy questions" onClick={() => copy(questions, 'questions')}>{copied === 'questions' ? <Check size={14}/> : <Copy size={14}/>}</button></div></div><JsonEditor label="Questions JSON" value={questions} onChange={value => { setQuestions(value); setResult(null); }}/><div className="editor-footer"><span className={questionError ? 'invalid' : 'valid'}><span className="status-dot green"/>{questionError ? 'Check JSON' : 'Valid JSON'}</span><span>choice · noul · score</span></div><form className="preset-save" onSubmit={e => { e.preventDefault(); saveCustom(); }}><input aria-label="Custom preset name" placeholder="Preset name" maxLength={80} value={presetName} onChange={e => setPresetName(e.target.value)}/><button className="secondary compact" disabled={!!busy || !!jsonError || !presetName.trim()}><Save size={13}/>{preset.startsWith('saved-') ? 'Save changes' : 'Save custom'}</button></form></section>
          </div>
          <section className="panel state-panel"><div className="panel-heading"><span><Terminal size={16}/> State <span className="muted normal-weight">The content to evaluate</span></span><div className="segmented"><button className={stateMode === 'text' ? 'on' : ''} onClick={() => switchStateMode('text')}>Text</button><button className={stateMode === 'json' ? 'on' : ''} onClick={() => switchStateMode('json')}>JSON</button></div></div>{stateMode === 'json' ? <div className="state-json"><JsonEditor label="State JSON" value={content} onChange={value => { setContent(value); setResult(null); }}/></div> : <textarea aria-label="State content" value={content} spellCheck={false} placeholder="Content to evaluate…" onChange={e => { setContent(e.target.value); setResult(null); }}/>}<div className="state-footer"><span>{content.length.toLocaleString()} characters<span className="separator">/</span>{stateMode === 'text' ? 'Plain text' : 'Structured JSON'}</span><button className="text-button" onClick={() => { setContent(''); setResult(null); }}>Clear</button></div></section>
          <div className="run-row"><span>{jsonError ? <span className="invalid">{jsonError}</span> : <>{Object.keys(JSON.parse(questions)).length} questions</>}</span><button className="primary" disabled={!!busy || !!jsonError || !ready} onClick={run}>{busy === 'decide' ? <RefreshCw className="spin" size={15}/> : <Play size={15}/>} {busy === 'decide' ? 'Deciding…' : 'Run decision'}<span className="button-arrow"><ArrowRight size={15}/></span></button></div>
          <section className="panel results-panel"><div className="panel-heading"><span><Activity size={16}/> Result {result && <span className="code-badge success">COMPLETE</span>}</span><div className="heading-actions">{result && <span className="timing">{elapsed.toFixed(1)} ms</span>}<div className="segmented"><button className={resultTab === 'decisions' ? 'on' : ''} onClick={() => setResultTab('decisions')}>Decisions</button><button className={resultTab === 'json' ? 'on' : ''} onClick={() => setResultTab('json')}>JSON</button></div>{result && <button className="icon-button" aria-label="Copy result" onClick={() => copy(pretty(result), 'result')}>{copied === 'result' ? <Check size={14}/> : <Copy size={14}/>}</button>}</div></div>{!result ? <div className="empty-result"><Activity size={17}/><p>{ready ? 'Run a decision to view results.' : 'Load a model to run a decision.'}</p></div> : resultTab === 'json' ? <div className="result-code"><JsonEditor label="Result JSON" value={pretty(result)} readOnly/></div> : <div className="decision-results">{Object.entries(result.answers ?? {}).map(([name, answer]) => <div className="answer" key={name}><div className="answer-heading"><span>{name}</span><strong>{answer.choice ?? (answer.noul !== undefined ? `${(answer.noul * 100).toFixed(1)}% yes` : answer.score?.toFixed(2))}</strong>{answer.confidence !== undefined && <small>{(answer.confidence * 100).toFixed(1)}% confidence</small>}</div>{Object.entries(answer.probabilities ?? (answer.noul !== undefined ? { yes: answer.noul, no: 1 - answer.noul } : {})).map(([key, value]) => <div className="probability" key={key}><span>{key}</span><div><i style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}/></div><code>{(value * 100).toFixed(1)}%</code></div>)}</div>)}<div className="usage">{result.usage?.input_tokens.toLocaleString()} input tokens <span>·</span> {result.usage?.output_tokens ?? 0} generated tokens <span>·</span> {elapsed.toFixed(1)} ms round trip</div></div>}</section>
        </>}
        {tab === 'models' && <>
          
          {snapshot?.download.active && snapshot.download.repo !== inspection?.repo && <button className="download-summary" onClick={() => action('inspect', async () => { setRepo(snapshot.download.repo); setInspection(await invoke('inspect_model', { repo: snapshot.download.repo })); })}><Download size={14}/>{snapshot.download.file}<span>{snapshot.download.progress === null ? 'Downloading' : `${snapshot.download.progress.toFixed(0)}%`}</span><ArrowRight size={14}/></button>}
          <section className="panel directory-panel"><div className="directory-field"><label>Model download directory<input aria-label="Model download directory" value={downloadDir} placeholder={snapshot ? `${snapshot.data_dir}/models` : 'Default app model directory'} onChange={e => setDownloadDir(e.target.value)}/></label><small>New downloads use this directory. Existing files stay in place.</small></div><button className="secondary" disabled={!!busy || !!snapshot?.download.active || downloadDir === snapshot?.settings.download_dir} onClick={saveDownloadDirectory}>Save directory</button></section>
          <section className="panel auth-panel"><div><strong>Hugging Face connection</strong><p>{session?.authenticated ? `Connected as ${session.account} · Using your Hugging Face session` : 'Uses HF_TOKEN or your existing hf auth login session.'}</p><small>{session && !session.cli_available ? 'hf CLI not found. Install it or set the executable path in Runtime.' : 'Public models work without signing in. Downloads use the hf CLI.'}</small></div><button className="secondary" disabled={!!busy} onClick={() => action('session', async () => setSession(await invoke('hf_session')))}><RefreshCw size={14}/>Check session</button></section>
          <details className="token-details"><summary>Use a token for this session instead</summary><div className="inline-form"><input type="password" autoComplete="off" aria-label="Hugging Face token" placeholder="hf_… (kept in memory only)" value={token} onChange={e => setToken(e.target.value)}/><button className="secondary" onClick={() => action('token', async () => { await invoke('set_hf_token', { token }); setToken(''); setSession(await invoke('hf_session')); })}>Apply token</button><button className="text-button" onClick={() => action('token', async () => { await invoke('set_hf_token', { token: '' }); setSession(await invoke('hf_session')); })}>Use saved login</button></div></details>
          {localModels.length > 0 && <><div className="section-heading"><h2>On your machine</h2><span>{localModels.length} downloaded files</span></div><div className="model-list">{localModels.map(model => <div key={model.path} className="local-model-entry"><button className="model-row" onClick={() => { edit(model.name.toLowerCase().includes('mmproj') ? 'mmproj_path' : 'model_path', model.path); setTab('runtime'); }}><FolderOpen size={18}/><span><strong>{model.name}</strong><small>{(model.size / 1e9).toFixed(2)} GB · {model.path}</small></span><span className="pill">{model.path === snapshot?.loaded_model?.path ? 'Loaded' : model.path === settings.model_path ? 'Selected' : 'Select'}</span></button><button className="icon-button reveal-file" aria-label={`Show ${model.name} in folder`} title="Show in folder" disabled={!!busy} onClick={() => revealFile(model.path)}><FolderOpen size={16}/></button></div>)}</div></>}
          <section className="featured-model"><div className="featured-top"><Cpu size={20}/><div><h2>Liquid d1-2B</h2><p>1.92B parameters · GGUF · choice / noul / score</p></div></div><div className="featured-bottom"><button className="primary" disabled={!!busy} onClick={() => action('inspect', async () => { setRepo('LiquidAI/d1-2B-GGUF'); setInspection(await invoke('inspect_model', { repo: 'LiquidAI/d1-2B-GGUF' })); })}>View files<ArrowRight size={15}/></button></div></section>
          <div className="section-heading"><h2>Hub search</h2><span>Decision-tagged repositories</span></div><form className="search-row" onSubmit={e => { e.preventDefault(); action('search', async () => setModels(await invoke('search_models', { query }))); }}><div className="search-input"><Search size={17}/><input aria-label="Search decision models" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search decision models…"/></div><button className="secondary" disabled={!!busy}>{busy === 'search' ? 'Searching…' : 'Search Hub'}</button></form>
          {models.length > 0 && <div className="model-list">{models.map(m => <button key={m.id} className="model-row" onClick={() => action('inspect', async () => { setRepo(m.id); setInspection(await invoke('inspect_model', { repo: m.id })); })}><Layers3 size={18}/><span><strong>{m.id}</strong><small>{m.downloads?.toLocaleString() ?? 0} downloads · {m.tags?.includes('gguf') ? 'GGUF available' : 'Inspect repository'}</small></span><ChevronRight size={17}/></button>)}</div>}
          <section className="panel repo-panel"><div className="panel-heading"><span><FolderOpen size={16}/> Open any decision repository</span></div><div className="inline-form"><input aria-label="Hugging Face repository" value={repo} onChange={e => setRepo(e.target.value)} placeholder="organization/decision-model-GGUF"/><button className="secondary" disabled={!!busy} onClick={() => action('inspect', async () => setInspection(await invoke('inspect_model', { repo })))}>Inspect<ArrowRight size={14}/></button></div>{inspection && <div className="files"><div className="file-caption">{inspection.repo}<code>{inspection.sha?.slice(0, 10)}</code></div>{inspection.files.length === 0 ? <p className="hint">No GGUF files in this repository. Open a GGUF conversion of this model.</p> : inspection.files.map(file => <div className="file-entry" key={file.name}><div className="file-row"><FileJson size={17}/><div><strong>{file.name}</strong><small>{file.projector ? 'Multimodal projector' : 'Model weights'}{file.size ? ` · ${(file.size / 1e9).toFixed(2)} GB` : ''}</small></div><button className="secondary compact" disabled={snapshot?.download.active || !!busy} onClick={() => startDownload(inspection.repo, file.name)}><ArrowDownToLine size={14}/>Download</button></div>{snapshot?.download.repo === inspection.repo && snapshot.download.file === file.name ? <DownloadPanel download={snapshot.download} busy={!!busy} onReveal={() => revealFile(snapshot.download.path!)} onSelect={() => { edit(file.projector ? 'mmproj_path' : 'model_path', snapshot.download.path!); setTab('runtime'); }}/> : downloadAnchor?.repo === inspection.repo && downloadAnchor.file === file.name && busy === 'download' ? <div className="download-preparing" role="status"><RefreshCw className="spin" size={14}/>Preparing download…</div> : null}</div>)}</div>}</section>

          <div className="import-note"><FolderOpen size={18}/><div><strong>Local GGUF</strong><p>Use an existing file from Runtime settings.</p></div><button className="text-button" onClick={() => browseFile('model_path')}>Browse local model<ArrowRight size={14}/></button></div>
        </>}
        {tab === 'runtime' && <>
          
          <section className="panel settings-panel"><div className="panel-heading"><span><Terminal size={16}/> Runtime & model</span><span className="code-badge">CUSTOM PATHS</span></div><div className="settings-body"><label>llama.cpp directory or llama-server executable<input value={settings.runtime_path} onChange={e => edit('runtime_path', e.target.value)}/><small>Accepts an extracted runtime directory or the executable itself.</small></label><div className="runtime-probe"><button className="secondary" disabled={!!busy} onClick={() => action('probe', async () => setVersion(await invoke('probe_runtime', { path: settings.runtime_path, libraryPath: settings.library_path })))}><RefreshCw size={14}/>Check runtime</button>{version && <pre>{version}</pre>}</div><label>CUDA / shared-library directories · optional<input placeholder="Automatic detection, or /path/to/cuda/lib64" value={settings.library_path} onChange={e => edit('library_path', e.target.value)}/><small>Separate multiple directories with a colon on Linux. Applied only to the runtime process.</small></label><label>Selected model · GGUF<div className="file-input-row"><input placeholder="/path/to/d1-2B-Q4_K_M.gguf" value={settings.model_path} onChange={e => edit('model_path', e.target.value)}/><button className="secondary" type="button" disabled={!!busy} onClick={() => browseFile('model_path')}><FolderOpen size={14}/>Browse</button></div></label><label>Multimodal projector · optional<div className="file-input-row"><input placeholder="/path/to/mmproj.gguf" value={settings.mmproj_path} onChange={e => edit('mmproj_path', e.target.value)}/><button className="secondary" type="button" disabled={!!busy} onClick={() => browseFile('mmproj_path')}><FolderOpen size={14}/>Browse</button></div><small>Needed for image-capable models. The first playground supports text and JSON states.</small></label><div className="form-grid"><label>Context / batch size<input type="number" min={512} max={131072} value={settings.context_size} onChange={e => edit('context_size', Number(e.target.value))}/><small>d1 evaluates one question in a batch. Default: 8192.</small></label><label>GPU layers<input type="number" min={-1} value={settings.gpu_layers} onChange={e => edit('gpu_layers', Number(e.target.value))}/><small>0 for CPU; 99 offloads available layers to GPU.</small></label></div></div></section>
          <section className="panel settings-panel"><div className="panel-heading"><span><Settings2 size={16}/> Connection settings</span></div><div className="settings-body"><div className="form-grid"><label>Public local API port<input type="number" min={1} max={65535} value={settings.api_port} onChange={e => edit('api_port', Number(e.target.value))}/><small>Changing this port takes effect after restarting the app.</small></label><label>Private llama-server port<input type="number" min={1} max={65535} value={settings.runtime_port} onChange={e => edit('runtime_port', Number(e.target.value))}/><small>Both listeners bind to 127.0.0.1.</small></label></div><label>Hugging Face CLI executable<input value={settings.hf_path} onChange={e => edit('hf_path', e.target.value)} placeholder="hf"/><small>Use an absolute path if hf is not available in the desktop app's PATH.</small></label></div></section>
          <div className="runtime-actions"><span className="hint">{running ? 'Stop the runtime before changing configuration.' : dirty ? 'You have unsaved changes.' : 'Configuration is saved locally.'}</span><button className="secondary" disabled={!dirty || !!busy || running} onClick={() => action('save', async () => { await save(); setNotice({ text: 'Settings saved.', error: false }); })}>Save settings</button></div>
          <div className="runtime-info"><Activity size={17}/><div><strong>{ready ? `Loaded: ${basename(model.loaded)}` : running ? `Loading: ${basename(model.loaded)}` : 'No model loaded'}</strong><p>{ready ? `Serving decisions at ${apiBase}` : 'Select a file, then use Load selected above. Startup details appear in Runtime logs.'}</p></div><button className="text-button" onClick={() => setShowLogs(true)}>View logs<ArrowRight size={14}/></button></div>
        </>}
        {tab === 'api' && <>
          
          <section className="endpoint-card"><div><span className={`status-dot ${!snapshot?.api_error && desktop ? 'green' : ''}`}/><strong>{apiBase}</strong><button className="icon-button" aria-label="Copy API URL" onClick={() => copy(apiBase, 'api')}>{copied === 'api' ? <Check size={16}/> : <Copy size={16}/>}</button></div><p>{snapshot?.api_error ?? 'Localhost only · no API key required · available while noullama is open'}</p></section>
          <div className="endpoint-list"><div><span className="method get">GET</span><code>/v1/models</code><span>List the loaded decision model</span></div><div><span className="method">POST</span><code>/v1/systemone</code><span>Native state + typed questions</span></div><div><span className="method">POST</span><code>/v1/chat/completions</code><span>OpenAI chat response envelope</span></div><div><span className="method get">GET</span><code>/health</code><span>Gateway health and model readiness (outside /v1)</span></div></div>
          <section className="panel code-panel"><div className="panel-heading"><span>Native decision request</span><button className="icon-button" aria-label="Copy native API example" onClick={() => copy(`curl ${apiBase}/systemone -H 'Content-Type: application/json' -d '${requestJson.replace(/'/g, "'\\''")}'`, 'curl')}>{copied === 'curl' ? <Check size={14}/> : <Copy size={14}/>}</button></div><pre>{`curl ${apiBase}/systemone \\\n  -H 'Content-Type: application/json' \\\n  -d '${requestJson.replace(/'/g, "'\\''")}'`}</pre></section>
          <section className="panel code-panel"><div className="panel-heading"><span>Python · OpenAI SDK</span><span className="code-badge">NON-STREAMING</span></div><pre>{`import json\nfrom openai import OpenAI\n\nclient = OpenAI(base_url="${apiBase}", api_key="local")\n\nrequest = {\n    "state": "I was charged twice. Please refund the duplicate.",\n    "questions": {\n        "refund": {\n            "type": "noul",\n            "instructions": "Is the customer requesting a refund?"\n        }\n    }\n}\n\nresponse = client.chat.completions.create(\n    model="noullama",\n    messages=[{"role": "user", "content": json.dumps(request)}],\n    stream=False,\n)\ndecision = json.loads(response.choices[0].message.content)\nprint(decision["answers"])`}</pre></section>
          <div className="api-note"><BookOpen size={18}/><div><strong>Request format</strong><p>Send one user message containing a JSON request with state and questions. The assistant content contains the native result as JSON. Chat history, tools, and streaming are unsupported. Questions use choice, noul, or score; this runtime does not support extraction or multi-select.</p><a href="https://github.com/ggml-org/llama.cpp/blob/b11938/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api" target="_blank" rel="noreferrer">llama.cpp decision API reference<ExternalLink size={12}/></a></div></div>
        </>}
      </div>
    </main>
    {showLogs && <div className="log-drawer"><div className="panel-heading"><span><Terminal size={16}/> Runtime activity</span><button className="icon-button" aria-label="Close runtime logs" onClick={() => setShowLogs(false)}><X size={17}/></button></div><pre>{snapshot?.logs.join('\n') || 'No runtime activity yet.'}</pre></div>}
  </div>;
}

createRoot(document.getElementById('root')!).render(<App/>);
