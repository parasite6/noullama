import { PendingActions, LatestRequest, retainEqual, usePageVisible } from './refresh-state';
import { MemoryWarningDialog, type CrowdedWarning } from './MemoryWarningDialog';
import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { Toggle } from './Toggle';
import { ModelDetails, type MemoryUse } from './ModelDetails';
import { ImageAttachments, MAX_TOTAL_BYTES, readImages, type Attachment } from './ImageAttachments';
import { Activity, AlertCircle, ArrowDownToLine, ArrowRight, BookOpen, Check, ChevronDown, ChevronRight, ChevronUp, CircleCheck, Code2, Copy, Cpu, Download, ExternalLink, FileJson, FlaskConical, FolderOpen, Gauge, Layers3, Mail, MemoryStick, Microchip, Network, Play, Plus, RefreshCw, Route, Search, Settings2, Shield, ShieldCheck, Square, Tags, Terminal, Trash2, Save, Power, Unplug, X, CircleArrowUp, Library as LibraryIcon } from 'lucide-react';
import { buildRequest, convertState, presets } from './presets';
import { QuitDialog } from './QuitDialog';
import { DeletePresetDialog } from './DeletePresetDialog';
import { CopyButton } from './CopyButton';
import { KeepContext } from './KeepContext';
import { ModelAliases, ModelRouting, type Route as ApiRoute } from './ModelNames';
import { InstanceCard } from './InstanceCard';
import { Select } from './Select';
import { Toast } from './Toast';
import { Library, SHA_CHECKS, type Companion, type LibraryEntry, type LibraryState } from './Library';
import { UpdatesDialog, type UpdateView } from './UpdatesDialog';
import { ContextMenu } from './ContextMenu';
import { NumberField } from './NumberField';
import { runtimeStatus } from './runtime-state';
import { modelStatus } from './model-state';
import { JsonEditor } from './JsonEditor';
import { DownloadPanel, type DownloadState } from './DownloadPanel';
import { AuthorLogo, authorFromPath } from './AuthorLogo';
import { HuggingFaceLogo } from './HuggingFaceLogo';
import { RuntimeManager, type RuntimeDownload } from './RuntimeManager';
import { instancesOf, playModelAfterLoad, reloadSlot, visibleBanner, waitText } from './ui-state';
import { containScrollChaining } from './scroll-contain';
import '@fontsource-variable/geist';
import '@fontsource-variable/geist-mono';
import './styles.css';

// 96 px stays sharp at 28 CSS px up to 3x scaling; the 960 px original
// decoded to 3.7 MB of bitmap for the sidebar.
const logoUrl = new URL('./logo-96.png', import.meta.url).href;

type Settings = { runtime_path: string; model_path: string; mmproj_path: string; mtp: boolean; mtp_path: string; decision_mode: string; context_size: number; gpu_layers: number; api_port: number; runtime_port: number; hf_path: string; download_dir: string; library_path: string; keep_context: boolean; sha256_check: string; allow_force_fit: boolean; hide_crowded_load_warning: boolean; fall_through_to_other_instances: boolean; any_available_when_busy: string; any_available_wait_ms: number };

// One loaded model, with the settings it was loaded with.
type Instance = { id: string; name: string; label: string; slot: number; suffix: string; default_suffix: string; api_names: string[]; instance_aliases: string[]; path: string; device: string; runtime_path: string; projector: boolean; mmproj_path: string; mtp: boolean; mtp_path: string; layout: string | null; ready: boolean; context_size: number; gpu_layers: number; required_bytes: number | null; context: { entries: number; last_flushed: number }; keep_context: boolean; settings: Settings };
type LoadError = { id: string; path: string; slot: number; name: string; message: string };
type Snapshot = { settings: Settings; running: boolean; ready: boolean; instances: Instance[]; default_model: string | null; routes: ApiRoute[]; library: LibraryState; api_port: number; api_error: string | null; api_notice: string | null; load_errors: LoadError[]; runtime_download: RuntimeDownload; quit_confirmation: boolean; startup_warning: string | null; logs: string[]; downloads: DownloadState[]; data_dir: string; download_dir: string; loaded_model: { path: string; device: string; runtime_path: string; projector: boolean; mtp: boolean; layout: string | null } | null; hardware: Hardware | null; fit: Fit | null; model_missing: boolean };
type Hardware = { ram_bytes: number; vram_bytes: number | null; gpu: string | null; cpu_cores: number };
type Fit = { level: number; label: string; detail: string; model_bytes: number; required_bytes: number };
type SavedPreset = { id: string; name: string; questions: unknown; state: string; state_mode: 'text' | 'json' };
type Model = { id: string; downloads: number; likes: number; tags: string[]; fit: Fit | null };
type ModelFile = { name: string; size?: number; projector: boolean; mtp_head: boolean; fit: Fit | null };
type Inspection = { repo: string; files: ModelFile[]; sha: string };
type Session = { authenticated: boolean; account: string | null; cli_available: boolean; token_present: boolean };
type Result = { answers?: Record<string, { type?: string; choice?: string; score?: number; noul?: number; confidence?: number; probabilities?: Record<string, number> }>; usage?: { input_tokens: number; output_tokens: number }; [key: string]: unknown };
// The Settings tab's choice for `any-available` when every ready instance is running a request.
const WHEN_BUSY = [
  { value: 'reject', label: 'Answer 503', detail: (_ms: number) => 'The request gets 503 at once, saying every loaded instance is busy.' },
  { value: 'wait', label: 'Wait for one', detail: (ms: number) => `The request waits for the first instance to finish, up to ${waitText(ms)}, then gets 503.` },
] as const;
// "Wait up to (ms)": the backend accepts 1–600000 (set_wait_ms in runtime.rs).
const WAIT_MS = { min: 1, max: 600000, step: 1000 };
const defaultSettings: Settings = { runtime_path: '', model_path: '', mmproj_path: '', mtp: false, mtp_path: '', decision_mode: 'auto', context_size: 8192, gpu_layers: 99, api_port: 11435, runtime_port: 11436, hf_path: 'hf', download_dir: '', library_path: '', keep_context: false, sha256_check: 'once', allow_force_fit: false, hide_crowded_load_warning: false, fall_through_to_other_instances: false, any_available_when_busy: 'reject', any_available_wait_ms: 120000 };
// The form is edited in two places, each saved on its own: the Local API
// load panel (settings a load uses) and the Runtime tab (build and ports).
// keep_context is the default a new instance starts with; loaded instances
// have their own switch, so it is not compared for Reload.
const LOAD_FIELDS = ['model_path', 'mmproj_path', 'mtp', 'mtp_path', 'decision_mode', 'context_size', 'gpu_layers', 'keep_context'] as const;
// Decision mode: stored value, UI label, hint under the segmented control.
const DECISION_MODES = [
  ['auto', 'Auto', 'Uses Prompt layout for Winnow GGUFs, otherwise Built-in when the model and runtime support it.'],
  ['native', 'Built-in', "The model's own decision support, answered by llama.cpp. The model and build must support it."],
  ['winnow', 'Prompt layout', 'Noullama prompts the model and reads the probability of each answer label (Winnow layout).'],
] as const;
const RUNTIME_FIELDS = ['runtime_path', 'library_path', 'api_port', 'runtime_port', 'hf_path'] as const;
const isLoadField = (key: keyof Settings) => (LOAD_FIELDS as readonly string[]).includes(key);
const pick = <K extends keyof Settings>(settings: Settings, keys: readonly K[]) => Object.fromEntries(keys.map(key => [key, settings[key]])) as Pick<Settings, K>;
const icons = [Route, ShieldCheck, Shield, Mail, Network, Plus];
const pages: Record<string, [string, string]> = {
  playground: ['Playground', 'Ask typed questions about a piece of content.'],
  models: ['Models', 'GGUF files on this machine.'],
  hub: ['Hub', 'Find and download decision models from Hugging Face.'],
  runtime: ['Runtime', 'llama.cpp builds, ports and the Hugging Face CLI.'],
  api: ['Local API', 'Load models and call them from your own code.'],
  library: ['Library', 'models.yml: the name, aliases, file, account and sha256 of every model.'],
  settings: ['Settings', 'Memory protection, API routing and model file checks.'],
};
const pretty = (v: unknown) => JSON.stringify(v, null, 2);
const errorText = (e: unknown) => e instanceof Error ? e.message : String(e);
const gb = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(1)} GB`;
const basename = (path: string) => path.split(/[\\/]/).pop() || 'No model loaded';
const mtpDetail = (s: { mtp_path: string } | null) => s ? s.mtp_path ? basename(s.mtp_path) : 'Built-in layers' : null;

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
  // Unsaved edits on the Runtime tab, and in the Local API load panel.
  const [runtimeFormDirty, setRuntimeFormDirty] = useState(false);
  const [loadDirty, setLoadDirty] = useState(false);
  const [preset, setPreset] = useState('triage');
  const [savedPresets, setSavedPresets] = useState<SavedPreset[]>([]);
  const [presetName, setPresetName] = useState('');
  const [presetToDelete, setPresetToDelete] = useState<SavedPreset | null>(null);
  // Keep context switches shown at once while set_instance_keep_context saves,
  // by instance id; the snapshot is the truth after.
  const [keepOverride, setKeepOverride] = useState<Record<string, boolean>>({});
  const [dismissedApi, setDismissedApi] = useState<string | null>(null);
  const [dismissedApiNotice, setDismissedApiNotice] = useState(false);
  // The loaded model the Playground runs on; the default model when unset or ejected.
  const [playModel, setPlayModel] = useState<string | null>(null);
  // Which loaded instance of the selected file the load bar's Eject and Reload act on.
  const [loadTarget, setLoadTarget] = useState<string | null>(null);
  // Model files whose alias editor is open on the Models page.
  const [aliasEditors, setAliasEditors] = useState<string[]>([]);
  const [dismissedStartup, setDismissedStartup] = useState(false);
  const [previewDismissed, setPreviewDismissed] = useState(false);
  // Downloads whose request is still being checked, by repo/file id.
  const [preparing, setPreparing] = useState<string[]>([]);
  const [questions, setQuestions] = useState(pretty(presets[0].questions));
  const [content, setContent] = useState(presets[0].state);
  const [stateMode, setStateMode] = useState<'text' | 'json'>('text');
  const [images, setImages] = useState<Attachment[]>([]);
  const [dropping, setDropping] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  // `replace`: the loaded instance the interrupted load was reloading, so Eject-and-retry reloads it too.
  const [crowdedWarning, setCrowdedWarning] = useState<(CrowdedWarning & { replace: string | null }) | null>(null);
  const [hideCrowded, setHideCrowded] = useState(false);
  const [memory, setMemory] = useState<MemoryUse | null>(null);
  const [stateMemory, setStateMemory] = useState<{ json: string; text: string } | null>(null);
  const [downloadDir, setDownloadDir] = useState('');
  const [result, setResult] = useState<Result | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [resultTab, setResultTab] = useState('decisions');
  const actionOwners = useRef(new PendingActions());
  const refreshRequests = useRef(new LatestRequest());
  const [pendingActions, setPendingActions] = useState<Set<string>>(new Set());
  const isBusy = (...names: string[]) => names.some(name => pendingActions.has(name));
  const modelPending = isBusy('start', 'save', 'save-load') || [...pendingActions].some(name => name.startsWith('eject:'));
  const [notice, setNotice] = useState<{ text: string; error: boolean } | null>(null);
  const [showLogs, setShowLogs] = useState(false);
  const [query, setQuery] = useState('');
  const [searched, setSearched] = useState('');
  const [toastDismissed, setToastDismissed] = useState<string[]>([]);
  const [toastCollapsed, setToastCollapsed] = useState(false);
  // The downloads card's height, so the page can keep its last controls clear of it.
  const toastRef = useRef<HTMLElement>(null);
  const [toastHeight, setToastHeight] = useState(0);
  const [models, setModels] = useState<Model[]>([]);
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [scanTick, setScanTick] = useState(0);
  const [localModels, setLocalModels] = useState<{ path: string; name: string; size: number; mtp?: boolean; mtp_head?: boolean; decision?: string | null }[]>([]);
  const [mtpResult, setMtpResult] = useState<{ path: string; built_in: boolean; head: string | null }>({ path: '', built_in: false, head: null });
  const selectedMtp = mtpResult.path === settings.model_path ? mtpResult : { built_in: false, head: null };
  const [scanWarnings, setScanWarnings] = useState<string[]>([]);
  const [session, setSession] = useState<Session | null>(null);
  const [token, setToken] = useState('');
  const [version, setVersion] = useState('');
  const [copied, setCopied] = useState('');
  const [updates, setUpdates] = useState<UpdateView | null>(null);
  const [updatesOpen, setUpdatesOpen] = useState(false);
  const desktop = isTauri();
  const pageVisible = usePageVisible();
  // Rescan local models when a download finishes.
  const finishedDownloads = (snapshot?.downloads ?? []).map(d => d.path ?? '').filter(Boolean).join('\n');

  useEffect(() => { window.scrollTo({ top: 0, behavior: 'instant' }); }, [tab]);
  useEffect(containScrollChaining, []);
  useEffect(() => {
    const stay = (event: DragEvent) => { if (event.dataTransfer?.types.includes('Files')) event.preventDefault(); };
    window.addEventListener('dragover', stay);
    window.addEventListener('drop', stay);
    return () => { window.removeEventListener('dragover', stay); window.removeEventListener('drop', stay); };
  }, []);
  useEffect(() => {
    if (!detailsOpen || !desktop || !pageVisible) return;
    let live = true;
    const read = () => invoke<MemoryUse>('model_memory').then(value => { if (live) setMemory(value); }).catch(() => {});
    read();
    const timer = setInterval(read, 2000);
    return () => { live = false; clearInterval(timer); };
  }, [detailsOpen, pageVisible]);

  useEffect(() => { if (!desktop || tab !== 'models' || !pageVisible) return; const timer = setInterval(() => setScanTick(t => t + 1), 5000); return () => clearInterval(timer); }, [tab, pageVisible]);
  async function refresh() {
    if (!desktop) return;
    const request = refreshRequests.current.begin();
    await Promise.all([
      invoke<Snapshot>('snapshot').then(next => {
        if (refreshRequests.current.isLatest(request)) setSnapshot(current => retainEqual(current, next));
      }),
      invoke<UpdateView>('update_state').then(next => {
        if (refreshRequests.current.isLatest(request)) setUpdates(current => retainEqual(current, next));
      }),
    ]);
  }
  async function updateCommand(command: string, args?: Record<string, unknown>) {
    try { const next = await invoke<UpdateView | null>(command, args); if (next) setUpdates(next); }
    // Check and install failures show in the dialog from update_state; a
    // preference that could not be saved has nowhere else to appear.
    catch (e) { if (command === 'set_update_prefs') setNotice({ text: errorText(e), error: true }); }
    await refresh().catch(() => {});
  }
  useEffect(() => {
    if (!desktop) return;
    invoke<SavedPreset[]>('list_presets').then(setSavedPresets).catch(e => setNotice({ text: errorText(e), error: true }));
    refresh().catch(e => setNotice({ text: errorText(e), error: true }));
  }, []);
  // Hidden in the tray, nothing on screen needs the poll; the backend watches
  // loaded runtimes on its own. Coming back shows current state at once.
  useEffect(() => {
    if (!desktop || !pageVisible) return;
    refresh().catch(() => {});
    const timer = setInterval(() => { refresh().catch(() => {}); }, 1200);
    return () => clearInterval(timer);
  }, [pageVisible]);
  // The form follows the saved settings, except the group that has unsaved edits.
  const savedSettings = snapshot?.settings ?? null;
  const savedKey = savedSettings ? JSON.stringify(savedSettings) : '';
  useEffect(() => {
    if (!savedSettings) return;
    setSettings(current => ({ ...savedSettings, ...(runtimeFormDirty ? pick(current, RUNTIME_FIELDS) : {}), ...(loadDirty ? pick(current, LOAD_FIELDS) : {}) }));
  }, [savedKey, runtimeFormDirty, loadDirty]);
  // Editing a group back to its saved values leaves nothing to save.
  useEffect(() => {
    if (!savedSettings) return;
    const unchanged = (keys: readonly (keyof Settings)[]) => keys.every(key => settings[key] === savedSettings[key]);
    if (loadDirty && unchanged(LOAD_FIELDS)) setLoadDirty(false);
    if (runtimeFormDirty && unchanged(RUNTIME_FIELDS)) setRuntimeFormDirty(false);
  }, [settings, savedKey, loadDirty, runtimeFormDirty]);
  useEffect(() => { if (snapshot) setDownloadDir(snapshot.settings.download_dir); }, [snapshot?.settings.download_dir]);
  useEffect(() => {
    const none = { path: settings.model_path, built_in: false, head: null };
    if (!desktop || !settings.model_path) { setMtpResult(none); return; }
    let live = true;
    invoke<typeof none>('model_mtp', { path: settings.model_path }).then(found => { if (live) setMtpResult(current => retainEqual(current, { ...found, path: settings.model_path })); }).catch(() => { if (live) setMtpResult(none); });
    return () => { live = false; };
  }, [settings.model_path, scanTick]);
  useEffect(() => { if (desktop && tab === 'hub' && !session) invoke<Session>('hf_session').then(setSession).catch(e => setNotice({ text: errorText(e), error: true })); }, [tab]);
  useEffect(() => { if (desktop && tab !== 'hub') invoke<{ models: typeof localModels; warnings: string[] }>('local_models').then(scan => { setLocalModels(scan.models); setScanWarnings(scan.warnings); }).catch(e => setNotice({ text: errorText(e), error: true })); }, [tab, finishedDownloads, scanTick, snapshot?.download_dir, settings.model_path]);
  // "Wait up to (ms)" saves on its own, once typing pauses, so half-typed
  // numbers are not sent. The draft shows until the saved value comes back.
  const whenBusy = snapshot?.settings.any_available_when_busy ?? 'reject';
  const savedWait = snapshot?.settings.any_available_wait_ms ?? 120000;
  const [waitDraft, setWaitDraft] = useState<number | null>(null);
  useEffect(() => {
    if (waitDraft === null) return;
    if (waitDraft === savedWait) { setWaitDraft(null); return; }
    const value = waitDraft;
    const timer = setTimeout(() => void action(`wait-ms:${value}`, async () => { await invoke('set_any_available_wait_ms', { ms: value }); })
      .then(() => setWaitDraft(draft => draft === value ? null : draft)), 600);
    return () => clearTimeout(timer);
  }, [waitDraft]);
  async function action(name: string, task: () => Promise<void>) {
    if (!desktop) { setNotice({ text: 'Launch the desktop app with npm run desktop to use the local backend.', error: true }); return; }
    if (!actionOwners.current.begin(name)) return;
    setPendingActions(actionOwners.current.snapshot());
    try { await task(); await refresh(); }
    catch (e) { setNotice({ text: errorText(e), error: true }); }
    finally { actionOwners.current.end(name); setPendingActions(actionOwners.current.snapshot()); }
  }
  function edit<K extends keyof Settings>(key: K, value: Settings[K]) { setSettings(s => ({ ...s, [key]: value })); if (isLoadField(key)) setLoadDirty(true); else setRuntimeFormDirty(true); }
  const isProjector = (path: string) => basename(path).toLowerCase().includes('mmproj');
  const folder = (path: string) => path.replace(/[\\/][^\\/]*$/, '');
  // A projector or MTP head belongs to the model in the same folder; switching models drops one that does not.
  // Built-in MTP layers belong to one file, so a model switch turns MTP off unless its head file stays valid.
  function selectModel(path: string) {
    edit('model_path', path);
    if (settings.mmproj_path && folder(settings.mmproj_path) !== folder(path) && settings.mmproj_path !== libraryEntries.find(e => e.path === path)?.mmproj) edit('mmproj_path', '');
    if (path !== settings.model_path) { edit('mtp', false); edit('mtp_path', ''); }
  }
  function setMtp(on: boolean) {
    edit('mtp', on);
    edit('mtp_path', on && !selectedMtp.built_in ? mtpHead : '');
  }
  function addImages(files: File[]) {
    if (!files.length) return;
    readImages(files).then(added => {
      const total = [...images, ...added].reduce((sum, image) => sum + image.bytes, 0);
      if (total > MAX_TOTAL_BYTES) throw new Error('These images add up to more than 60 MB. Remove some or use smaller files.');
      setImages(current => [...current, ...added]); setResult(null);
    }).catch(e => setNotice({ text: errorText(e), error: true }));
  }
  // Each save sends only its own group; the backend applies it over the
  // settings saved at that moment, so unsaved edits in the other place are
  // neither saved nor lost, and two saves close together both stick. The
  // refresh comes first so the form does not flash back to the old values.
  async function saveGroup(keys: readonly (keyof Settings)[], clean: () => void) {
    await invoke('save_settings', { settings: pick(settings, keys) });
    await refresh(); clean();
  }
  const saveLoadSettings = () => saveGroup(LOAD_FIELDS, () => setLoadDirty(false));
  const saveRuntimeSettings = () => saveGroup(RUNTIME_FIELDS, () => setRuntimeFormDirty(false));
  async function run() {
    await action('decide', async () => {
      const payload = { ...buildRequest(questions, content, stateMode), ...(visionLoaded && images.length ? { images: images.map(image => image.url) } : {}) };
      const start = performance.now();
      const response = await invoke<Result>('decide', { payload, model: activeInstance?.id ?? null });
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
    setPresetToDelete(null);
  }
  const copyError = (text: string) => setNotice({ text, error: true });
  // One loaded instance's Keep context; off also clears that instance's history.
  async function toggleKeepContext(id: string, on: boolean) {
    if (!desktop) { setNotice({ text: 'Launch the desktop app with npm run desktop to use the local backend.', error: true }); return; }
    setKeepOverride(current => ({ ...current, [id]: on }));
    try { await invoke('set_instance_keep_context', { id, on }); await refresh(); }
    catch (e) { setNotice({ text: errorText(e), error: true }); }
    finally { setKeepOverride(current => { const next = { ...current }; delete next[id]; return next; }); }
  }
  async function clearContext(id: string) { await action(`clear-context:${id}`, async () => { await invoke('clear_context', { id }); }); }
  async function saveAliases(path: string, aliases: string[]) {
    if (!desktop) throw new Error('Launch the desktop app with npm run desktop to use the local backend.');
    await invoke('set_model_aliases', { path, aliases }); await refresh();
  }
  async function saveLibrary(entries: LibraryEntry[], revision: number) {
    if (!desktop) throw new Error('Launch the desktop app with npm run desktop to use the local backend.');
    await invoke('save_library', { entries, revision }); await refresh();
    setNotice({ text: 'models.yml saved. The previous version was backed up.', error: false });
  }
  async function libraryCommand(command: string, args?: Record<string, unknown>) {
    if (!desktop) throw new Error('Launch the desktop app with npm run desktop to use the local backend.');
    await invoke(command, args); await refresh();
  }
  async function setShaCheck(mode: string) { await action('sha-check', async () => { await invoke('set_sha256_check', { mode }); }); }
  async function eject(id: string) { await action(`eject:${id}`, async () => { await invoke('unload_model', { id }); }); }
  async function dismissLoadError(id: string) { if (desktop) { await invoke('dismiss_load_error', { id }).catch(() => {}); await refresh().catch(() => {}); } }
  async function saveInstanceAliases(id: string, aliases: string[]) {
    if (!desktop) throw new Error('Launch the desktop app with npm run desktop to use the local backend.');
    await invoke('set_instance_aliases', { id, aliases }); await refresh();
  }
  async function saveSuffix(id: string, suffix: string) {
    if (!desktop) throw new Error('Launch the desktop app with npm run desktop to use the local backend.');
    await invoke('set_instance_suffix', { id, suffix }); await refresh();
  }
  // `openTab`: where to go once a file was picked (the load panel, from the Models tab).
  async function browseFile(field: 'model_path' | 'mmproj_path' | 'mtp_path', openTab?: string) {
    await action('browse', async () => {
      const path = await invoke<string | null>('pick_model_file');
      if (path && openTab) setTab(openTab);
      if (path && field === 'model_path') selectModel(path);
      else if (path && field === 'mtp_path') { edit('mtp_path', path); edit('mtp', true); }
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
  async function cancelDownload(id: string) { await action('cancel', async () => { await invoke('cancel_download', { id }); }); }
  async function dismissDownload(id: string) {
    setToastDismissed(current => [...current, id]);
    if (desktop) await invoke('dismiss_download', { id }).catch(() => {});
  }
  // Track download preparation by file so other downloads stay available.
  async function startDownload(repo: string, file: string) {
    const id = `${repo}/${file}`;
    if (!desktop) { setNotice({ text: 'Launch the desktop app with npm run desktop to use the local backend.', error: true }); return; }
    setPreparing(current => [...current, id]); setToastDismissed(current => current.filter(other => other !== id)); setToastCollapsed(false); setNotice(null);
    try { await invoke('download_model', { repo, filename: file }); await refresh(); }
    catch (e) { setNotice({ text: errorText(e), error: true }); }
    finally { setPreparing(current => current.filter(other => other !== id)); }
  }
  function switchStateMode(next: 'text' | 'json') {
    try {
      const converted = convertState(content, stateMode, next, stateMemory?.json, stateMemory?.text);
      if (stateMode === 'json' && next === 'text') setStateMemory({ json: content, text: converted });
      setContent(converted); setStateMode(next);
    } catch (e) { setNotice({ text: errorText(e), error: true }); }
  }
  // Loads the selected model beside any already loaded, after saving the load
  // panel. `replace` is the loaded instance being reloaded: only it is ejected,
  // by id, just before the new load. `ejectOther` comes from the crowded-load
  // dialog. The new instance becomes the Playground's model unless another
  // loaded model is already chosen there.
  async function loadSelected({ replace = null, ejectOther }: { replace?: string | null; ejectOther?: string } = {}) {
    await action('start', async () => {
      await saveLoadSettings();
      if (ejectOther) await invoke('unload_model', { id: ejectOther });
      // The instance being reloaded is unloaded first, so the check leaves it out.
      const warning = await invoke<CrowdedWarning | null>('crowded_load_warning', { replace });
      if (warning) { setHideCrowded(false); setCrowdedWarning({ ...warning, replace }); return; }
      setCrowdedWarning(null);
      // It may have stopped on its own meanwhile; then there is nothing to eject.
      const slot = reloadSlot(instances, replace);
      if (replace) await invoke('unload_model', { id: replace }).catch(() => {});
      const id = await invoke<string>('start_runtime', { slot });
      setLoadTarget(id);
      setPlayModel(current => playModelAfterLoad(current, id, instances, replace, ejectOther));
    });
  }
  async function saveDownloadDirectory() {
    await action('directory', async () => {
      if (!snapshot) return;
      await invoke('save_settings', { settings: { download_dir: downloadDir } });
      setSettings(current => ({ ...current, download_dir: downloadDir }));
      setNotice({ text: 'Download directory saved. Existing files stay in place.', error: false });
    });
  }
  async function copy(value: string, id: string) {
    try { await navigator.clipboard.writeText(value); setCopied(id); setTimeout(() => setCopied(''), 1800); }
    catch (e) { setNotice({ text: `Could not copy: ${errorText(e)}`, error: true }); }
  }
  const banner = visibleBanner(notice ?? (!dismissedApiNotice && snapshot?.api_notice ? { text: snapshot.api_notice, error: false } : null) ?? (!dismissedStartup && snapshot?.startup_warning ? { text: snapshot.startup_warning, error: true } : null), snapshot?.api_error, dismissedApi, desktop, previewDismissed);
  const instances = snapshot?.instances ?? [];
  const running = instances.length > 0;
  const readyCount = instances.filter(i => i.ready).length;
  const modelMissing = !loadDirty && !!snapshot?.model_missing;
  // The selected file's loaded model, if it is loaded.
  // Several instances of one file can be loaded; the load bar acts on one, chosen in the panel.
  const { list: selectedInstances, selected: selectedInstance } = instancesOf(instances, settings.model_path, loadTarget);
  const model = modelStatus(settings.model_path, selectedInstance, modelMissing);
  // Whether a load now would differ from what this instance was loaded with:
  // the load panel's values, plus the saved build (unsaved Runtime tab edits
  // are not used by a load).
  const differsFrom = (instance: Instance) => instance.path === settings.model_path && (
    (['mmproj_path', 'mtp', 'mtp_path', 'decision_mode', 'context_size', 'gpu_layers'] as const).some(key => settings[key] !== instance.settings[key])
    || (['runtime_path', 'library_path'] as const).some(key => (savedSettings ?? defaultSettings)[key] !== instance.settings[key]));
  const reloadSelected = !!selectedInstance && differsFrom(selectedInstance);
  // The fit leaves out one instance of the selected file, the one a reload replaces.
  const otherLoaded = instances.length - (selectedInstance ? 1 : 0);
  const fit = snapshot?.fit ?? null;
  const fitDetail = fit ? `${fit.detail}${otherLoaded ? ` Counts the memory the ${otherLoaded === 1 ? 'other loaded model holds' : `${otherLoaded} other loaded models hold`}.` : ''}` : '';
  const defaultInstance = instances.find(i => i.id === snapshot?.default_model) ?? null;
  const activeInstance = instances.find(i => i.id === playModel) ?? defaultInstance ?? instances[0] ?? null;
  const ready = !!activeInstance?.ready;
  const visionLoaded = ready && !!activeInstance?.projector;
  const projector = settings.mmproj_path || (snapshot?.library?.entries ?? []).find(e => e.path === settings.model_path)?.mmproj || localModels.find(file => isProjector(file.path) && folder(file.path) === folder(settings.model_path))?.path || '';
  const mtpHead = settings.mtp_path || selectedMtp.head || '';
  const builtInMtp = selectedMtp.built_in;
  const mtpAvailable = builtInMtp || !!mtpHead;
  const mtpTitle = settings.mtp ? settings.mtp_path ? `Uses ${basename(settings.mtp_path)}` : "Uses the model's built-in MTP layers" : builtInMtp ? "Use the model's built-in MTP layers" : `Load ${basename(mtpHead)} with this model`;
  const modelFiles = localModels.filter(file => !isProjector(file.path) && !file.mtp_head);
  const loadErrors = snapshot?.load_errors ?? [];
  const hardware = snapshot?.hardware ?? null;
  const chosenRuntime = runtimeStatus(settings.runtime_path, selectedInstance?.runtime_path ?? null, !!selectedInstance, snapshot?.data_dir ?? '');
  const runtimeName = (path: string) => { const r = runtimeStatus(path, null, false, snapshot?.data_dir ?? ''); return path ? `${r.source} · ${r.name}` : r.name; };
  // The build a load uses: the saved one.
  const loadRuntime = runtimeName(savedSettings?.runtime_path ?? '');
  const keepsContext = (instance: Instance) => keepOverride[instance.id] ?? instance.keep_context;
  const playContext = activeInstance?.context ?? { entries: 0, last_flushed: 0 };
  const libraryEntries = snapshot?.library?.entries ?? [];
  const aliasesFor = (path: string) => libraryEntries.find(e => e.path === path)?.aliases ?? [];
  // The avatar account: models.yml first, then the download folder's name.
  const accountFor = (path: string) => libraryEntries.find(e => e.path === path)?.account || authorFromPath(path);
  const nameFor = (path: string) => libraryEntries.find(e => e.path === path)?.name;
  const routes = snapshot?.routes ?? [];
  // Shown in the sidebar card: the Playground's model, else the selection.
  const shown = activeInstance;
  const apiOrigin = `http://127.0.0.1:${snapshot?.api_port ?? 11435}`;
  const apiBase = `${apiOrigin}/v1`;
  // Every request names its model exactly; examples use a real name when one is loaded.
  // An instance's identifiers: its file's names and aliases plus its suffix.
  const apiName = (instance: Instance) => instance.api_names[0] ?? `${instance.name}${instance.suffix}`;
  const usable = (instance: Instance) => routes.filter(r => r.instance === instance.id && !r.fallthrough && !r.conflict);
  const exampleFor = (instance: Instance | null) => instance ? (usable(instance).find(r => r.kind === 'alias') ?? usable(instance)[0])?.name ?? null : null;
  const exampleModel = exampleFor(activeInstance) || 'your-model-name';
  const pythonExample = `import json\nfrom openai import OpenAI\n\nclient = OpenAI(base_url="${apiBase}", api_key="local")\n\nrequest = {\n    "state": "I was charged twice. Please refund the duplicate.",\n    "questions": {\n        "refund": {\n            "type": "noul",\n            "instructions": "Is the customer requesting a refund?"\n        }\n    }\n}\n\nresponse = client.chat.completions.create(\n    model="${exampleModel}",\n    messages=[{"role": "user", "content": json.dumps(request)}],\n    stream=False,\n)\ndecision = json.loads(response.choices[0].message.content)\nprint(decision["answers"])`;
  let questionError = '';
  try { buildRequest(questions, '', 'text'); } catch (e) { questionError = errorText(e); }
  let jsonError = '';
  try { buildRequest(questions, content, stateMode); } catch (e) { jsonError = errorText(e); }
  const requestJson = (() => { try { return pretty({ model: exampleModel, ...buildRequest(questions, content, stateMode) }); } catch { return '{}'; } })();

  const [pageTitle, pageSubtitle] = pages[tab];
  const repoLike = /^[\w.-]+\/[\w.-]+$/.test(query.trim()) && !models.some(m => m.id === query.trim());
  const downloads = snapshot?.downloads ?? [];
  const downloadFor = (repo: string, file: string) => downloads.find(d => d.id === `${repo}/${file}`);
  const anyDownloadActive = downloads.some(d => d.active);
  // Downloads shown in the floating list: running or finished, not dismissed,
  // and not already on screen in the repository being viewed.
  const toastDownloads = downloads.filter(d => (d.active || (!!d.path && !d.error && !d.cancelled)) && !toastDismissed.includes(d.id) && !(tab === 'hub' && inspection?.repo === d.repo));
  const showDownloadToast = toastDownloads.length > 0 && !showLogs;
  const hubRow = (group: string, id: string, meta: React.ReactNode, fitInfo: Fit | null, fitTitle?: string) => <button key={`${group}:${id}`} className={`hub-row ${inspection?.repo === id ? 'selected' : ''}`} onClick={() => openRepo(id)}><AuthorLogo author={id.split('/')[0]} size={34}/><span className="hub-row-name"><strong>{id.split('/').slice(1).join('/') || id}</strong><small>{id.split('/')[0]}</small></span><span className="hub-row-meta">{meta}</span>{fitInfo && <span className={`fit-dot fit-level-${fitInfo.level}`} title={fitTitle ?? fitInfo.detail}><span className="fit-bars" aria-hidden="true">{[1, 2, 3, 4].map(n => <i key={n} className={n <= fitInfo.level ? 'on' : ''}/>)}</span><span className="sr-only">{fitInfo.label}</span></span>}</button>;

  useEffect(() => {
    const card = toastRef.current;
    if (!showDownloadToast || !card) { setToastHeight(0); return; }
    const observer = new ResizeObserver(() => setToastHeight(card.offsetHeight));
    observer.observe(card);
    return () => observer.disconnect();
  }, [showDownloadToast]);

  return <div className={`app-shell ${showDownloadToast ? 'has-toast' : ''}`} style={{ '--toast-h': `${toastHeight}px` } as React.CSSProperties}>
    <aside className="sidebar">
      <div className="brand"><img className="brand-logo" src={logoUrl} width={28} height={28} alt=""/><span>noullama</span><span className="brand-tag">Beta</span></div>
      <nav>{[
        ['playground', FlaskConical, 'Playground'], ['models', Layers3, 'Models'], ['hub', Network, 'Hub'], ['runtime', Cpu, 'Runtime'], ['api', Code2, 'Local API'], ['library', LibraryIcon, 'Library'], ['settings', Settings2, 'Settings'],
      ].map(([id, Icon, title]) => { const I = Icon as typeof Cpu; return <button key={String(id)} className={`nav-item ${tab === id ? 'active' : ''}`} aria-current={tab === id ? 'page' : undefined} title={String(title)} onClick={() => setTab(String(id))}><I size={18}/><span>{String(title)}</span></button>; })}</nav>
      <div className="sidebar-bottom">
        {loadErrors.length > 0 && <button className="nav-item load-error-nav" title={`${loadErrors.length === 1 ? `Load error: ${loadErrors[0].name}` : `${loadErrors.length} load errors`}. Open Local API`} aria-label={`${loadErrors.length === 1 ? '1 load error' : `${loadErrors.length} load errors`}, open Local API`} onClick={() => setTab('api')}><AlertCircle size={18}/><span>{loadErrors.length === 1 ? 'Load error' : `${loadErrors.length} load errors`}</span></button>}
        {desktop &&<button className={`nav-item ${updates?.offer ? 'has-update' : ''}`} title={updates?.offer ? `Update to ${updates.offer.version}` : 'Updates'} aria-haspopup="dialog" onClick={() => setUpdatesOpen(true)}><CircleArrowUp size={18}/><span>{updates?.offer ? `Update to ${updates.offer.version}` : 'Updates'}</span></button>}
        <button className="nav-item" title="Runtime logs" aria-label="Runtime logs" aria-pressed={showLogs} onClick={() => setShowLogs(v => !v)}><Terminal size={18}/><span>Runtime logs</span></button>
        <button className="nav-item" title="Quit Noullama" aria-label="Quit Noullama" disabled={!desktop || isBusy('quit')} onClick={() => action('quit-prompt', async () => { await invoke('request_quit'); })}><Power size={18}/><span>Quit Noullama</span></button>
        <div className="runtime-card-wrap" onMouseEnter={() => setDetailsOpen(true)} onMouseLeave={() => setDetailsOpen(false)} onFocus={() => setDetailsOpen(true)} onBlur={() => setDetailsOpen(false)}>
          <button className="runtime-card" onClick={() => setTab('runtime')} aria-label={`Selected llama.cpp: ${chosenRuntime.source} ${chosenRuntime.name}${chosenRuntime.inUse ? ', in use' : ', not running'}`}><span className={`runtime-card-icon ${readyCount ? 'ready' : running ? 'loading' : ''}`}><Cpu size={16}/></span><span><strong>{readyCount > 1 ? `${readyCount} models ready` : readyCount ? 'Model ready' : running ? 'Loading model' : 'Runtime offline'}</strong><small>llama.cpp {chosenRuntime.source} · {chosenRuntime.name}</small>{chosenRuntime.pending && <small className="runtime-pending">Next selection not applied</small>}</span></button>
          {detailsOpen && <ModelDetails
            name={basename(shown?.path || settings.model_path || '') }
            status={shown ? `${shown.ready ? 'Ready' : 'Loading'}${instances.length > 1 ? ` · 1 of ${instances.length} loaded` : ''}` : model.selected ? 'Not loaded' : 'No model selected'}
            path={shown?.path || settings.model_path}
            device={shown?.device || null}
            projector={shown ? shown.projector ? basename(shown.mmproj_path) : null : settings.mmproj_path ? basename(settings.mmproj_path) : null}
            mtp={mtpDetail(shown ? shown.mtp ? shown : null : settings.mtp ? settings : null)}
            contextSize={shown?.context_size ?? settings.context_size}
            gpuLayers={shown?.gpu_layers ?? settings.gpu_layers}
            runtime={(() => { const r = runtimeStatus(shown?.runtime_path ?? settings.runtime_path, null, false, snapshot?.data_dir ?? ''); return `${r.source} / ${r.name}`; })()}
            endpoint={shown?.ready ? apiBase : null}
            memory={desktop ? memory : null}
            estimate={shown ? shown.required_bytes : fit?.required_bytes ?? null}
            modelId={shown?.id ?? null}
            loadedCount={instances.length}
            gpuName={hardware?.gpu ?? null}/>}
        </div>
        {instances.length > 0 && <ul className="sidebar-loaded" aria-label="Loaded models">{instances.map(instance => <li key={instance.id}>
          <span className={`status-dot ${instance.ready ? 'green' : 'amber'}`} title={instance.ready ? 'Ready' : 'Loading'}/>
          <button type="button" className="sidebar-loaded-name" title={`${apiName(instance)}: ${instance.path}`} onClick={() => { setPlayModel(instance.id); setTab('api'); }}>{apiName(instance)}</button>
          <button type="button" className="icon-button" aria-label={`Eject ${apiName(instance)}`} title={`Eject ${apiName(instance)}`} disabled={isBusy(`eject:${instance.id}`)} onClick={() => void eject(instance.id)}><Unplug size={14}/></button>
        </li>)}</ul>}
      </div>
    </aside>
    <main>
      <div className={`page-content page-${tab}`}>
        <header className="page-heading">
          <div><h1>{pageTitle}</h1><p>{pageSubtitle}</p></div>
          {hardware && <div className="stat-chips" aria-label="This machine">{hardware.vram_bytes ? <span className="stat-chip" title={hardware.gpu ?? 'Graphics memory'}><Gauge size={14}/><strong>{gb(hardware.vram_bytes)}</strong>VRAM</span> : <span className="stat-chip"><Gauge size={14}/>No GPU memory</span>}{hardware.ram_bytes > 0 && <span className="stat-chip"><MemoryStick size={14}/><strong>{gb(hardware.ram_bytes)}</strong>RAM</span>}{hardware.cpu_cores > 0 && <span className="stat-chip"><Microchip size={14}/><strong>{hardware.cpu_cores}</strong>CPU threads</span>}</div>}
        </header>
        <Toast banner={banner} onDismiss={() => { setNotice(null); setDismissedStartup(true); setDismissedApiNotice(true); setDismissedApi(snapshot?.api_error ?? null); setPreviewDismissed(true); }}/>

        {tab === 'playground' && <div className="playground">
          <div className="playground-main">
            <section className={`panel state-panel ${dropping ? 'dropping' : ''}`} onDragOver={event => { if (!visionLoaded || !event.dataTransfer.types.includes('Files')) return; event.preventDefault(); setDropping(true); }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setDropping(false); }} onDrop={event => { if (!visionLoaded) return; event.preventDefault(); setDropping(false); addImages([...event.dataTransfer.files]); }}><div className="panel-heading"><span><Terminal size={16}/> State <span className="muted normal-weight">The content to evaluate</span></span><div className="segmented"><button className={stateMode === 'text' ? 'on' : ''} onClick={() => switchStateMode('text')}>Text</button><button className={stateMode === 'json' ? 'on' : ''} onClick={() => switchStateMode('json')}>JSON</button></div></div>{stateMode === 'json' ? <div className="state-json"><JsonEditor label="State JSON" value={content} onChange={value => { setContent(value); setResult(null); }}/></div> : <textarea aria-label="State content" value={content} spellCheck={false} placeholder={visionLoaded ? 'Content to evaluate, or leave empty to ask about the images only…' : 'Content to evaluate…'} onPaste={event => { const files = [...event.clipboardData.files].filter(file => file.type.startsWith('image/')); if (visionLoaded && files.length) { if (!event.clipboardData.types.includes('text/plain')) event.preventDefault(); addImages(files); } }} onChange={e => { setContent(e.target.value); setResult(null); }}/>}{visionLoaded && <ImageAttachments images={images} onFiles={addImages} onRemove={index => { setImages(current => current.filter((_, i) => i !== index)); setResult(null); }}/>}<div className="state-footer"><span>{content.length.toLocaleString()} characters<span className="separator">/</span>{stateMode === 'text' ? 'Plain text' : 'Structured JSON'}{visionLoaded && images.length > 0 && <><span className="separator">/</span>{images.length} {images.length === 1 ? 'image' : 'images'}</>}</span><div className="state-run"><button className="text-button" onClick={() => { setContent(''); setImages([]); setResult(null); }}>Clear</button><button className="primary" disabled={isBusy('decide') || !!jsonError || !ready} onClick={run}>{isBusy('decide') ? <RefreshCw className="spin" size={15}/> : <Play size={15}/>} {isBusy('decide') ? 'Deciding…' : 'Run decision'}</button></div></div></section>
            {(jsonError || !ready) && <p className="run-hint">{jsonError ? <span className="invalid">{jsonError}</span> : 'Load a model to run a decision.'}</p>}
            <section className="panel results-panel"><div className="panel-heading"><span><Activity size={16}/> Result {result && <span className="chip success">Complete</span>}</span><div className="heading-actions">{result && <span className="timing">{elapsed.toFixed(1)} ms</span>}<div className="segmented"><button className={resultTab === 'decisions' ? 'on' : ''} onClick={() => setResultTab('decisions')}>Decisions</button><button className={resultTab === 'json' ? 'on' : ''} onClick={() => setResultTab('json')}>JSON</button></div>{result && <button className="icon-button" aria-label="Copy result" onClick={() => copy(pretty(result), 'result')}>{copied === 'result' ? <Check size={14}/> : <Copy size={14}/>}</button>}</div></div>{!result ? <div className="empty-result"><Activity size={17}/><p>{ready ? 'Run a decision to view results.' : 'Results appear here after a run.'}</p></div> : resultTab === 'json' ? <div className="result-code"><JsonEditor label="Result JSON" value={pretty(result)} readOnly/></div> : <div className="decision-results">{Object.entries(result.answers ?? {}).map(([name, answer]) => <div className="answer" key={name}><div className="answer-heading"><span>{name}</span><strong>{answer.choice ?? (answer.noul !== undefined ? answer.noul >= 0.5 ? `${(answer.noul * 100).toFixed(1)}% yes` : `${((1 - answer.noul) * 100).toFixed(1)}% no` : answer.score?.toFixed(2))}</strong>{answer.confidence !== undefined && <small>{(answer.confidence * 100).toFixed(1)}% confidence</small>}</div>{Object.entries(answer.probabilities ?? (answer.noul !== undefined ? { yes: answer.noul, no: 1 - answer.noul } : {})).map(([key, value]) => <div className="probability" key={key}><span>{key}</span><div><i style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}/></div><code>{(value * 100).toFixed(1)}%</code></div>)}</div>)}<div className="usage">{result.usage?.input_tokens.toLocaleString()} input tokens <span>/</span> {result.usage?.output_tokens ?? 0} generated tokens <span>/</span> {elapsed.toFixed(1)} ms round trip</div></div>}</section>
          </div>
          <aside className="run-settings" aria-label="Run settings">
            <h2>Run settings</h2>
            <div className="rail-section"><div className="rail-label">Model</div>{instances.length ? <Select label="Playground model" value={activeInstance?.id ?? ''} options={instances.map(i => ({ value: i.id, label: `${apiName(i)}${i.slot > 1 ? ` · instance ${i.slot}` : ''}${i.ready ? '' : ' (loading)'}` }))} onChange={setPlayModel}/> : <p className="rail-note">No model loaded. Load one in <button type="button" className="text-button inline" onClick={() => setTab('api')}>Local API</button>.</p>}{activeInstance && exampleFor(activeInstance) && <p className="rail-note">In the local API, send <code>"model": "{exampleFor(activeInstance)}"</code>.</p>}</div>
            <div className="rail-section"><div className="rail-label">Preset</div><div className="preset-chips">{presets.map((p, i) => { const Icon = icons[i]; return <button key={p.id} className={`preset-chip ${p.id === preset ? 'selected' : ''}`} aria-pressed={p.id === preset} title={p.subtitle} onClick={() => selectPreset(p.id)}><Icon size={14}/>{p.title}</button>; })}{savedPresets.map(p => <span className={`preset-chip saved ${p.id === preset ? 'selected' : ''}`} key={p.id}><button aria-pressed={p.id === preset} onClick={() => selectPreset(p.id)}><FileJson size={14}/>{p.name}</button><button className="chip-remove" aria-label={`Delete preset ${p.name}`} title={`Delete ${p.name}`} disabled={isBusy('preset')} onClick={() => setPresetToDelete(p)}><X size={13}/></button></span>)}</div><p className="rail-note">{presets.find(p => p.id === preset)?.subtitle ?? (preset.startsWith('saved-') ? 'Custom preset' : 'Custom questions')}</p></div>
            <div className="rail-section"><KeepContext on={!!activeInstance && keepsContext(activeInstance)} entries={playContext.entries} lastFlushed={playContext.last_flushed} scope={instances.length > 1 && activeInstance ? `for ${apiName(activeInstance)}` : undefined} disabled={!desktop || !activeInstance} onToggle={on => { if (activeInstance) void toggleKeepContext(activeInstance.id, on); }} onClear={() => { if (activeInstance) void clearContext(activeInstance.id); }}/></div>
            <section className="question-panel"><div className="rail-heading"><span className="rail-label">Questions <span className="code-badge">JSON</span></span><div className="heading-actions"><button className="text-button" onClick={() => { try { setQuestions(pretty(JSON.parse(questions))); } catch (e) { setNotice({ text: errorText(e), error: true }); } }}>Format</button><button className="icon-button" aria-label="Copy questions" onClick={() => copy(questions, 'questions')}>{copied === 'questions' ? <Check size={14}/> : <Copy size={14}/>}</button></div></div><JsonEditor label="Questions JSON" value={questions} onChange={value => { setQuestions(value); setResult(null); }}/><div className="editor-footer"><span className={questionError ? 'invalid' : 'valid'}><span className={`status-dot ${questionError ? 'red' : 'green'}`}/>{questionError ? 'Check JSON' : `${Object.keys(JSON.parse(questions)).length} questions`}</span><span>choice, noul or score</span></div></section>
            <form className="preset-save" onSubmit={e => { e.preventDefault(); saveCustom(); }}><input aria-label="Custom preset name" placeholder="Preset name" maxLength={80} value={presetName} onChange={e => setPresetName(e.target.value)}/><button className="secondary compact" disabled={isBusy('preset') || !!jsonError || !presetName.trim()}><Save size={13}/>{preset.startsWith('saved-') ? 'Save changes' : 'Save preset'}</button></form>
          </aside>
        </div>}
        {tab === 'models' && <>
          {modelFiles.length > 0 && <><div className="section-heading"><h2>On your machine</h2><span>{modelFiles.length} {modelFiles.length === 1 ? 'model' : 'models'}</span></div><div className="model-list">{modelFiles.map(model => { const hasProjector = localModels.some(file => isProjector(file.path) && folder(file.path) === folder(model.path)); return <div key={model.path} className="local-model-entry"><button className="model-row" onClick={() => { selectModel(model.path); setTab('api'); }}><AuthorLogo author={accountFor(model.path)}/><span><strong>{nameFor(model.path) ?? model.name}</strong><small>{nameFor(model.path) && nameFor(model.path) !== model.name.replace(/\.gguf$/i, '') ? `${model.name} / ` : ''}{(model.size / 1e9).toFixed(2)} GB / {model.path}</small></span>{hasProjector && <span className="chip" title="A multimodal projector is in the same folder. Turn it on in Local API.">Vision available</span>}{model.mtp && <span className="chip" title="This model has MTP layers, or a matching MTP head is in the same folder. Turn it on in Local API.">MTP available</span>}{model.decision === 'winnow-v1' && <span className="chip" title="No built-in decision support, so Decision mode Auto reads its answers through the Prompt layout (Winnow).">Prompt layout (Winnow)</span>}{model.decision == null && settings.decision_mode === 'auto' && <span className="chip warning" title="No built-in decision support and no prompt layout Noullama knows. Loading it will fail unless you set Decision mode in Local API.">Not a decision model</span>}{aliasesFor(model.path).length > 0 && <span className="chip" title={`API aliases: ${aliasesFor(model.path).join(', ')}`}>{aliasesFor(model.path).length === 1 ? aliasesFor(model.path)[0] : `${aliasesFor(model.path).length} aliases`}</span>}<span className="pill">{instances.filter(i => i.path === model.path).length > 1 ? `${instances.filter(i => i.path === model.path).length} loaded` : instances.some(i => i.path === model.path) ? 'Loaded' : model.path === settings.model_path ? 'Selected' : 'Select'}</span></button><button className="icon-button reveal-file" aria-label={`API names for ${model.name}`} title="API names" aria-expanded={aliasEditors.includes(model.path)} onClick={() => setAliasEditors(open => open.includes(model.path) ? open.filter(p => p !== model.path) : [...open, model.path])}><Tags size={16}/></button><CopyButton className="reveal-file" value={model.path} label={`Copy path of ${model.name}`} size={15} onError={copyError}/><button className="icon-button reveal-file" aria-label={`Show ${model.name} in folder`} title="Show in folder" disabled={isBusy('reveal')} onClick={() => revealFile(model.path)}><FolderOpen size={16}/></button>{aliasEditors.includes(model.path) && <div className="row-aliases"><ModelAliases path={model.path} aliases={aliasesFor(model.path)} routes={routes} disabled={!desktop} onSave={saveAliases} onError={copyError}/></div>}</div>; })}</div></>}
          {modelFiles.length === 0 && <div className="local-empty"><FolderOpen size={18}/><p>No models on this machine yet. Download one from the <button className="text-button" onClick={() => setTab('hub')}>Hub</button> or point Noullama at a GGUF file.</p></div>}
          {scanWarnings.length > 0 && <div className="hint" role="status">{scanWarnings.map(warning => <p key={warning}>{warning}</p>)}</div>}
          <div className="import-note"><FolderOpen size={18}/><div><strong>Local GGUF</strong><p>Already have a GGUF file? Pick it here instead of downloading.</p></div><button className="text-button" onClick={() => browseFile('model_path', 'api')}>Browse local model<ArrowRight size={14}/></button></div>
          <section className="panel directory-panel"><label>Model download directory<div className="file-input-row"><input aria-label="Model download directory" value={downloadDir} placeholder={snapshot ? `${snapshot.data_dir}/models` : 'Default app model directory'} onChange={e => setDownloadDir(e.target.value)}/><button className="secondary" type="button" disabled={isBusy('directory') || anyDownloadActive || downloadDir === snapshot?.settings.download_dir} onClick={saveDownloadDirectory}>Save directory</button></div><small>New downloads use this directory. Existing files stay in place.</small></label></section>
        </>}
        {tab === 'hub' && <>
          <form className="hub-search" onSubmit={e => { e.preventDefault(); const q = query.trim(); action('search', async () => { setModels(await invoke('search_models', { query: q, contextSize: settings.context_size, gpuLayers: settings.gpu_layers })); setSearched(q); }); }}><div className="search-input"><Search size={17}/><input aria-label="Search decision models" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search decision models, or enter organization/repository"/>{query && <button type="button" className="icon-button" aria-label="Clear search" onClick={() => { setQuery(''); setModels([]); setSearched(''); }}><X size={15}/></button>}</div><button className="secondary" disabled={isBusy('search')}>{isBusy('search') ? 'Searching…' : 'Search'}</button></form>
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
                <div className="hf-card-actions"><button className="secondary compact" disabled={isBusy('session')} onClick={() => action('session', async () => setSession(await invoke('hf_session')))}><RefreshCw size={13}/>Check session</button></div>
                <details className="token-details"><summary><ChevronRight className="disclosure" size={14}/>Use a token for this session</summary><div className="inline-form"><input type="password" autoComplete="off" aria-label="Hugging Face token" placeholder="hf_… (kept in memory only)" value={token} onChange={e => setToken(e.target.value)}/><button className="secondary compact" onClick={() => action('token', async () => { await invoke('set_hf_token', { token }); setToken(''); setSession(await invoke('hf_session')); })}>Apply</button></div><button className="text-button" onClick={() => action('token', async () => { await invoke('set_hf_token', { token: '' }); setSession(await invoke('hf_session')); })}>Use saved login instead</button></details>
              </div>
            </section>
            <section className="hub-detail" id="hub-detail" aria-label="Repository details">
              {!inspection ? <div className="hub-empty">{isBusy('inspect') ? <><RefreshCw className="spin" size={20}/><p>Reading repository…</p></> : <><Network size={22}/><p>Pick a repository to see its GGUF files.</p></>}</div> : <>
                <div className="detail-head"><AuthorLogo author={inspection.repo.split('/')[0]} size={56}/><div><h2>{inspection.repo.split('/').slice(1).join('/')}</h2><p>{inspection.repo.split('/')[0]}</p></div><CopyButton className="detail-copy" value={inspection.repo} label="Copy repository id" size={15} onError={copyError}/></div>
                <div className="detail-chips"><span className="chip">GGUF</span><span className="chip">{inspection.files.length} {inspection.files.length === 1 ? 'file' : 'files'}</span>{inspection.sha && <span className="chip chip-copy" title={inspection.sha}>Revision <code>{inspection.sha.slice(0, 10)}</code><CopyButton value={inspection.sha} label="Copy full revision" size={12} onError={copyError}/></span>}</div>
                <div className="files" id="hub-files">{inspection.files.length === 0 ? <p className="hint">No GGUF files in this repository. Open a GGUF conversion of this model.</p> : inspection.files.map(file => { const fileDownload = downloadFor(inspection.repo, file.name); const isPreparing = preparing.includes(`${inspection.repo}/${file.name}`); return <div className="file-entry" key={file.name}><div className="file-row"><FileJson size={17}/><div><span className="identity-line"><strong>{file.name}</strong><CopyButton value={file.name} label={`Copy file name ${file.name}`} size={12} onError={copyError}/></span><small>{file.projector ? 'Multimodal projector' : file.mtp_head ? 'MTP head' : 'Model weights'}{file.size ? ` / ${(file.size / 1e9).toFixed(2)} GB` : ''}</small></div>{!file.projector && !file.mtp_head && <FileFit fit={file.fit}/>}<button className="secondary compact" disabled={!!fileDownload?.active || isPreparing} onClick={() => startDownload(inspection.repo, file.name)}><ArrowDownToLine size={14}/>Download</button></div>{fileDownload ? <DownloadPanel download={fileDownload} busy={isBusy('cancel')} onCancel={() => cancelDownload(fileDownload.id)} onReveal={() => revealFile(fileDownload.path!)} onSelect={() => { if (file.projector) edit('mmproj_path', fileDownload.path!); else if (file.mtp_head) { edit('mtp_path', fileDownload.path!); edit('mtp', true); } else selectModel(fileDownload.path!); setTab('api'); }}/> : isPreparing ? <div className="download-preparing" role="status"><RefreshCw className="spin" size={14}/>Preparing download…</div> : null}</div>; })}</div>
              </>}
            </section>
          </div>
        </>}
        {tab === 'runtime' && <>
          <RuntimeManager currentPath={settings.runtime_path} running={running} progress={snapshot?.runtime_download} onRefresh={refresh} onSelect={runtime => { edit('runtime_path', runtime.path); if (runtime.backend) edit('gpu_layers', runtime.backend === 'cpu' ? 0 : 99); }}>
            <label>Path to llama-server<div className="file-input-row"><input placeholder="/path/to/llama.cpp/build/bin" value={settings.runtime_path} onChange={e => edit('runtime_path', e.target.value)}/><button className="secondary" type="button" disabled={isBusy('probe')} onClick={() => action('probe', async () => setVersion(await invoke('probe_runtime', { path: settings.runtime_path, libraryPath: settings.library_path })))}><RefreshCw size={14} className={isBusy('probe') ? 'spin' : ''}/>Check</button></div><small>A runtime folder or the llama-server executable. Choosing a build above fills this in.</small></label>
            {version && <pre className="runtime-version">{version}</pre>}
            <label>Shared-library folders (optional)<input placeholder="Found automatically, or /usr/local/cuda/lib64" value={settings.library_path} onChange={e => edit('library_path', e.target.value)}/><small>For CUDA or other libraries the runtime cannot find. Separate folders with a colon. Only the runtime process sees them.</small></label>
          </RuntimeManager>
          <section className="panel settings-panel"><div className="panel-heading"><span><Settings2 size={16}/> Connection</span></div><div className="settings-body">
            <div className="form-grid"><label>Local API port<NumberField min={1} max={65535} value={settings.api_port} onChange={value => edit('api_port', value)}/><small>Used after a restart. If it is taken, Noullama picks a free nearby port and shows it in Local API.</small></label><label>Private llama-server port<NumberField min={1} max={65535} value={settings.runtime_port} onChange={value => edit('runtime_port', value)}/><small>Internal only, on 127.0.0.1. Each loaded model takes the next free port from here.</small></label></div>
            <label>Hugging Face CLI<input value={settings.hf_path} onChange={e => edit('hf_path', e.target.value)} placeholder="hf"/><small>Use the full path if hf is not on the desktop app's PATH.</small></label>
          </div></section>
          <div className={`save-bar ${runtimeFormDirty ? 'dirty' : ''}`}><span className="hint">{runtimeFormDirty ? running ? 'You have unsaved changes. Loading a model does not save them, and loaded models keep the build they started with.' : 'You have unsaved changes. Loading a model does not save them.' : running ? 'All changes saved. Loaded models keep the build they started with.' : 'All changes saved.'}</span><button className={runtimeFormDirty ? 'primary' : 'secondary'} disabled={!runtimeFormDirty || isBusy('save', 'save-load', 'start')} onClick={() => action('save', async () => { await saveRuntimeSettings(); setNotice({ text: 'Settings saved.', error: false }); })}>Save settings</button></div>
        </>}
        {tab === 'library' && <Library library={snapshot?.library ?? null} loadedPaths={instances.map(i => i.path)} desktop={desktop}
          companions={localModels.flatMap((file): Companion[] => file.mtp_head ? [{ path: file.path, kind: 'mtp' }] : isProjector(file.path) ? [{ path: file.path, kind: 'mmproj' }] : [])}
          onSave={saveLibrary} onReload={() => libraryCommand('reload_library')} onOpenFile={() => libraryCommand('open_models_file')} onOpenBackups={() => libraryCommand('open_backups_folder')}
          onRecordSha={(path, slot) => libraryCommand('record_sha256', { path, slot })} onBrowse={() => invoke<string | null>('pick_model_file')} onError={copyError}/>}
        {tab === 'settings' && <>
          <section className="panel settings-panel"><div className="panel-heading"><span><MemoryStick size={16}/> Memory protection</span></div><div className="settings-body">
            <div className="field"><div className="field-row"><span>Allow force fit</span><Toggle label={snapshot?.settings.allow_force_fit ? 'On' : 'Off'} ariaLabel="Allow force fit" checked={snapshot?.settings.allow_force_fit ?? false} disabled={!desktop || isBusy('memory-pref')} onChange={value => void action('memory-pref', async () => { await invoke('set_memory_preference', { key: 'allow_force_fit', value }); })}/></div><small>Disables OOM load prevention and automatic recovery from sustained critical memory pressure. The operating system or runtime may still stop a model if memory runs out.</small></div>
            {snapshot?.settings.hide_crowded_load_warning && <button className="text-button" disabled={isBusy('memory-pref')} onClick={() => void action('memory-pref', async () => { await invoke('set_memory_preference', { key: 'hide_crowded_load_warning', value: false }); })}>Show crowded-load warning again</button>}
          </div></section>
          <section className="panel settings-panel"><div className="panel-heading"><span><Route size={16}/> Local API routing</span></div><div className="settings-body">
            <div className="field"><div className="field-row"><span>Fall through to another instance</span><Toggle label={snapshot?.settings.fall_through_to_other_instances ? 'On' : 'Off'} ariaLabel="Fall through to another instance" checked={snapshot?.settings.fall_through_to_other_instances ?? false} disabled={!desktop || isBusy('fall-through')} onChange={on => void action('fall-through', async () => { await invoke('set_fall_through', { on }); })}/></div><small>When instance 1 of a model file is not loaded but another instance is, requests for its plain names (without a suffix) go to the loaded instance with the lowest number. Off, they get 503. Suffixed names never fall through.</small></div>
            <div className="field"><div className="field-row"><span>When every instance is busy</span><div className="segmented" role="group" aria-label="When every instance is busy">{WHEN_BUSY.map(option => <button key={option.value} type="button" className={(snapshot?.settings.any_available_when_busy ?? 'reject') === option.value ? 'on' : ''} aria-pressed={(snapshot?.settings.any_available_when_busy ?? 'reject') === option.value} disabled={!desktop || isBusy('when-busy')} onClick={() => void action('when-busy', async () => { await invoke('set_any_available_when_busy', { mode: option.value }); })}>{option.label}</button>)}</div></div>
              {whenBusy === 'wait' && <div className="field-row"><span>Wait up to (ms)</span><NumberField value={waitDraft ?? savedWait} min={WAIT_MS.min} max={WAIT_MS.max} step={WAIT_MS.step} onChange={value => setWaitDraft(Math.min(Number.MAX_SAFE_INTEGER, Math.max(0, Math.round(value))))}/></div>}
              <small>{WHEN_BUSY.find(option => option.value === whenBusy)?.detail(waitDraft ?? savedWait)}</small>
              <small>Applies to requests that send "any-available" as the model. They go to a ready instance of any model that is not running a request, the one idle longest first.</small></div>
          </div></section>
          <section className="panel settings-panel"><div className="panel-heading"><span><ShieldCheck size={16}/> Model file checks</span></div><div className="settings-body">
            <div className="field"><div className="field-row"><span>sha256 before loading</span><div className="segmented" role="group" aria-label="sha256 check">{SHA_CHECKS.map(option => <button key={option.value} type="button" className={(snapshot?.settings.sha256_check ?? 'once') === option.value ? 'on' : ''} aria-pressed={(snapshot?.settings.sha256_check ?? 'once') === option.value} disabled={!desktop || isBusy('sha-check')} onClick={() => void setShaCheck(option.value)}>{option.label}</button>)}</div></div>
              <small>{SHA_CHECKS.find(option => option.value === (snapshot?.settings.sha256_check ?? 'once'))?.detail}</small>
              <small>Each model's sha256 is kept in models.yml. It guards against loading a file that was replaced, truncated or swapped by mistake; it is not a security boundary.</small></div>
          </div></section>
          <section className="panel settings-panel"><div className="panel-heading"><span><FileJson size={16}/> Models file</span></div><div className="settings-body">
            <div className="field"><div className="field-row"><span className="identity-line"><code>{snapshot?.library?.file ?? 'models.yml'}</code>{snapshot?.library?.file && <CopyButton value={snapshot.library.file} label="Copy models.yml path" size={12} onError={copyError}/>}</span><span className="save-actions"><button className="secondary compact" disabled={!desktop} onClick={() => action('open-yml', () => libraryCommand('open_models_file'))}>Edit models.yml</button><button className="secondary compact" disabled={!desktop} onClick={() => action('open-backups', () => libraryCommand('open_backups_folder'))}>Backups</button><button className="text-button" onClick={() => setTab('library')}>Open Library</button></span></div>
              <small>Noullama rereads models.yml whenever it changes, and before loading, ejecting, routing a request or changing Keep context. It backs the file up before every change it makes, other than recording a sha256, and keeps the newest 20 copies.</small></div>
          </div></section>
        </>}
        {tab === 'api' && <>
          <section className="endpoint-card"><div><span className={`status-dot ${!snapshot?.api_error && desktop ? 'green' : ''}`}/><strong>{apiBase}</strong><button className="icon-button" aria-label="Copy API URL" onClick={() => copy(apiBase, 'api')}>{copied === 'api' ? <Check size={16}/> : <Copy size={16}/>}</button></div><p>{snapshot?.api_error ?? 'Localhost only. No API key needed. Available while Noullama is open.'}</p></section>
          {loadErrors.map(error => <div className="load-error" role="alert" key={error.id}><AlertCircle size={18}/><div><strong>{error.name}: {error.message.split('\n')[0]}</strong>{error.message.includes('\n') && <pre>{error.message.split('\n').slice(1).join('\n')}</pre>}</div><div className="load-error-actions"><button className="text-button" onClick={() => setShowLogs(true)}>View logs</button><button className="icon-button" aria-label={`Dismiss error for ${error.name}`} onClick={() => void dismissLoadError(error.id)}><X size={15}/></button></div></div>)}
          <section className="panel settings-panel load-panel" aria-label="Load a model"><div className="panel-heading"><span><Play size={16}/> Load a model</span>{fit && <span className={`fit-badge fit-level-${fit.level}`} title={`Estimated memory fit, not model/runtime compatibility. ${fitDetail}`}><span className="fit-bars" aria-hidden="true">{[1, 2, 3, 4].map(n => <i key={n} className={n <= fit.level ? 'on' : ''}/>)}</span>{fit.label}</span>}</div><div className="settings-body">
            <div className="load-identity">
              <div className="load-identity-icon">{accountFor(model.selected) ? <AuthorLogo author={accountFor(model.selected)} size={40}/> : <Cpu size={20}/>}</div>
              <div className="model-identity"><small>{selectedInstance ? selectedInstance.ready ? 'Loaded model' : 'Loading model' : 'Selected model'}</small><span className="identity-line"><strong title={model.selected}>{model.selected ? nameFor(model.selected) ?? basename(model.selected) : 'No model selected'}</strong>{model.selected && <CopyButton value={basename(model.selected)} label="Copy model file name" size={13} onError={copyError}/>}</span>{model.device && <small>{model.device}</small>}</div>
              <span className={`chip ${model.ready ? 'success' : ''}`}><span className={`status-dot ${model.ready ? 'green' : model.loaded ? 'amber' : ''}`}/>{isBusy('start') ? 'Starting…' : model.label}</span>
              {selectedInstances.length > 1 && <div className="load-instance"><Select label="Instance" value={selectedInstance?.id ?? ''} options={selectedInstances.map(i => ({ value: i.id, label: `${apiName(i)} · instance\u00a0${i.slot}${i.ready ? '' : ' (loading)'}` }))} onChange={setLoadTarget}/></div>}
            </div>
            <p className="hint fit-note">{fit ? fitDetail : model.selected ? 'The selected model file could not be read.' : 'Select a model to see how it would run on this machine.'}</p>
            <label>Model file (GGUF)<div className="file-input-row"><input placeholder="/path/to/model.gguf" value={settings.model_path} onChange={e => edit('model_path', e.target.value)}/><button className="secondary" type="button" disabled={modelPending} onClick={() => browseFile('model_path')}><FolderOpen size={14}/>Browse file</button><button className="secondary" type="button" disabled={modelPending} onClick={() => setTab('models')}><Layers3 size={14}/>Select model</button></div></label>
            <div className="field"><div className="field-row"><span>Multimodal projector</span><Toggle label={settings.mmproj_path ? 'On' : 'Off'} ariaLabel="Multimodal projector" title={settings.mmproj_path ? `Loads ${basename(settings.mmproj_path)}` : projector ? `Load ${basename(projector)} with this model` : undefined} checked={!!settings.mmproj_path} disabled={!settings.mmproj_path && !projector} onChange={on => edit('mmproj_path', on ? projector : '')}/></div>{settings.mmproj_path ? <div className="file-input-row"><input aria-label="Multimodal projector path" placeholder="/path/to/mmproj.gguf" value={settings.mmproj_path} onChange={e => edit('mmproj_path', e.target.value)}/><button className="secondary" type="button" disabled={modelPending} onClick={() => browseFile('mmproj_path')}><FolderOpen size={14}/>Browse</button></div> : <small>{projector ? `Found ${basename(projector)} next to the selected model.` : <>No projector next to the selected model. <button type="button" className="text-button inline" disabled={modelPending} onClick={() => browseFile('mmproj_path')}>Choose a file</button></>}</small>}<small>Lets image-capable models read images in the Playground.</small></div>
            <div className="field"><div className="field-row"><span>Multi-token prediction (MTP)</span><Toggle label={settings.mtp ? 'On' : 'Off'} ariaLabel="Multi-token prediction" title={settings.mtp || mtpAvailable ? mtpTitle : undefined} checked={settings.mtp} disabled={!settings.mtp && !mtpAvailable} onChange={setMtp}/></div>{settings.mtp_path ? <div className="file-input-row"><input aria-label="MTP head path" placeholder="/path/to/mtp-model.gguf" value={settings.mtp_path} onChange={e => edit('mtp_path', e.target.value)}/><button className="secondary" type="button" disabled={modelPending} onClick={() => browseFile('mtp_path')}><FolderOpen size={14}/>Browse</button></div> : <small>{builtInMtp ? 'The selected model has built-in MTP layers.' : mtpHead ? `Found ${basename(mtpHead)} next to the selected model.` : <>No MTP layers in the selected model and no matching MTP head next to it. <button type="button" className="text-button inline" disabled={modelPending} onClick={() => browseFile('mtp_path')}>Choose a file</button></>}</small>}<small>Drafts several tokens per step with the model's MTP layers, which can speed up generation. Needs a llama.cpp build with MTP support.</small></div>
            <div className="field"><div className="field-row"><span>Decision mode</span><div className="segmented" role="group" aria-label="Decision mode">{DECISION_MODES.map(([value, label]) => <button key={value} type="button" className={settings.decision_mode === value ? 'on' : ''} aria-pressed={settings.decision_mode === value} onClick={() => edit('decision_mode', value)}>{label}</button>)}</div></div><small>{(DECISION_MODES.find(([value]) => value === settings.decision_mode) ?? DECISION_MODES[0])[2]}</small></div>
            <div className="field"><div className="field-row"><span>Keep context</span><Toggle label={settings.keep_context ? 'On' : 'Off'} ariaLabel="Keep context for new loads" checked={settings.keep_context} onChange={on => edit('keep_context', on)}/></div><small>Whether a newly loaded model starts with Keep context on. Loaded models have their own switch on their card below; this does not change them.</small></div>
            <div className="form-grid"><label>Context size<NumberField min={512} max={131072} step={1024} value={settings.context_size} onChange={value => edit('context_size', value)}/><small>Maximum tokens per request. The default is 8192.</small></label><label>GPU layers<NumberField min={-1} value={settings.gpu_layers} onChange={value => edit('gpu_layers', value)}/><small>0 runs on the CPU. 99 puts every layer that fits on the GPU.</small></label></div>
            <div className="field"><div className="field-row"><span>llama.cpp build</span><span className="load-build"><code title={savedSettings?.runtime_path || undefined}>{loadRuntime}</code><button type="button" className="text-button" onClick={() => setTab('runtime')}>Change in Runtime<ArrowRight size={13}/></button></span></div><small>{runtimeFormDirty ? 'The Runtime tab has unsaved changes. Loads use the saved build and shared-library folders until you save them there.' : 'The build and shared-library folders saved on the Runtime tab.'}</small></div>
          </div></section>
          <div className={`save-bar load-bar ${loadDirty ? 'dirty' : ''}`}><span className="hint">{loadDirty ? 'You have unsaved changes. Loading saves them first.' : reloadSelected ? 'The selected model was loaded with other settings. Reload model applies these.' : selectedInstances.length > 1 ? `The selected model is loaded ${selectedInstances.length} times. Eject and Reload act on the instance chosen above.` : selectedInstance ? 'The selected model is loaded with these settings.' : running ? 'Settings apply when you load. Loaded models keep the settings they started with.' : 'Settings apply when you load.'}</span><div className="load-bar-actions">
            <button className="secondary" disabled={!loadDirty || modelPending} onClick={() => action('save-load', async () => { await saveLoadSettings(); setNotice({ text: 'Settings saved.', error: false }); })}>Save settings</button>
            {selectedInstance && <button className="secondary" disabled={modelPending} title={selectedInstance.ready ? `Unload ${apiName(selectedInstance)}; other loaded models keep running` : undefined} onClick={() => void eject(selectedInstance.id)}>{selectedInstance.ready ? <Unplug size={13}/> : <Square size={13}/>}{selectedInstance.ready ? 'Eject' : 'Cancel loading'}</button>}
            {(!selectedInstance || reloadSelected) && <button className="primary" disabled={modelPending || !model.selected || model.missing || (!!selectedInstance && !selectedInstance.ready)} title={!selectedInstance && running ? 'Loads beside the models already loaded' : undefined} onClick={() => void loadSelected({ replace: selectedInstance?.id ?? null })}>{isBusy('start') ? <RefreshCw className="spin" size={14}/> : <Play size={14}/>} {isBusy('start') ? 'Starting…' : selectedInstance ? 'Reload model' : running ? 'Load beside' : 'Load model'}</button>}
            {selectedInstance && <button className={reloadSelected ? 'secondary' : 'primary'} disabled={modelPending || !model.selected || model.missing} title="Loads one more instance of this file beside the loaded one. It answers to the file's names with a suffix such as -i-2." onClick={() => void loadSelected()}>{isBusy('start') ? <RefreshCw className="spin" size={14}/> : <Plus size={14}/>} {isBusy('start') ? 'Starting…' : 'Load another instance'}</button>}
          </div></div>
          {instances.length > 0 && <><div className="section-heading"><h2>Loaded</h2><span>{instances.length} {instances.length === 1 ? 'model' : 'models'} · each keeps the settings it was loaded with</span></div><div className="loaded-list">{instances.map(instance => <InstanceCard key={instance.id} instance={instance} account={accountFor(instance.path)} runtime={runtimeName(instance.runtime_path || instance.settings.runtime_path)} aliases={aliasesFor(instance.path)} routes={routes} displayName={apiName(instance)}
            stale={differsFrom(instance)} canReload={differsFrom(instance) && instance.ready} starting={isBusy('start')} ejecting={isBusy(`eject:${instance.id}`)} reloadDisabled={modelPending || model.missing} namesDisabled={!desktop}
            keepContext={keepsContext(instance)} onKeepContext={on => void toggleKeepContext(instance.id, on)} onClearContext={() => void clearContext(instance.id)}
            onEject={() => void eject(instance.id)} onReload={() => void loadSelected({ replace: instance.id })} onSaveAliases={saveAliases} onSaveSuffix={suffix => saveSuffix(instance.id, suffix)} onSaveInstanceAliases={aliases => saveInstanceAliases(instance.id, aliases)} onError={copyError}/>)}</div></>}
          <ModelRouting loadedCount={instances.length} anyReady={instances.some(i => i.ready)} whenBusy={whenBusy} waitMs={savedWait} routes={routes} aliasesFor={aliasesFor} disabled={!desktop} onSave={saveAliases} onError={copyError}/>
          <div className="endpoint-list">{([['GET', '/v1/models', 'List the names that reach a loaded model'], ['POST', '/v1/systemone', 'Native state + typed questions'], ['POST', '/v1/chat/completions', 'OpenAI chat response envelope'], ['GET', '/health', 'Gateway health and model readiness (outside /v1)']] as const).map(([method, path, text]) => <div key={path}><span className={`method ${method === 'GET' ? 'get' : ''}`}>{method}</span><code>{path}</code><span>{text}</span><CopyButton value={`${apiOrigin}${path}`} label={`Copy ${apiOrigin}${path}`} size={13} onError={copyError}/></div>)}</div>
          <section className="panel code-panel"><div className="panel-heading"><span>Native decision request</span><button className="icon-button" aria-label="Copy native API example" onClick={() => copy(`curl ${apiBase}/systemone -H 'Content-Type: application/json' -d '${requestJson.replace(/'/g, "'\\''")}'`, 'curl')}>{copied === 'curl' ? <Check size={14}/> : <Copy size={14}/>}</button></div><pre>{`curl ${apiBase}/systemone \\\n  -H 'Content-Type: application/json' \\\n  -d '${requestJson.replace(/'/g, "'\\''")}'`}</pre></section>
          <section className="panel code-panel"><div className="panel-heading"><span>Python · OpenAI SDK</span><div className="heading-actions"><span className="chip">Non-streaming</span><CopyButton value={pythonExample} label="Copy Python example" onError={copyError}/></div></div><pre>{pythonExample}</pre></section>
          <div className="api-note"><BookOpen size={18}/><div><strong>Request format</strong><p>Send one user message containing a JSON request with state and questions. The assistant content contains the native result as JSON. Chat history, tools, and streaming are unsupported. Questions use choice, noul, or score; this runtime does not support extraction or multi-select.</p><a href="https://github.com/ggml-org/llama.cpp/blob/b11938/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api" target="_blank" rel="noreferrer">llama.cpp decision API reference<ExternalLink size={12}/></a></div></div>
        </>}
      </div>
    </main>
    {showDownloadToast && <aside className="download-toast" aria-label="Downloads" ref={toastRef}>
      <div className="toast-head"><strong>Downloads{toastDownloads.length > 1 ? ` (${toastDownloads.length})` : ''}</strong><button className="icon-button" aria-label={toastCollapsed ? 'Show downloads' : 'Collapse downloads'} aria-expanded={!toastCollapsed} onClick={() => setToastCollapsed(v => !v)}>{toastCollapsed ? <ChevronUp size={15}/> : <ChevronDown size={15}/>}</button></div>
      {!toastCollapsed && <div className="toast-list">{toastDownloads.map(item => <div className="toast-item" key={item.id}>
        <div className="toast-row"><button className="toast-open" onClick={() => { setTab('hub'); openRepo(item.repo); }}><strong>{item.file}</strong><small>{item.active ? `${item.repo} / ${item.progress === null ? 'Connecting' : `${item.progress.toFixed(0)}%`}` : `${item.repo} / Downloaded`}</small></button>{item.active ? <button className="icon-button" aria-label={`Cancel download of ${item.file}`} title="Cancel download" disabled={isBusy('cancel')} onClick={() => cancelDownload(item.id)}><X size={15}/></button> : <><CircleCheck className="toast-done" size={17}/><button className="icon-button" aria-label={`Dismiss ${item.file}`} title="Dismiss" onClick={() => void dismissDownload(item.id)}><X size={15}/></button></>}</div>
        {item.active && <div className="download-track" aria-hidden="true"><i style={{ width: `${Math.max(0, Math.min(100, item.progress ?? 0))}%` }}/></div>}
      </div>)}</div>}
    </aside>}
    <MemoryWarningDialog warning={crowdedWarning} pending={isBusy('memory-warning', 'start')} hide={hideCrowded} onHide={setHideCrowded}
      onCancel={() => void action('memory-warning', async () => { if (hideCrowded) await invoke('set_memory_preference', { key: 'hide_crowded_load_warning', value: true }); setCrowdedWarning(null); })}
      onEject={id => { void (async () => { try { if (hideCrowded) await invoke('set_memory_preference', { key: 'hide_crowded_load_warning', value: true }); const replace = crowdedWarning?.replace ?? null; setCrowdedWarning(null); await loadSelected({ replace, ejectOther: id }); } catch (e) { setNotice({ text: errorText(e), error: true }); } })(); }}/>
    <ContextMenu/>
    {updates && <UpdatesDialog open={updatesOpen} view={updates} onClose={() => setUpdatesOpen(false)} onCheck={() => { void updateCommand('check_for_update'); }} onInstall={() => { void updateCommand('install_update'); }} onPrefs={(includePrereleases, checkOnLaunch) => { setUpdates(view => view && { ...view, include_prereleases: includePrereleases, check_on_launch: checkOnLaunch }); void updateCommand('set_update_prefs', { includePrereleases, checkOnLaunch }); }}/>}
    <DeletePresetDialog name={presetToDelete?.name ?? null} pending={isBusy('preset')} onCancel={() => setPresetToDelete(null)} onConfirm={() => { if (presetToDelete) void trashPreset(presetToDelete.id); }}/>
    <QuitDialog open={snapshot?.quit_confirmation ?? false} pending={isBusy('quit') || isBusy('cancel-quit')} onCancel={() => { void action('cancel-quit', async () => { await invoke('cancel_quit'); }); }} onConfirm={() => { void action('quit', async () => { await invoke('quit_app'); }); }}/>
    {showLogs && <div className="log-drawer"><div className="panel-heading"><span><Terminal size={16}/> Runtime activity</span><button className="icon-button" aria-label="Close runtime logs" onClick={() => setShowLogs(false)}><X size={17}/></button></div><pre>{snapshot?.logs.join('\n') || 'No runtime activity yet.'}</pre></div>}
  </div>;
}

createRoot(document.getElementById('root')!).render(<App/>);
