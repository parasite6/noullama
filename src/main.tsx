import { useOptimisticWrites } from './optimistic-writes';
import { Reveal } from './Reveal';
import { busyButton, busyDelay } from './busy-button';
import { IconContext } from '@phosphor-icons/react/dist/lib/context';
import { SegmentedControl } from './SegmentedControl';
import { ApiKeys } from './ApiKeys';
import { PendingActions, equalJson, retainEqual, retainSnapshotSections, shouldScanModels, usePageVisible } from './refresh-state';
import { MemoryWarningDialog, type CrowdedWarning } from './MemoryWarningDialog';
import React, { useDeferredValue, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { createRoot } from 'react-dom/client';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Toggle } from './Toggle';
import { ModelDetails, type MemoryUse } from './ModelDetails';
import { ImageAttachments, MAX_TOTAL_BYTES, readImages, type Attachment } from './ImageAttachments';
import { PulseIcon as Activity } from '@phosphor-icons/react/dist/icons/Pulse';
import { WarningCircleIcon as AlertCircle } from '@phosphor-icons/react/dist/icons/WarningCircle';
import { DownloadSimpleIcon as ArrowDownToLine } from '@phosphor-icons/react/dist/icons/DownloadSimple';
import { ArrowRightIcon as ArrowRight } from '@phosphor-icons/react/dist/icons/ArrowRight';
import { BookOpenIcon as BookOpen } from '@phosphor-icons/react/dist/icons/BookOpen';
import { CheckIcon as Check } from '@phosphor-icons/react/dist/icons/Check';
import { CaretDownIcon as ChevronDown } from '@phosphor-icons/react/dist/icons/CaretDown';
import { CaretRightIcon as ChevronRight } from '@phosphor-icons/react/dist/icons/CaretRight';
import { CaretUpIcon as ChevronUp } from '@phosphor-icons/react/dist/icons/CaretUp';
import { CheckCircleIcon as CircleCheck } from '@phosphor-icons/react/dist/icons/CheckCircle';
import { CopyIcon as Copy } from '@phosphor-icons/react/dist/icons/Copy';
import { CpuIcon as Cpu } from '@phosphor-icons/react/dist/icons/Cpu';
import { DownloadIcon as Download } from '@phosphor-icons/react/dist/icons/Download';
import { ArrowSquareOutIcon as ExternalLink } from '@phosphor-icons/react/dist/icons/ArrowSquareOut';
import { FileCodeIcon as FileJson } from '@phosphor-icons/react/dist/icons/FileCode';
import { FolderOpenIcon as FolderOpen } from '@phosphor-icons/react/dist/icons/FolderOpen';
import { GaugeIcon as Gauge } from '@phosphor-icons/react/dist/icons/Gauge';
import { StackIcon as Layers3 } from '@phosphor-icons/react/dist/icons/Stack';
import { EnvelopeSimpleIcon as Mail } from '@phosphor-icons/react/dist/icons/EnvelopeSimple';
import { MemoryIcon as MemoryStick } from '@phosphor-icons/react/dist/icons/Memory';
import { CircuitryIcon as Microchip } from '@phosphor-icons/react/dist/icons/Circuitry';
import { GraphIcon as Network } from '@phosphor-icons/react/dist/icons/Graph';
import { PlayIcon as Play } from '@phosphor-icons/react/dist/icons/Play';
import { PlusIcon as Plus } from '@phosphor-icons/react/dist/icons/Plus';
import { ArrowsClockwiseIcon as RefreshCw } from '@phosphor-icons/react/dist/icons/ArrowsClockwise';
import { PathIcon as Route } from '@phosphor-icons/react/dist/icons/Path';
import { MagnifyingGlassIcon as Search } from '@phosphor-icons/react/dist/icons/MagnifyingGlass';
import { SlidersHorizontalIcon as Settings2 } from '@phosphor-icons/react/dist/icons/SlidersHorizontal';
import { ShieldIcon as Shield } from '@phosphor-icons/react/dist/icons/Shield';
import { ShieldCheckIcon as ShieldCheck } from '@phosphor-icons/react/dist/icons/ShieldCheck';
import { SquareIcon as Square } from '@phosphor-icons/react/dist/icons/Square';
import { TagIcon as Tags } from '@phosphor-icons/react/dist/icons/Tag';
import { TerminalWindowIcon as Terminal } from '@phosphor-icons/react/dist/icons/TerminalWindow';
import { TrashIcon as Trash2 } from '@phosphor-icons/react/dist/icons/Trash';
import { FloppyDiskIcon as Save } from '@phosphor-icons/react/dist/icons/FloppyDisk';
import { PowerIcon as Power } from '@phosphor-icons/react/dist/icons/Power';
import { PlugsIcon as Unplug } from '@phosphor-icons/react/dist/icons/Plugs';
import { XIcon as X } from '@phosphor-icons/react/dist/icons/X';
import { ArrowCircleUpIcon as CircleArrowUp } from '@phosphor-icons/react/dist/icons/ArrowCircleUp';
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
import { SidebarTabs } from './SidebarTabs';
import { NumberField } from './NumberField';
import { SettingRow, SettingSection, StatusNote, type Tone } from './SettingsLayout';
import { runtimeStatus } from './runtime-state';
import { modelStatus, verifyingText, type Verifying } from './model-state';
import { JsonEditor } from './JsonEditor';
import { DownloadPanel, type DownloadState } from './DownloadPanel';
import { AuthorLogo, authorFromPath } from './AuthorLogo';
import { HuggingFaceLogo } from './HuggingFaceLogo';
import { RuntimeManager, type RuntimeDownload } from './RuntimeManager';
import { enablesKeyboardFocus, instancesOf, latestOnly, matchingProbe, settingsGroupMatches, panelReplace, playModelAfterLoad, reloadSlot, visibleBanner, waitText, deviceChoice, deviceLabel, CPU_DEVICE, CHOOSE_GPUS, SPLIT_MODES, isSplit, chosenGpus, toggleGpu, ratioError, type RuntimeDevice } from './ui-state';
import { containScrollChaining } from './scroll-contain';
import '@fontsource-variable/manrope';
import '@fontsource-variable/geist-mono';
import './styles.css';

// 96 px stays sharp at 28 CSS px up to 3x scaling; the 960 px original
// decoded to 3.7 MB of bitmap for the sidebar.
const logoUrl = new URL('./logo-96.png', import.meta.url).href;

type Settings = { runtime_path: string; model_path: string; mmproj_path: string; mtp: boolean; mtp_path: string; decision_mode: string; context_size: number; gpu_layers: number; device: string; split_mode: string; tensor_split: string; cpu_runtime_path: string; api_port: number; require_api_key: boolean; runtime_port: number; hf_path: string; download_transport: string; download_dir: string; library_path: string; keep_context: boolean; sha256_check: string; allow_force_fit: boolean; hide_crowded_load_warning: boolean; fall_through_to_other_instances: boolean; any_available_when_busy: string; any_available_wait_ms: number };

// One loaded model, with the settings it was loaded with.
type Instance = { verifying: Verifying | null; id: string; name: string; label: string; slot: number; suffix: string; default_suffix: string; api_names: string[]; instance_aliases: string[]; path: string; device: string; runtime_path: string; projector: boolean; mmproj_path: string; mtp: boolean; mtp_path: string; layout: string | null; ready: boolean; context_size: number; gpu_layers: number; required_bytes: number | null; context: { entries: number; last_flushed: number }; keep_context: boolean; settings: Settings };
type LoadError = { id: string; path: string; slot: number; name: string; message: string };
type Snapshot = { settings: Settings; running: boolean; ready: boolean; instances: Instance[]; default_model: string | null; routes: ApiRoute[]; library: LibraryState; api_port: number; api_error: string | null; api_notice: string | null; load_errors: LoadError[]; runtime_download: RuntimeDownload; cpu_runtime_download: RuntimeDownload; quit_confirmation: boolean; startup_warning: string | null; logs: string[]; downloads: DownloadState[]; data_dir: string; download_dir: string; loaded_model: { path: string; device: string; runtime_path: string; projector: boolean; mtp: boolean; layout: string | null } | null; hardware: Hardware | null; model_missing: boolean };
type Hardware = { ram_bytes: number; vram_bytes: number | null; gpu: string | null; cpu_cores: number };
type Fit = { level: number; label: string; detail: string; model_bytes: number; required_bytes: number };
type SavedPreset = { id: string; name: string; questions: unknown; state: string; state_mode: 'text' | 'json' };
type Model = { id: string; downloads: number; likes: number; tags: string[]; fit: Fit | null };
type ModelFile = { name: string; size?: number; projector: boolean; mtp_head: boolean; fit: Fit | null };
type Inspection = { repo: string; files: ModelFile[]; sha: string };
type Session = { authenticated: boolean; account: string | null; cli_available: boolean; xet: string | null; xet_known: boolean; token_present: boolean };
const TRANSPORTS = [
  { value: 'auto', label: 'Auto', detail: 'hf decides: Xet for files stored on Xet when it has hf_xet, otherwise HTTPS. An HF_HUB_DISABLE_XET set where Noullama was started still applies.' },
  { value: 'xet', label: 'Xet', detail: 'Xet only, even if HF_HUB_DISABLE_XET is set. A file not stored on Xet, or an hf without hf_xet, is not downloaded.' },
  { value: 'https', label: 'HTTPS', detail: 'Whole files over HTTPS, without Xet deduplication.' },
];
type Result = { answers?: Record<string, { type?: string; choice?: string; score?: number; noul?: number; confidence?: number; probabilities?: Record<string, number> }>; usage?: { input_tokens: number; output_tokens: number }; [key: string]: unknown };
// The Settings tab's choice for `any-available` when every ready instance is running a request.
const WHEN_BUSY = [
  { value: 'reject', label: 'Answer 503', detail: (_ms: number) => 'The request gets 503 at once, saying every loaded instance is busy.' },
  { value: 'wait', label: 'Wait for one', detail: (ms: number) => `The request waits for the first instance to finish, up to ${waitText(ms)}, then gets 503.` },
] as const;
// "Wait up to (ms)": the backend accepts 1–600000 (set_wait_ms in runtime.rs).
const WAIT_MS = { min: 1, max: 600000, step: 1000 };
const defaultSettings: Settings = { runtime_path: '', model_path: '', mmproj_path: '', mtp: false, mtp_path: '', decision_mode: 'auto', context_size: 8192, gpu_layers: 99, device: '', split_mode: 'layer', tensor_split: '', cpu_runtime_path: '', api_port: 11435, require_api_key: false, runtime_port: 11436, hf_path: 'hf', download_transport: 'auto', download_dir: '', library_path: '', keep_context: false, sha256_check: 'once', allow_force_fit: false, hide_crowded_load_warning: false, fall_through_to_other_instances: false, any_available_when_busy: 'reject', any_available_wait_ms: 120000 };
// The form is edited in two places, each saved on its own: the Loading
// tab's load panel (settings a load uses) and the Runtime tab (build and ports).
// keep_context is the default a new instance starts with; loaded instances
// have their own switch, so it is not compared for Reload.
const LOAD_FIELDS = ['model_path', 'mmproj_path', 'mtp', 'mtp_path', 'decision_mode', 'context_size', 'gpu_layers', 'device', 'split_mode', 'tensor_split', 'keep_context'] as const;
// Decision mode: stored value, UI label, hint under the segmented control.
const DECISION_MODES = [
  ['auto', 'Auto', 'Uses Prompt layout for Winnow GGUFs, otherwise Built-in when the model and runtime support it.'],
  ['native', 'Built-in', "The model's own decision support, answered by llama.cpp. The model and build must support it."],
  ['winnow', 'Prompt layout', 'Noullama prompts the model and reads the probability of each answer label (Winnow layout).'],
] as const;
const RUNTIME_FIELDS = ['runtime_path', 'cpu_runtime_path', 'library_path', 'api_port', 'runtime_port', 'hf_path'] as const;
const isLoadField = (key: keyof Settings) => (LOAD_FIELDS as readonly string[]).includes(key);
const pick = <K extends keyof Settings>(settings: Settings, keys: readonly K[]) => Object.fromEntries(keys.map(key => [key, settings[key]])) as Pick<Settings, K>;
const icons = [Route, ShieldCheck, Shield, Mail, Network, Plus];
const pages: Record<string, [string, string]> = {
  playground: ['Playground', 'Ask typed questions about a piece of content.'],
  models: ['Models', 'GGUF files on this machine.'],
  hub: ['Hub', 'Find and download decision models from Hugging Face.'],
  runtime: ['Runtime', 'llama.cpp builds, ports and the Hugging Face CLI.'],
  loading: ['Loading', 'Load models and choose how each one runs.'],
  api: ['Local API', 'Call loaded models from your own code.'],
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

class PlaygroundValue<T> {
  private listeners = new Set<() => void>();
  constructor(private value: T) {}
  get = () => this.value;
  set = (value: T) => {
    if (Object.is(value, this.value)) return;
    this.value = value;
    for (const listener of this.listeners) listener();
  };
  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };
}

function PlaygroundSubscriber<T>({ store, children }: { store: PlaygroundValue<T>; children: (value: T) => React.ReactNode }) {
  return children(useSyncExternalStore(store.subscribe, store.get));
}

function PlaygroundValidation({ store, validate, children }: {
  store: PlaygroundValue<string>; validate: (value: string) => string; children: (error: string) => React.ReactNode;
}) {
  return children(useSyncExternalStore(store.subscribe, () => validate(store.get())));
}

const noInstances: Instance[] = [];

function App() {
  const [selectedTab, setTab] = useState('playground');
  const tab = useDeferredValue(selectedTab);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  // Unsaved edits on the Runtime tab, and in the Loading tab's load panel.
  const [runtimeFormDirty, setRuntimeFormDirty] = useState(false);
  const [loadDirty, setLoadDirty] = useState(false);
  const [savingGroup, setSavingGroup] = useState(false);
  // The load panel's fit badge for its current values: the fit and how many
  // loaded instances it sets memory aside for, or why there is none.
  const [panelFit, setPanelFit] = useState<{ fit: Fit; others: number } | { error: string } | null>(null);
  const fitRequests = useRef<ReturnType<typeof latestOnly<{ settings: Partial<Settings>; replace: string | null }, { fit: Fit; others: number }>> | null>(null);
  const [preset, setPreset] = useState('triage');
  const [savedPresets, setSavedPresets] = useState<SavedPreset[]>([]);
  const [presetName, setPresetName] = useState('');
  const [presetToDelete, setPresetToDelete] = useState<SavedPreset | null>(null);
  // Keep context switches shown at once while set_instance_keep_context saves,
  // by instance id; the snapshot is the truth after.
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
  const [stateDraft] = useState(() => new PlaygroundValue(presets[0].state));
  const setContent = stateDraft.set;
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
  const [decisionResult] = useState(() => new PlaygroundValue<Result | null>(null));
  const setResult = decisionResult.set;
  const [elapsed, setElapsed] = useState(0);
  const [resultTab, setResultTab] = useState('decisions');
  const actionOwners = useRef(new PendingActions());
  const refreshRequests = useRef<Promise<void> | null>(null);
  const [pendingActions, setPendingActions] = useState<Set<string>>(new Set());
  const [visibleActions, setVisibleActions] = useState<Set<string>>(new Set());
  const showBusy = (name: string) => visibleActions.has(name);
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
  const [inspecting, setInspecting] = useState(false);
  const inspectRequest = useRef(0);
  const sessionRequest = useRef(0);
  const lastScan = useRef<Parameters<typeof shouldScanModels>[0]>(null);
  const scanRequest = useRef(0);
  const libraryScroll = useRef({ top: 0, estimatedHeight: 0 });
  const snapshotText = useRef<string | null>(null);
  const [localModels, setLocalModels] = useState<{ path: string; name: string; size: number; mtp?: boolean; mtp_head?: boolean; decision?: string | null }[]>([]);
  const [mtpResult, setMtpResult] = useState<{ path: string; built_in: boolean; head: string | null }>({ path: '', built_in: false, head: null });
  const selectedMtp = mtpResult.path === settings.model_path ? mtpResult : { built_in: false, head: null };
  const [scanWarnings, setScanWarnings] = useState<string[]>([]);
  const [session, setSession] = useState<Session | null>(null);
  const [token, setToken] = useState('');
  const [tokenOpen, setTokenOpen] = useState(false);
  const tokenFormId = React.useId();
  useEffect(() => { if (tab !== 'hub') setTokenOpen(false); }, [tab]);
  type RuntimeProbe = { path: string; libraryPath: string; output: string };
  const [runtimeProbe, setRuntimeProbe] = useState<RuntimeProbe | null>(null);
  const [cpuProbe, setCpuProbe] = useState<RuntimeProbe | null>(null);
  const version = matchingProbe(runtimeProbe, settings.runtime_path, settings.library_path);
  const cpuVersion = matchingProbe(cpuProbe, settings.cpu_runtime_path, settings.library_path);
  // The saved GPU runtime's devices for the load panel's "Device" select,
  // with the runtime and library folders they were read for.
  const [deviceList, setDeviceList] = useState<{ key: string; devices: RuntimeDevice[]; error: string | null } | null>(null);
  const [copied, setCopied] = useState('');
  const [updates, setUpdates] = useState<UpdateView | null>(null);
  const [updatesOpen, setUpdatesOpen] = useState(false);
  const desktop = isTauri();
  const pageVisible = usePageVisible();
  // Rescan local models when a download finishes.
  const finishedDownloads = useMemo(() => (snapshot?.downloads ?? []).flatMap(d => d.path && !d.active && !d.error && !d.cancelled ? [d.path] : []), [snapshot?.downloads]);

  useEffect(() => { if (tab !== 'library' || !libraryScroll.current.top) window.scrollTo({ top: 0, behavior: 'instant' }); }, [tab]);
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

  async function refresh(poll = false) {
    if (!desktop) return;
    while (refreshRequests.current) {
      if (poll) return;
      await refreshRequests.current.catch(() => {});
    }
    const request = Promise.allSettled([
      invoke<string>('snapshot').then(raw => {
        if (raw === snapshotText.current) return;
        const next = JSON.parse(raw) as Snapshot;
        snapshotText.current = raw;
        setSnapshot(current => retainSnapshotSections(current, { ...next }));
      }),
      invoke<UpdateView>('update_state').then(next => setUpdates(current => retainEqual(current, next))),
    ]).then(results => {
      const failure = results.find(result => result.status === 'rejected');
      if (failure) throw failure.reason;
    }).finally(() => { refreshRequests.current = null; });
    refreshRequests.current = request;
    await request;
  }
  async function updateCommand(command: 'check_for_update' | 'install_update') {
    try { const next = await invoke<UpdateView | null>(command); if (next) setUpdates(next); }
    // Check and install errors appear in the dialog's update state.
    catch {}
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
    refresh(true).catch(() => {});
    const timer = setInterval(() => { refresh(true).catch(() => {}); }, 1200);
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
    const unchanged = (keys: readonly (keyof Settings)[]) => settingsGroupMatches(settings, savedSettings, keys, savingGroup);
    if (loadDirty && unchanged(LOAD_FIELDS)) setLoadDirty(false);
    if (runtimeFormDirty && unchanged(RUNTIME_FIELDS)) setRuntimeFormDirty(false);
  }, [settings, savedKey, loadDirty, runtimeFormDirty, savingGroup]);
  useEffect(() => { if (snapshot) setDownloadDir(snapshot.settings.download_dir); }, [snapshot?.settings.download_dir]);
  // Read when the Loading tab is shown and whenever the saved GPU runtime
  // changes; the backend keeps the list until the runtime changes.
  const deviceKey = savedSettings ? `${savedSettings.runtime_path}\n${savedSettings.library_path}` : null;
  useEffect(() => {
    if (!desktop || tab !== 'loading' || deviceKey === null) return;
    let live = true;
    setDeviceList(current => current?.key === deviceKey ? current : null);
    invoke<{ devices: RuntimeDevice[]; error: string | null }>('runtime_devices')
      .then(list => { if (live) setDeviceList({ key: deviceKey, devices: list.devices, error: list.error }); })
      .catch(e => { if (live) setDeviceList({ key: deviceKey, devices: [], error: errorText(e) }); });
    return () => { live = false; };
  }, [tab, deviceKey]);
  useEffect(() => {
    const none = { path: settings.model_path, built_in: false, head: null };
    if (!desktop || !settings.model_path) {
      setMtpResult(current => equalJson(current, none) ? current : none);
      return;
    }
    let live = true;
    const read = () => invoke<typeof none>('model_mtp', { path: settings.model_path }).then(found => {
      if (live) setMtpResult(current => retainEqual(current, { ...found, path: settings.model_path }));
    }).catch(() => { if (live) setMtpResult(current => retainEqual(current, none)); });
    if (mtpResult.path !== settings.model_path) void read();
    const timer = tab === 'models' && pageVisible ? setInterval(read, 5000) : undefined;
    return () => { live = false; clearInterval(timer); };
  }, [settings.model_path, tab === 'models' && pageVisible]);
  // A session answer belongs to the hf path it was checked with: a new saved
  // path drops it, and an answer for an older request is ignored.
  const savedHfPath = snapshot?.settings.hf_path;
  function checkSession() {
    const request = ++sessionRequest.current;
    return invoke<Session>('hf_session').then(next => { if (request === sessionRequest.current) setSession(next); });
  }
  useEffect(() => { sessionRequest.current++; setSession(null); }, [savedHfPath]);
  useEffect(() => { if (desktop && tab === 'hub' && !session) checkSession().catch(e => setNotice({ text: errorText(e), error: true })); }, [tab, session === null]);
  function scanModels() {
    const request = ++scanRequest.current;
    invoke<{ models: typeof localModels; warnings: string[] }>('local_models').then(scan => {
      if (request !== scanRequest.current) return;
      setLocalModels(current => equalJson(current, scan.models) ? current : scan.models);
      setScanWarnings(current => equalJson(current, scan.warnings) ? current : scan.warnings);
    }).catch(e => {
      if (request === scanRequest.current) setNotice({ text: errorText(e), error: true });
    });
  }
  useEffect(() => {
    if (!desktop) return;
    const current = { models: tab === 'models', folder: snapshot?.settings.download_dir, finished: finishedDownloads };
    if (shouldScanModels(lastScan.current, current)) scanModels();
    lastScan.current = current;
  }, [tab, snapshot?.settings.download_dir, finishedDownloads]);
  useEffect(() => {
    if (!desktop || tab !== 'models' || !pageVisible) return;
    const timer = setInterval(scanModels, 5000);
    return () => clearInterval(timer);
  }, [tab, pageVisible]);
  // "Wait up to (ms)" saves on its own, once typing pauses, so half-typed
  // numbers are not sent. The draft shows until the saved value comes back.
  // The quit prompt opens on the click, or on the backend's quit-requested
  // event (tray, window close), without waiting for a snapshot. A snapshot
  // that shows a pending confirmation also opens it, in case an event was missed.
  const [quitPrompt, setQuitPrompt] = useState(false);
  useEffect(() => {
    if (!desktop) return;
    const stop = listen('quit-requested', () => setQuitPrompt(true));
    return () => { void stop.then(unlisten => unlisten()); };
  }, []);
  useEffect(() => { if (snapshot?.quit_confirmation) setQuitPrompt(true); }, [snapshot?.quit_confirmation]);
  const showWriteError = (reason: unknown) => setNotice({ text: errorText(reason), error: true });
  const settingWrites = useOptimisticWrites<Settings>(showWriteError);
  const contextWrites = useOptimisticWrites<Record<string, boolean>>(showWriteError);
  type UpdatePrefs = Pick<UpdateView, 'include_prereleases' | 'check_on_launch'>;
  const preferenceWrites = useOptimisticWrites<{ prefs: UpdatePrefs }>(showWriteError);
  const shownSetting = <K extends keyof Settings>(key: K, fallback: Settings[K]): Settings[K] =>
    settingWrites.drafts[key] ?? snapshot?.settings[key] ?? fallback;
  const transport = shownSetting('download_transport', 'auto');
  function changeSetting<K extends keyof Settings>(key: K, value: Settings[K], command: string, args: Record<string, unknown>) {
    if (!desktop) { showWriteError('Launch the desktop app with npm run desktop to use the local backend.'); return; }
    settingWrites.write(key, value, async () => { await invoke(command, args); await refresh(); });
  }
  function changeUpdatePrefs(includePrereleases: boolean, checkOnLaunch: boolean) {
    preferenceWrites.write('prefs', { include_prereleases: includePrereleases, check_on_launch: checkOnLaunch }, async () => {
      const next = await invoke<UpdateView>('set_update_prefs', { includePrereleases, checkOnLaunch });
      setUpdates(next);
      await refresh();
    });
  }
  const whenBusy = shownSetting('any_available_when_busy', 'reject');
  const shownWait = shownSetting('any_available_wait_ms', 120000);
  const [waitDraft, setWaitDraft] = useState<number | null>(null);
  useEffect(() => {
    if (waitDraft === null) return;
    if (waitDraft === shownWait) { setWaitDraft(null); return; }
    const timer = setTimeout(() => {
      changeSetting('any_available_wait_ms', waitDraft, 'set_any_available_wait_ms', { ms: waitDraft });
      setWaitDraft(null);
    }, 600);
    return () => clearTimeout(timer);
  }, [waitDraft]);
  async function action(name: string, task: () => Promise<void>) {
    if (!desktop) { setNotice({ text: 'Launch the desktop app with npm run desktop to use the local backend.', error: true }); return; }
    if (!actionOwners.current.begin(name)) return;
    setPendingActions(actionOwners.current.snapshot());
    const timer = window.setTimeout(() => setVisibleActions(current => new Set(current).add(name)), busyDelay);
    try { await task(); await refresh(); }
    catch (e) { setNotice({ text: errorText(e), error: true }); }
    finally {
      window.clearTimeout(timer);
      actionOwners.current.end(name);
      setPendingActions(actionOwners.current.snapshot());
      setVisibleActions(current => {
        if (!current.has(name)) return current;
        const next = new Set(current);
        next.delete(name);
        return next;
      });
    }
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
  async function saveGroup(keys: readonly (keyof Settings)[]) {
    setSavingGroup(true);
    try {
      await invoke('save_settings', { settings: pick(settings, keys) });
      await refresh();
    } finally {
      setSavingGroup(false);
    }
  }
  const saveLoadSettings = () => saveGroup(LOAD_FIELDS);
  const saveRuntimeSettings = () => saveGroup(RUNTIME_FIELDS);
  async function run() {
    await action('decide', async () => {
      const payload = { ...buildRequest(questions, stateDraft.get(), stateMode), ...(visionLoaded && images.length ? { images: images.map(image => image.url) } : {}) };
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
      const request = buildRequest(questions, stateDraft.get(), stateMode);
      const next = await invoke<SavedPreset[]>('save_preset', { preset: { id: preset.startsWith('saved-') ? preset : '', name: presetName, questions: request.questions, state: stateDraft.get(), state_mode: stateMode } });
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
  function toggleKeepContext(id: string, on: boolean) {
    if (!desktop) { showWriteError('Launch the desktop app with npm run desktop to use the local backend.'); return; }
    contextWrites.write(id, on, async () => { await invoke('set_instance_keep_context', { id, on }); await refresh(); });
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
  // These commands only use the desktop flag, stable setters, and refresh refs.
  const libraryActions = useMemo(() => ({
    onSave: saveLibrary,
    onReload: () => libraryCommand('reload_library'),
    onOpenFile: () => libraryCommand('open_models_file'),
    onOpenBackups: () => libraryCommand('open_backups_folder'),
    onRecordSha: (path: string, slot: string) => libraryCommand('record_sha256', { path, slot }),
    onBrowse: () => invoke<string | null>('pick_model_file'),
    onError: copyError,
  }), [desktop]);
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
  // The newest request wins: opening another repository while one is still
  // being read (from the Hub list or Loading's Hub button) replaces it.
  async function openRepo(id: string) {
    if (!desktop) { setNotice({ text: 'Launch the desktop app with npm run desktop to use the local backend.', error: true }); return; }
    const request = ++inspectRequest.current;
    const timer = window.setTimeout(() => { if (request === inspectRequest.current) setInspecting(true); }, busyDelay);
    try {
      const next = await invoke<Inspection>('inspect_model', { contextSize: settings.context_size, gpuLayers: settings.gpu_layers, repo: id });
      if (request === inspectRequest.current) setInspection(next);
    } catch (e) {
      if (request === inspectRequest.current) setNotice({ text: errorText(e), error: true });
    } finally {
      window.clearTimeout(timer);
      if (request === inspectRequest.current) setInspecting(false);
    }
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
    try { await invoke('download_model', { repo, filename: file, transport }); await refresh(); }
    catch (e) { setNotice({ text: errorText(e), error: true }); }
    finally { setPreparing(current => current.filter(other => other !== id)); }
  }
  function switchStateMode(next: 'text' | 'json') {
    try {
      const content = stateDraft.get();
      const converted = convertState(content, stateMode, next, stateMemory?.json, stateMemory?.text);
      if (stateMode === 'json' && next === 'text') setStateMemory({ json: content, text: converted });
      setContent(converted); setStateMode(next);
    } catch (e) { setNotice({ text: errorText(e), error: true }); }
  }
  // Loads the selected model beside any already loaded, after saving the load
  // panel. The backend validates the new runtime before ejecting `replace`.
  // `ejectOther` comes from the crowded-load
  // dialog. The new instance becomes the Playground's model unless another
  // loaded model is already chosen there.
  async function loadSelected({ replace = null, ejectOther }: { replace?: string | null; ejectOther?: string } = {}) {
    if (splitBlocked) return;
    await action('start', async () => {
      await saveLoadSettings();
      if (ejectOther) await invoke('unload_model', { id: ejectOther });
      // The instance being reloaded is unloaded first, so the check leaves it out.
      const warning = await invoke<CrowdedWarning | null>('crowded_load_warning', { replace });
      if (warning) { setHideCrowded(false); setCrowdedWarning({ ...warning, replace }); return; }
      setCrowdedWarning(null);
      // It may have stopped on its own meanwhile; then there is nothing to eject.
      const slot = reloadSlot(instances, replace);
      const id = await invoke<string>('start_runtime', { slot, replace });
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
  const instances = snapshot?.instances ?? noInstances;
  const libraryLoadedPaths = useMemo(() => instances.map(instance => instance.path), [instances]);
  const libraryCompanions = useMemo(() => localModels.flatMap((file): Companion[] =>
    file.mtp_head ? [{ path: file.path, kind: 'mtp' }] : isProjector(file.path) ? [{ path: file.path, kind: 'mmproj' }] : []), [localModels]);
  const running = instances.length > 0;
  const readyCount = instances.filter(i => i.ready).length;
  const modelMissing = !loadDirty && !!snapshot?.model_missing;
  // The selected file's loaded model, if it is loaded.
  // Several instances of one file can be loaded; the load bar acts on one, chosen in the panel.
  const { list: selectedInstances, selected: selectedInstance } = instancesOf(instances, settings.model_path, loadTarget);
  const model = modelStatus(settings.model_path, selectedInstance, modelMissing);
  // The repository the Hub tab downloaded the selected model from, for Loading's Hub button.
  const [selectedRepo, setSelectedRepo] = useState<{ path: string; repo: string | null } | null>(null);
  useEffect(() => {
    const path = model.selected;
    if (!desktop || !path) return;
    let live = true;
    invoke<string | null>('model_repo', { path }).then(repo => { if (live) setSelectedRepo({ path, repo }); }).catch(() => {});
    return () => { live = false; };
  }, [model.selected, finishedDownloads]);
  const hubRepo = selectedRepo && selectedRepo.path === model.selected ? selectedRepo.repo : null;
  // Whether a load now would differ from what this instance was loaded with:
  // the load panel's values, plus the saved build (unsaved Runtime tab edits
  // are not used by a load).
  // An instance's settings.device is the device it was loaded on. On the CPU
  // it ran the CPU runtime with 0 GPU layers, so those are what compare.
  const deviceListed = deviceList?.key === deviceKey ? deviceList.devices : [];
  const device = deviceChoice(settings.device, deviceList?.key === deviceKey ? deviceList.devices : null);
  const saved = savedSettings ?? defaultSettings;
  const savedCpuRuntime = saved.cpu_runtime_path.trim() ? saved.cpu_runtime_path : !settings.device.trim() ? saved.runtime_path : '';
  const differsFrom = (instance: Instance) => instance.path === settings.model_path && (
    (['mmproj_path', 'mtp', 'mtp_path', 'decision_mode', 'context_size'] as const).some(key => settings[key] !== instance.settings[key])
    || (instance.settings.device !== CPU_DEVICE && settings.gpu_layers !== instance.settings.gpu_layers)
    || (device.effective !== null && device.effective !== instance.settings.device)
    || (isSplit(instance.settings.device) && (settings.split_mode !== instance.settings.split_mode || settings.tensor_split.trim() !== instance.settings.tensor_split.trim()))
    || (instance.settings.device === CPU_DEVICE ? savedCpuRuntime : saved.runtime_path) !== instance.runtime_path
    || saved.library_path !== instance.settings.library_path);
  const reloadSelected = !!selectedInstance && differsFrom(selectedInstance);
  // The panel's load: a reload of the picked instance, or a fresh load.
  const replaceOnLoad = panelReplace(selectedInstance, reloadSelected);
  // The fit is worked out by the backend for the panel's current values, the
  // device the load would use and the instance this load replaces. Typing
  // waits for a pause, one request runs at a time, and the badge keeps its
  // last result until the new one arrives.
  const fitArgs = JSON.stringify({ settings: pick(settings, LOAD_FIELDS), replace: replaceOnLoad });
  const fitContext = `${savedKey}\n${instances.map(i => i.id).join(',')}\n${!!snapshot?.hardware}`;
  useEffect(() => {
    if (!desktop) return;
    if (!settings.model_path) { fitRequests.current?.cancel(); setPanelFit(null); return; }
    fitRequests.current ??= latestOnly(250, args => invoke<{ fit: Fit; others: number }>('load_fit', args), result => setPanelFit(result.ok ? result.value : { error: errorText(result.error) }));
    fitRequests.current.request(JSON.parse(fitArgs));
  }, [fitArgs, fitContext]);
  useEffect(() => () => fitRequests.current?.cancel(), []);
  const fit = panelFit && 'fit' in panelFit ? panelFit.fit : null;
  const otherLoaded = panelFit && 'fit' in panelFit ? panelFit.others : 0;
  const fitDetail = fit ? `${fit.detail}${otherLoaded ? ` Counts the memory ${otherLoaded === 1 ? 'another loaded model holds' : `${otherLoaded} other loaded models hold`}.` : ''}` : '';
  const fitNote = !model.selected ? 'Select a model to see how it would run on this machine.' : fit ? fitDetail : panelFit && 'error' in panelFit ? panelFit.error : '';
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
  // An instance on Device CPU ran the CPU runtime, so that is the selection it is compared with.
  const chosenRuntime = runtimeStatus(selectedInstance?.settings.device === CPU_DEVICE ? (settings.cpu_runtime_path.trim() ? settings.cpu_runtime_path : settings.runtime_path) : settings.runtime_path, selectedInstance?.runtime_path ?? null, !!selectedInstance, snapshot?.data_dir ?? '');
  const runtimeName = (path: string) => { const r = runtimeStatus(path, null, false, snapshot?.data_dir ?? ''); return path ? `${r.source} · ${r.name}` : r.name; };
  // The build a load uses: the saved one.
  const loadRuntime = runtimeName(savedSettings?.runtime_path ?? '');
  const listError = deviceList?.key === deviceKey && deviceList.error ? `The llama.cpp build in Runtime could not list its devices (${deviceList.error}).` : '';
  // The Device row's live state; the static explanation is its description.
  // A split runs on 2 or more GPUs; its ratio has one number per GPU.
  const splitting = !!device.split && device.split.gpus.length >= 2;
  const splitRatioError = splitting ? ratioError(settings.tensor_split, device.split!.gpus.length) : null;
  const splitModeError = splitting && !SPLIT_MODES.some(([mode]) => mode === settings.split_mode);
  // A split that cannot load as set blocks the load buttons.
  const splitBlocked = !!device.split && (device.effective === null || !!device.problem || !!splitRatioError || splitModeError);
  const deviceStatus: { tone: Tone; text: string } | null = device.effective === null ? { tone: 'pending', text: 'Checking which devices the llama.cpp build reports…' }
    : device.problem ? { tone: 'error', text: device.problem }
    : device.split ? { tone: 'info', text: `Splits the model across ${device.split.label}.` }
    : device.missing ? { tone: 'error', text: `${settings.device} was not found: the llama.cpp build in Runtime does not report it. Loads are refused until you choose another device.` }
    : device.effective === CPU_DEVICE
      ? savedCpuRuntime.trim() ? { tone: 'info', text: `Runs on the CPU runtime (${runtimeName(savedCpuRuntime)}) with GPU layers 0.` } : { tone: 'warn', text: 'Device CPU needs a CPU runtime. Choose or download one under CPU runtime in Runtime.' }
      : null;
  const keepsContext = (instance: Instance) => contextWrites.drafts[instance.id] ?? instance.keep_context;
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
  const requireApiKey = shownSetting('require_api_key', false);
  const authHeader = requireApiKey ? ' -H "Authorization: Bearer $NOULLAMA_API_KEY"' : '';
  const pythonExample = tab === 'api' ? `import json\nfrom openai import OpenAI\n\nclient = OpenAI(base_url="${apiBase}", api_key="${requireApiKey ? "YOUR_API_KEY" : "local"}")\n\nrequest = {\n    "state": "I was charged twice. Please refund the duplicate.",\n    "questions": {\n        "refund": {\n            "type": "noul",\n            "instructions": "Is the customer requesting a refund?"\n        }\n    }\n}\n\nresponse = client.chat.completions.create(\n    model="${exampleModel}",\n    messages=[{"role": "user", "content": json.dumps(request)}],\n    stream=False,\n)\ndecision = json.loads(response.choices[0].message.content)\nprint(decision["answers"])` : '';
  const questionValidation = useMemo(() => {
    try { return { count: Object.keys(buildRequest(questions, '', 'text').questions as object).length, error: '' }; }
    catch (e) { return { count: 0, error: errorText(e) }; }
  }, [questions]);
  const questionError = questionValidation.error;
  const requestJson = useMemo(() => {
    if (tab !== 'api') return '';
    try { return pretty({ model: exampleModel, ...buildRequest(questions, stateDraft.get(), stateMode) }); }
    catch { return '{}'; }
  }, [tab, exampleModel, questions, stateDraft, stateMode]);
  const validateDraft = useMemo(() => {
    let previous: string | undefined;
    let error = '';
    return (content: string) => {
      if (questionError || stateMode === 'text') return questionError;
      if (content !== previous) {
        previous = content;
        try { JSON.parse(content); error = ''; }
        catch (e) { error = errorText(e); }
      }
      return error;
    };
  }, [questionError, stateMode]);

  const [pageTitle, pageSubtitle] = pages[tab];
  const repoLike = /^[\w.-]+\/[\w.-]+$/.test(query.trim()) && !models.some(m => m.id === query.trim());
  const downloads = snapshot?.downloads ?? [];
  const downloadFor = (repo: string, file: string) => downloads.find(d => d.id === `${repo}/${file}`);
  const anyDownloadActive = downloads.some(d => d.active);
  // Downloads shown in the floating list: running or finished, not dismissed,
  // and not already on screen in the repository being viewed.
  const toastDownloads = downloads.filter(d => (d.active || (!!d.path && !d.error && !d.cancelled)) && !toastDismissed.includes(d.id) && !(tab === 'hub' && inspection?.repo === d.repo));
  const showDownloadToast = toastDownloads.length > 0 && !showLogs;
  const hubRow = (group: string, id: string, meta: React.ReactNode, fitInfo: Fit | null, fitTitle?: string) => <button key={`${group}:${id}`} className={`hub-row ${inspection?.repo === id ? 'selected' : ''}`} onClick={() => openRepo(id)}><AuthorLogo author={id.split('/')[0]} size={34}/><span className="hub-row-name"><strong>{id.split('/').slice(1).join('/') || id}</strong><small>{id.split('/')[0]}</small></span><span className="hub-row-meta">{meta}</span>{fitInfo && <span className={`fit-dot fit-level-${fitInfo.level}`} title={fitTitle ?? fitInfo.detail}><span className="fit-bars" aria-hidden="true">{[1, 2, 3, 4].map(n => <i key={n} className={n <= fitInfo.level ? 'on' : ''}/>)}</span><span className="sr-only">{fitInfo.label}</span></span>}<span className="selected-badge selection-mark" aria-label={inspection?.repo === id ? "Selected" : undefined} aria-hidden={inspection?.repo !== id}><Check size={14} weight="bold"/></span></button>;

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
      <SidebarTabs tab={selectedTab} onSelect={setTab}/>
      <div className="sidebar-bottom">
        <Reveal open={loadErrors.length > 0}>{loadErrors.length > 0 && <button className="nav-item load-error-nav" title={`${loadErrors.length === 1 ? `Load error: ${loadErrors[0].name}` : `${loadErrors.length} load errors`}. Open Loading`} aria-label={`${loadErrors.length === 1 ? '1 load error' : `${loadErrors.length} load errors`}, open Loading`} onClick={() => setTab('loading')}><span className="nav-tile"><AlertCircle size={18} weight="duotone"/></span><span>{loadErrors.length === 1 ? 'Load error' : `${loadErrors.length} load errors`}</span></button>}</Reveal>
        {desktop &&<button className={`nav-item ${updates?.offer ? 'has-update' : ''}`} title={updates?.offer ? `Update to ${updates.offer.version}` : 'Updates'} aria-haspopup="dialog" onClick={() => setUpdatesOpen(true)}><span className="nav-tile"><CircleArrowUp size={18} weight="duotone"/></span><span>{updates?.offer ? `Update to ${updates.offer.version}` : 'Updates'}</span></button>}
        <button className="nav-item" title="Runtime logs" aria-label="Runtime logs" aria-pressed={showLogs} onClick={() => setShowLogs(v => !v)}><span className="nav-tile"><Terminal size={18} weight={showLogs ? 'fill' : 'duotone'}/></span><span>Runtime logs</span></button>
        <button className="nav-item" title="Quit Noullama" aria-label="Quit Noullama" {...busyButton(isBusy('quit', 'quit-prompt'), !desktop)} onClick={() => { setQuitPrompt(true); void action('quit-prompt', async () => { await invoke('request_quit'); }); }}><span className="nav-tile"><Power size={18} weight="duotone"/></span><span>Quit Noullama</span></button>
        <div className="runtime-card-wrap" onMouseEnter={() => setDetailsOpen(true)} onMouseLeave={() => setDetailsOpen(false)} onFocus={() => setDetailsOpen(true)} onBlur={() => setDetailsOpen(false)}>
          <button className="runtime-card" onClick={() => setTab('runtime')} aria-label={`Selected llama.cpp: ${chosenRuntime.source} ${chosenRuntime.name}${chosenRuntime.inUse ? ', in use' : ', not running'}`}><span className={`runtime-card-icon ${readyCount ? 'ready' : running ? 'loading' : ''}`}><Cpu size={18} weight="duotone"/></span><span><strong>{readyCount > 1 ? `${readyCount} models ready` : readyCount ? 'Model ready' : running ? instances.some(i => !i.ready && i.verifying) ? 'Checking sha256' : 'Loading model' : 'Runtime offline'}</strong><small>llama.cpp {chosenRuntime.source} · {chosenRuntime.name}</small>{chosenRuntime.pending && <small className="runtime-pending">Next selection not applied</small>}</span></button>
          {detailsOpen && <ModelDetails
            name={basename(shown?.path || settings.model_path || '') }
            status={shown ? `${shown.ready ? 'Ready' : verifyingText(shown.verifying) ?? 'Loading'}${instances.length > 1 ? ` · 1 of ${instances.length} loaded` : ''}` : model.selected ? 'Not loaded' : 'No model selected'}
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
          <button type="button" className="sidebar-loaded-name" title={`${apiName(instance)}: ${instance.path}`} onClick={() => { setPlayModel(instance.id); setTab('loading'); }}>{apiName(instance)}</button>
          <button type="button" className="icon-button" aria-label={`Eject ${apiName(instance)}`} title={`Eject ${apiName(instance)}`} {...busyButton(isBusy(`eject:${instance.id}`))} onClick={() => void eject(instance.id)}><Unplug size={14}/></button>
        </li>)}</ul>}
      </div>
    </aside>
    <main>
      <div className={`page-content page-${tab}${selectedTab !== tab ? ' page-leaving' : ''}`}>
        <header className="page-heading">
          <div><h1>{pageTitle}</h1><p>{pageSubtitle}</p></div>
          {hardware && <div className="stat-chips" aria-label="This machine">{hardware.vram_bytes ? <span className="stat-chip" title={hardware.gpu ?? 'Graphics memory'}><Gauge size={14}/><strong>{gb(hardware.vram_bytes)}</strong>VRAM</span> : <span className="stat-chip"><Gauge size={14}/>No GPU memory</span>}{hardware.ram_bytes > 0 && <span className="stat-chip"><MemoryStick size={14}/><strong>{gb(hardware.ram_bytes)}</strong>RAM</span>}{hardware.cpu_cores > 0 && <span className="stat-chip"><Microchip size={14}/><strong>{hardware.cpu_cores}</strong>CPU threads</span>}</div>}
        </header>
        <Toast banner={banner} onDismiss={() => { setNotice(null); setDismissedStartup(true); setDismissedApiNotice(true); setDismissedApi(snapshot?.api_error ?? null); setPreviewDismissed(true); }}/>

        {tab === 'playground' && <div className="playground">
          <div className="playground-main">
            <PlaygroundSubscriber store={stateDraft}>{content => {
              const jsonError = validateDraft(content);
              return <>
            <section className={`panel state-panel ${dropping ? 'dropping' : ''}`} onDragOver={event => { if (!visionLoaded || !event.dataTransfer.types.includes('Files')) return; event.preventDefault(); setDropping(true); }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setDropping(false); }} onDrop={event => { if (!visionLoaded) return; event.preventDefault(); setDropping(false); addImages([...event.dataTransfer.files]); }}><div className="panel-heading"><span><Terminal size={16}/> State <span className="muted normal-weight">The content to evaluate</span></span><SegmentedControl label="State format"><button className={stateMode === 'text' ? 'on' : ''} aria-pressed={stateMode === 'text'} onClick={() => switchStateMode('text')}>Text</button><button className={stateMode === 'json' ? 'on' : ''} aria-pressed={stateMode === 'json'} onClick={() => switchStateMode('json')}>JSON</button></SegmentedControl></div>{stateMode === 'json' ? <div className="state-json"><JsonEditor label="State JSON" value={content} onChange={value => { setContent(value); setResult(null); }}/></div> : <textarea aria-label="State content" value={content} spellCheck={false} placeholder={visionLoaded ? 'Content to evaluate, or leave empty to ask about the images only…' : 'Content to evaluate…'} onPaste={event => { const files = [...event.clipboardData.files].filter(file => file.type.startsWith('image/')); if (visionLoaded && files.length) { if (!event.clipboardData.types.includes('text/plain')) event.preventDefault(); addImages(files); } }} onChange={e => { setContent(e.target.value); setResult(null); }}/>}<Reveal open={visionLoaded}>{visionLoaded && <ImageAttachments images={images} onFiles={addImages} onRemove={index => { setImages(current => current.filter((_, i) => i !== index)); setResult(null); }}/>}</Reveal><div className="state-footer"><span>{content.length.toLocaleString()} characters<span className="separator">/</span>{stateMode === 'text' ? 'Plain text' : 'Structured JSON'}{visionLoaded && images.length > 0 && <><span className="separator">/</span>{images.length} {images.length === 1 ? 'image' : 'images'}</>}</span><div className="state-run"><button className="ghost" onClick={() => { setContent(''); setImages([]); setResult(null); }}>Clear</button><button className="primary" {...busyButton(isBusy('decide'), !!jsonError || !ready)} onClick={run}>{showBusy('decide') ? <RefreshCw className="spin" size={15}/> : <Play size={15}/>} {showBusy('decide') ? 'Deciding…' : 'Run decision'}</button></div></div></section>
            <Reveal open={jsonError || !ready} swapKey={String(!!jsonError)}>{(jsonError || !ready) && <p className="run-hint">{jsonError ? <span className="invalid">{jsonError}</span> : 'Load a model to run a decision.'}</p>}</Reveal>
              </>;
            }}</PlaygroundSubscriber>
            <PlaygroundSubscriber store={decisionResult}>{result =>
            <section className="panel results-panel"><div className="panel-heading"><span><Activity size={16}/> Result {result && <span className="chip success">Complete</span>}</span><div className="heading-actions">{result && <span className="timing">{elapsed.toFixed(1)} ms</span>}<SegmentedControl label="Result format"><button className={resultTab === 'decisions' ? 'on' : ''} aria-pressed={resultTab === 'decisions'} onClick={() => setResultTab('decisions')}>Decisions</button><button className={resultTab === 'json' ? 'on' : ''} aria-pressed={resultTab === 'json'} onClick={() => setResultTab('json')}>JSON</button></SegmentedControl>{result && <button className="icon-button" aria-label="Copy result" onClick={() => copy(pretty(result), 'result')}>{copied === 'result' ? <Check size={14} weight="bold"/> : <Copy size={14}/>}</button>}</div></div><Reveal open swapKey={result ? resultTab : "empty"}>{!result ? <div className="empty-result"><Activity size={17}/><p>{ready ? 'Run a decision to view results.' : 'Results appear here after a run.'}</p></div> : resultTab === 'json' ? <div className="result-code"><JsonEditor label="Result JSON" value={pretty(result)} readOnly/></div> : <div className="decision-results">{Object.entries(result.answers ?? {}).map(([name, answer]) => <div className="answer" key={name}><div className="answer-heading"><span>{name}</span><strong>{answer.choice ?? (answer.noul !== undefined ? answer.noul >= 0.5 ? `${(answer.noul * 100).toFixed(1)}% yes` : `${((1 - answer.noul) * 100).toFixed(1)}% no` : answer.score?.toFixed(2))}</strong>{answer.confidence !== undefined && <small>{(answer.confidence * 100).toFixed(1)}% confidence</small>}</div>{Object.entries(answer.probabilities ?? (answer.noul !== undefined ? { yes: answer.noul, no: 1 - answer.noul } : {})).map(([key, value]) => <div className="probability" key={key}><span>{key}</span><div><i style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}/></div><code>{(value * 100).toFixed(1)}%</code></div>)}</div>)}<div className="usage">{result.usage?.input_tokens.toLocaleString()} input tokens <span>/</span> {result.usage?.output_tokens ?? 0} generated tokens <span>/</span> {elapsed.toFixed(1)} ms round trip</div></div>}</Reveal></section>
            }</PlaygroundSubscriber>
          </div>
          <aside className="run-settings" aria-label="Run settings">
            <h2>Run settings</h2>
            <div className="rail-section rail-model"><div className="rail-label">Model</div>{instances.length ? <Select label="Playground model" value={activeInstance?.id ?? ''} options={instances.map(i => ({ value: i.id, label: `${apiName(i)}${i.slot > 1 ? ` · instance ${i.slot}` : ''}${i.ready ? '' : ' (loading)'}` }))} onChange={setPlayModel}/> : <p className="rail-note">No model loaded. Load one in <button type="button" className="ghost inline" onClick={() => setTab('loading')}>Loading</button>.</p>}<Reveal open={activeInstance && exampleFor(activeInstance)} className="rail-note-reveal">{activeInstance && exampleFor(activeInstance) && <p className="rail-note">In the local API, send <code>"model": "{exampleFor(activeInstance)}"</code>.</p>}</Reveal></div>
            <div className="rail-section"><div className="rail-label">Preset</div><div className="preset-chips">{presets.map((p, i) => { const Icon = icons[i]; return <button key={p.id} className={`preset-chip ${p.id === preset ? 'selected' : ''}`} aria-pressed={p.id === preset} title={p.subtitle} onClick={() => selectPreset(p.id)}><Icon size={15} weight={p.id === preset ? 'fill' : 'duotone'}/>{p.title}</button>; })}{savedPresets.map(p => <span className={`preset-chip saved ${p.id === preset ? 'selected' : ''}`} key={p.id}><button aria-pressed={p.id === preset} onClick={() => selectPreset(p.id)}><FileJson size={15} weight={p.id === preset ? 'fill' : 'duotone'}/>{p.name}</button><button className="icon-button small chip-remove" aria-label={`Delete preset ${p.name}`} title={`Delete ${p.name}`} {...busyButton(isBusy('preset'))} onClick={() => setPresetToDelete(p)}><X size={13}/></button></span>)}</div><Reveal open swapKey={preset}><p className="rail-note">{presets.find(p => p.id === preset)?.subtitle ?? (preset.startsWith('saved-') ? 'Custom preset' : 'Custom questions')}</p></Reveal></div>
            <div className="rail-section"><KeepContext on={!!activeInstance && keepsContext(activeInstance)} entries={playContext.entries} lastFlushed={playContext.last_flushed} scope={instances.length > 1 && activeInstance ? `for ${apiName(activeInstance)}` : undefined} disabled={!desktop || !activeInstance} onToggle={on => { if (activeInstance) void toggleKeepContext(activeInstance.id, on); }} onClear={() => { if (activeInstance) void clearContext(activeInstance.id); }}/></div>
            <section className="question-panel"><div className="rail-heading"><span className="rail-label">Questions <span className="code-badge">JSON</span></span><div className="heading-actions"><button className="ghost" onClick={() => { try { setQuestions(pretty(JSON.parse(questions))); } catch (e) { setNotice({ text: errorText(e), error: true }); } }}>Format</button><button className="icon-button" aria-label="Copy questions" onClick={() => copy(questions, 'questions')}>{copied === 'questions' ? <Check size={14} weight="bold"/> : <Copy size={14}/>}</button></div></div><JsonEditor label="Questions JSON" value={questions} onChange={value => { setQuestions(value); setResult(null); }}/><div className="editor-footer"><span className={questionError ? 'invalid' : 'valid'}><span className={`status-dot ${questionError ? 'red' : 'green'}`}/>{questionError ? 'Check JSON' : `${questionValidation.count} questions`}</span><span>choice, noul or score</span></div></section>
            <form className="preset-save" onSubmit={e => { e.preventDefault(); saveCustom(); }}><input aria-label="Custom preset name" placeholder="Preset name" maxLength={80} value={presetName} onChange={e => setPresetName(e.target.value)}/><PlaygroundValidation store={stateDraft} validate={validateDraft}>{jsonError => <button className="secondary compact" {...busyButton(isBusy('preset'), !!jsonError || !presetName.trim())}><Save size={13}/>{preset.startsWith('saved-') ? 'Save changes' : 'Save preset'}</button>}</PlaygroundValidation></form>
          </aside>
        </div>}
        {tab === 'models' && <>
          {modelFiles.length > 0 && <><div className="section-heading"><h2>On your machine</h2><span>{modelFiles.length} {modelFiles.length === 1 ? 'model' : 'models'}</span></div><div className="model-list">{modelFiles.map(model => { const hasProjector = localModels.some(file => isProjector(file.path) && folder(file.path) === folder(model.path)); return <div key={model.path} className={`local-model-entry${model.path === settings.model_path ? ' selected' : ''}`}><button className="model-row" onClick={() => { selectModel(model.path); setTab('loading'); }}><AuthorLogo author={accountFor(model.path)}/><span><strong>{nameFor(model.path) ?? model.name}</strong><small>{nameFor(model.path) && nameFor(model.path) !== model.name.replace(/\.gguf$/i, '') ? `${model.name} / ` : ''}{(model.size / 1e9).toFixed(2)} GB / <code>{model.path}</code></small></span>{hasProjector && <span className="chip" title="A multimodal projector is in the same folder. Turn it on in Loading.">Vision available</span>}{model.mtp && <span className="chip" title="This model has MTP layers, or a matching MTP head is in the same folder. Turn it on in Loading.">MTP available</span>}{model.decision === 'winnow-v1' && <span className="chip" title="No built-in decision support, so Decision mode Auto reads its answers through the Prompt layout (Winnow).">Prompt layout (Winnow)</span>}{model.decision == null && settings.decision_mode === 'auto' && <span className="chip warning" title="No built-in decision support and no prompt layout Noullama knows. Loading it will fail unless you set Decision mode in Loading.">Not a decision model</span>}{aliasesFor(model.path).length > 0 && <span className="chip" title={`API aliases: ${aliasesFor(model.path).join(', ')}`}>{aliasesFor(model.path).length === 1 ? aliasesFor(model.path)[0] : `${aliasesFor(model.path).length} aliases`}</span>}<span className={`pill model-choice${model.path === settings.model_path ? ' selected-badge' : ''}`}><span className="selection-mark"><Check size={14} weight="bold"/></span>{instances.filter(i => i.path === model.path).length > 1 ? `${instances.filter(i => i.path === model.path).length} loaded` : instances.some(i => i.path === model.path) ? 'Loaded' : model.path === settings.model_path ? 'Selected' : 'Select'}</span></button><button className="icon-button reveal-file" aria-label={`API names for ${model.name}`} title="API names" aria-expanded={aliasEditors.includes(model.path)} onClick={() => setAliasEditors(open => open.includes(model.path) ? open.filter(p => p !== model.path) : [...open, model.path])}><Tags size={18} weight="fill"/></button><CopyButton className="reveal-file" value={model.path} label={`Copy path of ${model.name}`} size={18} weight="fill" onError={copyError}/><button className="icon-button reveal-file" aria-label={`Show ${model.name} in folder`} title="Show in folder" {...busyButton(isBusy('reveal'))} onClick={() => revealFile(model.path)}><FolderOpen size={18} weight="fill"/></button><Reveal open={aliasEditors.includes(model.path)} className="model-alias-reveal">{aliasEditors.includes(model.path) && <div className="row-aliases"><ModelAliases path={model.path} aliases={aliasesFor(model.path)} routes={routes} disabled={!desktop} onSave={saveAliases} onError={copyError}/></div>}</Reveal></div>; })}</div></>}
          <Reveal open={modelFiles.length === 0}>{modelFiles.length === 0 && <div className="local-empty"><FolderOpen size={18}/><p>No models on this machine yet. Download one from the <button className="ghost" onClick={() => setTab('hub')}>Hub</button> or point Noullama at a GGUF file.</p></div>}</Reveal>
          <Reveal open={scanWarnings.length > 0}>{scanWarnings.length > 0 && <div className="hint" role="status">{scanWarnings.map(warning => <p key={warning}>{warning}</p>)}</div>}</Reveal>
          <div className="import-note"><span className="import-tile"><FolderOpen size={23} weight="fill"/></span><div><strong>Local GGUF</strong><p>Already have a GGUF file? Pick it here instead of downloading.</p></div><button className="ghost" {...busyButton(isBusy('browse'))} onClick={() => browseFile('model_path', 'loading')}>Browse local model<ArrowRight size={14}/></button></div>
          <section className="panel directory-panel"><label>Model download directory<div className="file-input-row"><input aria-label="Model download directory" value={downloadDir} placeholder={snapshot ? `${snapshot.data_dir}/models` : 'Default app model directory'} onChange={e => setDownloadDir(e.target.value)}/><button className="secondary" type="button" {...busyButton(isBusy('directory'), anyDownloadActive || downloadDir === snapshot?.settings.download_dir)} onClick={saveDownloadDirectory}>Save directory</button></div><small>New downloads use this directory. Existing files stay in place.</small></label></section>
        </>}
        {tab === 'hub' && <>
          <form className="hub-search" onSubmit={e => { e.preventDefault(); const q = query.trim(); action('search', async () => { setModels(await invoke('search_models', { query: q, contextSize: settings.context_size, gpuLayers: settings.gpu_layers })); setSearched(q); }); }}><div className="search-input"><Search size={17}/><input aria-label="Search decision models" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search decision models, or enter organization/repository"/>{query && <button type="button" className="icon-button" aria-label="Clear search" onClick={() => { setQuery(''); setModels([]); setSearched(''); }}><X size={15}/></button>}</div><button className="secondary" {...busyButton(isBusy('search'))}>{showBusy('search') ? 'Searching…' : 'Search'}</button></form>
          <div className="hub-panes">
            <section className="hub-list" aria-label="Repositories">
              <Reveal open={repoLike}>{repoLike && <><div className="pane-label">Repository</div>{hubRow('open', query.trim(), <>Open repository</>, null)}</>}</Reveal>
              <Reveal open={searched}>{searched && <div className="pane-label">{models.length ? `Results for “${searched}”` : `No decision models match “${searched}”`}</div>}</Reveal>
              {models.map(m => hubRow('result', m.id, <><Download size={12}/>{m.downloads?.toLocaleString() ?? 0}</>, m.fit, m.fit ? `Estimate for the smallest listed GGUF. ${m.fit.detail}` : undefined))}
              <div className="pane-label">Recommended</div>
              {hubRow('recommended', 'LiquidAI/d1-2B-GGUF', <>1.92B</>, null)}
              <div className="hf-card">
                <strong className="auth-heading"><HuggingFaceLogo size={16}/>{session?.authenticated ? `Connected as ${session.account}` : 'Hugging Face'}</strong>
                <p>{session && !session.cli_available ? 'hf CLI not found. Install it or set the executable path in Runtime.' : session?.authenticated ? 'Downloads use your Hugging Face session.' : 'Public models work without signing in. Uses HF_TOKEN or your hf auth login session.'}</p>
                <div className="hf-transport"><span>Download with</span><SegmentedControl label="Download with">{TRANSPORTS.map(option => <button key={option.value} type="button" className={transport === option.value ? 'on' : ''} aria-pressed={transport === option.value} disabled={!desktop} onClick={() => { if (transport !== option.value) void changeSetting('download_transport', option.value, 'set_download_transport', { mode: option.value }); }}>{option.label}</button>)}</SegmentedControl><Reveal open swapKey={`${transport}${session?.xet_known && !session.xet ? '-no-xet' : ''}`}><p>{TRANSPORTS.find(option => option.value === transport)?.detail}{session?.xet_known && !session.xet && transport !== 'https' ? ' This hf has no hf_xet, so ' + (transport === 'xet' ? 'downloads will not start.' : 'it uses HTTPS.') : ''} Applies to downloads started after a change.</p></Reveal></div>
                <div className="hf-card-actions"><button className="secondary compact" {...busyButton(isBusy('session'))} onClick={() => action('session', checkSession)}><RefreshCw size={13}/>Check session</button></div>
                <div className="token-details"><button type="button" className="token-summary" aria-expanded={tokenOpen} aria-controls={tokenFormId} onClick={() => setTokenOpen(open => !open)}><ChevronRight className="disclosure" size={14}/>Use a token for this session</button><Reveal open={tokenOpen} id={tokenFormId}><div className="inline-form"><input type="password" autoComplete="off" aria-label="Hugging Face token" placeholder="hf_… (kept in memory only)" value={token} onChange={e => setToken(e.target.value)}/><button className="secondary compact" {...busyButton(isBusy('token'))} onClick={() => action('token', async () => { await invoke('set_hf_token', { token }); setToken(''); await checkSession(); })}>Apply</button></div><button className="ghost" {...busyButton(isBusy('token'))} onClick={() => action('token', async () => { await invoke('set_hf_token', { token: '' }); await checkSession(); })}>Use saved login instead</button></Reveal></div>
              </div>
            </section>
            <section className="hub-detail" id="hub-detail" aria-label="Repository details">
              {!inspection ? <div className="hub-empty">{inspecting ? <><RefreshCw className="spin" size={20}/><p>Reading repository…</p></> : <><Network size={22}/><p>Pick a repository to see its GGUF files.</p></>}</div> : <>
                <div className="detail-head"><AuthorLogo author={inspection.repo.split('/')[0]} size={56}/><div><h2>{inspection.repo.split('/').slice(1).join('/')}</h2><p>{inspection.repo.split('/')[0]}</p></div><CopyButton className="detail-copy" value={inspection.repo} label="Copy repository id" size={15} onError={copyError}/></div>
                <div className="detail-chips"><span className="chip">GGUF</span><span className="chip">{inspection.files.length} {inspection.files.length === 1 ? 'file' : 'files'}</span>{inspection.sha && <span className="chip chip-copy" title={inspection.sha}>Revision <code>{inspection.sha.slice(0, 10)}</code><CopyButton value={inspection.sha} label="Copy full revision" size={12} onError={copyError}/></span>}</div>
                <div className="files" id="hub-files">{inspection.files.length === 0 ? <p className="hint">No GGUF files in this repository. Open a GGUF conversion of this model.</p> : inspection.files.map(file => { const fileDownload = downloadFor(inspection.repo, file.name); const isPreparing = preparing.includes(`${inspection.repo}/${file.name}`); return <div className="file-entry" key={file.name}><div className="file-row"><FileJson size={17}/><div><span className="identity-line"><strong>{file.name}</strong><CopyButton value={file.name} label={`Copy file name ${file.name}`} size={12} onError={copyError}/></span><small>{file.projector ? 'Multimodal projector' : file.mtp_head ? 'MTP head' : 'Model weights'}{file.size ? ` / ${(file.size / 1e9).toFixed(2)} GB` : ''}</small></div>{!file.projector && !file.mtp_head && <FileFit fit={file.fit}/>}<button className="secondary compact" {...busyButton(isPreparing, !!fileDownload?.active)} onClick={() => startDownload(inspection.repo, file.name)}><ArrowDownToLine size={14}/>Download</button></div>{fileDownload ? <DownloadPanel download={fileDownload} busy={isBusy('cancel')} onCancel={() => cancelDownload(fileDownload.id)} onReveal={() => revealFile(fileDownload.path!)} onSelect={() => { if (file.projector) edit('mmproj_path', fileDownload.path!); else if (file.mtp_head) { edit('mtp_path', fileDownload.path!); edit('mtp', true); } else selectModel(fileDownload.path!); setTab('loading'); }}/> : isPreparing ? <div className="download-preparing" role="status"><RefreshCw className="spin" size={14}/>Preparing download…</div> : null}</div>; })}</div>
              </>}
            </section>
          </div>
        </>}
        {tab === 'runtime' && <>
          <RuntimeManager currentPath={settings.runtime_path} currentName={runtimeName(settings.runtime_path)} running={running} progress={snapshot?.runtime_download} onRefresh={refresh} onSelect={runtime => edit('runtime_path', runtime.path)}
            cpuPath={settings.cpu_runtime_path} cpuProgress={snapshot?.cpu_runtime_download} cpuName={settings.cpu_runtime_path ? runtimeName(settings.cpu_runtime_path) : 'No CPU runtime selected'} onSelectCpu={runtime => edit('cpu_runtime_path', runtime.path)}
            cpuChildren={
              <SettingRow label="Path to CPU llama-server" htmlFor="cpu-runtime-path" description="A CPU build's folder or its llama-server." more={'"Use for CPU" above fills this in. Saved with the save bar below.'}>
                <div className="file-input-row"><input id="cpu-runtime-path" placeholder="/path/to/llama.cpp-cpu/build/bin" value={settings.cpu_runtime_path} onChange={e => edit('cpu_runtime_path', e.target.value)}/><button className="secondary" type="button" {...busyButton(isBusy('probe-cpu'), !settings.cpu_runtime_path.trim())} onClick={() => action('probe-cpu', async () => {
                  const path = settings.cpu_runtime_path;
                  const libraryPath = settings.library_path;
                  const output = await invoke<string>('probe_runtime', { path, libraryPath });
                  setCpuProbe({ path, libraryPath, output });
                })}><RefreshCw size={14} className={showBusy('probe-cpu') ? 'spin' : ''}/>Check</button></div>
                <Reveal open={cpuVersion}>{cpuVersion && <pre className="runtime-version">{cpuVersion}</pre>}</Reveal>
              </SettingRow>}>
            <SettingRow label="Path to llama-server" htmlFor="runtime-path" description="A runtime folder or the llama-server executable." more="Choosing a build above fills this in.">
              <div className="file-input-row"><input id="runtime-path" placeholder="/path/to/llama.cpp/build/bin" value={settings.runtime_path} onChange={e => edit('runtime_path', e.target.value)}/><button className="secondary" type="button" {...busyButton(isBusy('probe'))} onClick={() => action('probe', async () => {
                  const path = settings.runtime_path;
                  const libraryPath = settings.library_path;
                  const output = await invoke<string>('probe_runtime', { path, libraryPath });
                  setRuntimeProbe({ path, libraryPath, output });
                })}><RefreshCw size={14} className={showBusy('probe') ? 'spin' : ''}/>Check</button></div>
              <Reveal open={version}>{version && <pre className="runtime-version">{version}</pre>}</Reveal>
            </SettingRow>
            <SettingRow label="Shared-library folders (optional)" htmlFor="runtime-library-path" description="For CUDA or other libraries the runtime cannot find." more="Separate folders with a colon. Only the runtime process sees them.">
              <input id="runtime-library-path" placeholder="Found automatically, or /usr/local/cuda/lib64" value={settings.library_path} onChange={e => edit('library_path', e.target.value)}/>
            </SettingRow>
          </RuntimeManager>
          <section className="panel settings-panel"><div className="panel-heading"><span><Settings2 size={16}/> Connection</span></div><div className="settings-body">
            <SettingSection title="Ports">
              <SettingRow label="Local API port" description="Used after a restart." more="If it is taken, Noullama picks a free nearby port and shows it in Local API."
                control={<NumberField label="Local API port" min={1} max={65535} value={settings.api_port} onChange={value => edit('api_port', value)}/>}/>
              <SettingRow label="Private llama-server port" description="Internal only, on 127.0.0.1." more="Each loaded model takes the next free port from here."
                control={<NumberField label="Private llama-server port" min={1} max={65535} value={settings.runtime_port} onChange={value => edit('runtime_port', value)}/>}/>
            </SettingSection>
            <SettingSection title="Tools">
              <SettingRow label="Hugging Face CLI" htmlFor="hf-cli-path" description="Use the full path if hf is not on the desktop app's PATH."
                control={<input id="hf-cli-path" value={settings.hf_path} onChange={e => edit('hf_path', e.target.value)} placeholder="hf"/>}/>
            </SettingSection>
          </div></section>
          <div className={`save-bar ${runtimeFormDirty ? 'dirty' : ''}`}><Reveal open className="hint" swapKey={String(runtimeFormDirty)}>{runtimeFormDirty ? running ? 'You have unsaved changes. Loading a model does not save them, and loaded models keep the build they started with.' : 'You have unsaved changes. Loading a model does not save them.' : running ? 'All changes saved. Loaded models keep the build they started with.' : 'All changes saved.'}</Reveal><button className={runtimeFormDirty ? 'primary' : 'secondary'} {...busyButton(isBusy('save', 'save-load', 'start'), !runtimeFormDirty, isBusy('save'))} onClick={() => action('save', async () => { await saveRuntimeSettings(); setNotice({ text: 'Settings saved.', error: false }); })}>Save settings</button></div>
        </>}
        {tab === 'library' && <Library library={snapshot?.library ?? null} loadedPaths={libraryLoadedPaths}
          companions={libraryCompanions} desktop={desktop} scrollPosition={libraryScroll} {...libraryActions}/>}
        {tab === 'settings' && <>
          <section className="panel settings-panel"><div className="panel-heading"><span><MemoryStick size={16}/> Memory protection</span></div><div className="settings-body">
            <div className="setting-rows">
              <SettingRow label="Allow force fit" description="Disables OOM load prevention and automatic recovery from sustained critical memory pressure." more="The operating system or runtime may still stop a model if memory runs out."
                control={<Toggle label={shownSetting('allow_force_fit', false) ? 'On' : 'Off'} ariaLabel="Allow force fit" checked={shownSetting('allow_force_fit', false)} disabled={!desktop} onChange={value => void changeSetting('allow_force_fit', value, 'set_memory_preference', { key: 'allow_force_fit', value })}/>}/>
              <Reveal open={snapshot?.settings.hide_crowded_load_warning}>{snapshot?.settings.hide_crowded_load_warning && <SettingRow label="Crowded-load warning" description="Currently hidden."
                control={<button className="secondary compact" {...busyButton(isBusy('memory-pref'))} onClick={() => void action('memory-pref', async () => { await invoke('set_memory_preference', { key: 'hide_crowded_load_warning', value: false }); })}>Show crowded-load warning again</button>}/>}</Reveal>
            </div>
          </div></section>
          <section className="panel settings-panel"><div className="panel-heading"><span><Route size={16}/> Local API routing</span></div><div className="settings-body">
            <div className="setting-rows">
              <SettingRow label="Fall through to another instance" description="Plain names of an unloaded instance 1 go to another loaded instance of the file." more="When instance 1 of a model file is not loaded but another instance is, requests for its plain names (without a suffix) go to the loaded instance with the lowest number. Off, they get 503. Suffixed names never fall through."
                control={<Toggle label={shownSetting('fall_through_to_other_instances', false) ? 'On' : 'Off'} ariaLabel="Fall through to another instance" checked={shownSetting('fall_through_to_other_instances', false)} disabled={!desktop} onChange={on => void changeSetting('fall_through_to_other_instances', on, 'set_fall_through', { on })}/>}/>
              <SettingRow label="When every instance is busy" descriptionKey={whenBusy} description={WHEN_BUSY.find(option => option.value === whenBusy)?.detail(waitDraft ?? shownWait)} more={'Applies to requests that send "any-available" as the model. They go to a ready instance of any model that is not running a request, the one idle longest first.'}
                control={<SegmentedControl label="When every instance is busy">{WHEN_BUSY.map(option => <button key={option.value} type="button" className={whenBusy === option.value ? 'on' : ''} aria-pressed={whenBusy === option.value} disabled={!desktop} onClick={() => { if (whenBusy !== option.value) void changeSetting('any_available_when_busy', option.value, 'set_any_available_when_busy', { mode: option.value }); }}>{option.label}</button>)}</SegmentedControl>}/>
              <Reveal open={whenBusy === 'wait'}>{whenBusy === 'wait' && <SettingRow label="Wait up to (ms)" description="How long a request waits for a busy instance."
                control={<NumberField label="Wait up to (ms)" value={waitDraft ?? shownWait} min={WAIT_MS.min} max={WAIT_MS.max} step={WAIT_MS.step} onChange={value => setWaitDraft(Math.min(Number.MAX_SAFE_INTEGER, Math.max(0, Math.round(value))))}/>}/>}</Reveal>
            </div>
          </div></section>
          <section className="panel settings-panel"><div className="panel-heading"><span><ShieldCheck size={16}/> Model file checks</span></div><div className="settings-body">
            <div className="setting-rows">
              <SettingRow label="sha256 before loading" descriptionKey={shownSetting('sha256_check', 'once')} description={SHA_CHECKS.find(option => option.value === shownSetting('sha256_check', 'once'))?.detail} more="Each model's sha256 is kept in models.yml. It guards against loading a file that was replaced, truncated or swapped by mistake; it is not a security boundary."
                control={<SegmentedControl label="sha256 check">{SHA_CHECKS.map(option => <button key={option.value} type="button" className={shownSetting('sha256_check', 'once') === option.value ? 'on' : ''} aria-pressed={shownSetting('sha256_check', 'once') === option.value} disabled={!desktop} onClick={() => { if (shownSetting('sha256_check', 'once') !== option.value) void changeSetting('sha256_check', option.value, 'set_sha256_check', { mode: option.value }); }}>{option.label}</button>)}</SegmentedControl>}/>
            </div>
          </div></section>
          <section className="panel settings-panel"><div className="panel-heading"><span><FileJson size={16}/> Models file</span></div><div className="settings-body">
            <div className="setting-rows">
              <SettingRow label={<span className="identity-line"><code>{snapshot?.library?.file ?? 'models.yml'}</code>{snapshot?.library?.file && <CopyButton value={snapshot.library.file} label="Copy models.yml path" size={12} onError={copyError}/>}</span>}
                description="Reread whenever it changes; backed up before Noullama edits it." more="Noullama rereads models.yml whenever it changes, and before loading, ejecting, routing a request or changing Keep context. It backs the file up before every change it makes, other than recording a sha256, and keeps the newest 20 copies.">
                <span className="save-actions"><button className="secondary compact" {...busyButton(isBusy('open-yml'), !desktop)} onClick={() => action('open-yml', () => libraryCommand('open_models_file'))}>Edit models.yml</button><button className="secondary compact" {...busyButton(isBusy('open-backups'), !desktop)} onClick={() => action('open-backups', () => libraryCommand('open_backups_folder'))}>Backups</button><button className="ghost" onClick={() => setTab('library')}>Open Library</button></span>
              </SettingRow>
            </div>
          </div></section>
        </>}
        <ApiKeys active={tab === 'settings'} desktop={desktop}
          required={requireApiKey} models={libraryEntries} onCopyError={copyError} onOpenLibrary={() => setTab('library')}
          onToggle={on => void changeSetting('require_api_key', on, 'set_require_api_key', { on })}/>
        {tab === 'loading' && <>
          <Reveal open={loadErrors.length > 0} swapKey={loadErrors.map(error => error.id).join(",")}>{loadErrors.length > 0 && loadErrors.map(error => <div className="load-error" role="alert" key={error.id}><AlertCircle size={18}/><div><strong>{error.name}: {error.message.split('\n')[0]}</strong>{error.message.includes('\n') && <pre>{error.message.split('\n').slice(1).join('\n')}</pre>}</div><div className="load-error-actions"><button className="ghost" onClick={() => setShowLogs(true)}>View logs</button><button className="icon-button" aria-label={`Dismiss error for ${error.name}`} onClick={() => void dismissLoadError(error.id)}><X size={15}/></button></div></div>)}</Reveal>
          <section className="panel settings-panel load-panel" aria-label="Load a model"><div className="panel-heading"><span><Play size={16}/> Load a model</span></div><div className="settings-body">
            <div className="load-summary">
              <div className="load-identity">
                <div className="load-identity-icon">{accountFor(model.selected) ? <AuthorLogo author={accountFor(model.selected)} size={40}/> : <Cpu size={20}/>}</div>
                <div className="model-identity"><small>{selectedInstance ? selectedInstance.ready ? 'Loaded model' : 'Loading model' : 'Selected model'}{model.device && <> · {model.device}</>}</small><span className="identity-line"><strong title={model.selected}>{model.selected ? nameFor(model.selected) ?? basename(model.selected) : 'No model selected'}</strong>{model.selected && <CopyButton value={basename(model.selected)} label="Copy model file name" size={13} onError={copyError}/>}</span></div>
                <div className="load-identity-state">
                  <Reveal open={hubRepo}>{hubRepo && <button type="button" className="secondary compact hub-link" title={`Open ${hubRepo} in the Hub`} aria-label={`Open ${hubRepo} in the Hub`} onClick={() => { setTab('hub'); void openRepo(hubRepo); }}><HuggingFaceLogo size={14}/>Hub</button>}</Reveal>
                  <Reveal open={selectedInstances.length > 1}>{selectedInstances.length > 1 && <div className="load-instance"><Select label="Instance" value={selectedInstance?.id ?? ''} options={selectedInstances.map(i => ({ value: i.id, label: `${apiName(i)} · instance ${i.slot}${i.ready ? '' : ' (loading)'}` }))} onChange={setLoadTarget}/></div>}</Reveal>
                  <span className={`chip ${model.ready ? 'success' : ''}`}><span className={`status-dot ${model.ready ? 'green' : model.loaded ? 'amber' : ''}`}/>{showBusy('start') && !model.verifying ? 'Starting…' : model.label}</span>
                </div>
              </div>
              <div className="load-fit">
                {fit ? <span className={`fit-badge fit-level-${fit.level}`} title="Estimated memory fit, not model/runtime compatibility."><span className="fit-bars" aria-hidden="true">{[1, 2, 3, 4].map(n => <i key={n} className={n <= fit.level ? 'on' : ''}/>)}</span>{fit.label}</span> : <span className="fit-unavailable">{model.selected && panelFit && 'error' in panelFit ? 'Fit unavailable' : 'Memory fit'}</span>}
                <p className={model.selected && panelFit && 'error' in panelFit ? 'fit-error' : undefined}>{fitNote || ' '}</p>
              </div>
            </div>
            <SettingSection title="Model">
              <SettingRow label="Model file (GGUF)" htmlFor="load-model-path"><div className="file-input-row"><input id="load-model-path" placeholder="/path/to/model.gguf" value={settings.model_path} onChange={e => edit('model_path', e.target.value)}/><button className="secondary" type="button" {...busyButton(modelPending || isBusy('browse'), false, isBusy('browse'))} onClick={() => browseFile('model_path')}><FolderOpen size={14}/>Browse file</button><button className="secondary" type="button" {...busyButton(modelPending, false, false)} onClick={() => setTab('models')}><Layers3 size={14}/>Select model</button></div></SettingRow>
              <SettingRow label="Multimodal projector" description="Lets image-capable models read images in the Playground."
                control={<Toggle label={settings.mmproj_path ? 'On' : 'Off'} ariaLabel="Multimodal projector" title={settings.mmproj_path ? `Loads ${basename(settings.mmproj_path)}` : projector ? `Load ${basename(projector)} with this model` : undefined} checked={!!settings.mmproj_path} disabled={!settings.mmproj_path && !projector} onChange={on => edit('mmproj_path', on ? projector : '')}/>}
                statusKey={projector ? 'found' : 'missing'} status={settings.mmproj_path ? undefined : projector ? <StatusNote tone="ok">Found {basename(projector)} next to the selected model.</StatusNote> : <StatusNote tone="info">No projector next to the selected model. <button type="button" className="ghost inline" {...busyButton(modelPending || isBusy('browse'), false, isBusy('browse'))} onClick={() => browseFile('mmproj_path')}>Choose a file</button></StatusNote>}>
                {settings.mmproj_path ? <div className="file-input-row"><input aria-label="Multimodal projector path" placeholder="/path/to/mmproj.gguf" value={settings.mmproj_path} onChange={e => edit('mmproj_path', e.target.value)}/><button className="secondary" type="button" {...busyButton(modelPending || isBusy('browse'), false, isBusy('browse'))} onClick={() => browseFile('mmproj_path')}><FolderOpen size={14}/>Browse</button></div> : null}
              </SettingRow>
              <SettingRow label="Multi-token prediction (MTP)" description="Can speed up generation by drafting several tokens per step." more="Drafts several tokens per step with the model's MTP layers, which can speed up generation. Needs a llama.cpp build with MTP support."
                control={<Toggle label={settings.mtp ? 'On' : 'Off'} ariaLabel="Multi-token prediction" title={settings.mtp || mtpAvailable ? mtpTitle : undefined} checked={settings.mtp} disabled={!settings.mtp && !mtpAvailable} onChange={setMtp}/>}
                statusKey={builtInMtp ? 'built-in' : mtpHead ? 'found' : 'missing'} status={settings.mtp_path ? undefined : builtInMtp ? <StatusNote tone="ok">The selected model has built-in MTP layers.</StatusNote> : mtpHead ? <StatusNote tone="ok">Found {basename(mtpHead)} next to the selected model.</StatusNote> : <StatusNote tone="info">No MTP layers in the selected model and no matching MTP head next to it. <button type="button" className="ghost inline" {...busyButton(modelPending || isBusy('browse'), false, isBusy('browse'))} onClick={() => browseFile('mtp_path')}>Choose a file</button></StatusNote>}>
                {settings.mtp_path ? <div className="file-input-row"><input aria-label="MTP head path" placeholder="/path/to/mtp-model.gguf" value={settings.mtp_path} onChange={e => edit('mtp_path', e.target.value)}/><button className="secondary" type="button" {...busyButton(modelPending || isBusy('browse'), false, isBusy('browse'))} onClick={() => browseFile('mtp_path')}><FolderOpen size={14}/>Browse</button></div> : null}
              </SettingRow>
            </SettingSection>
            <SettingSection title="Hardware">
              <SettingRow label="Device" wide description="Where each instance runs: one device, or split across GPUs."
                control={<Select label="Device" value={device.value} disabled={device.effective === null} options={device.options} onChange={value => edit('device', value === CHOOSE_GPUS ? chosenGpus(device.split ? device.split.gpus : device.effective && device.effective !== CPU_DEVICE ? [device.effective] : [], deviceListed) : value)}/>}
                statusKey={JSON.stringify([!!listError, deviceStatus?.tone])} status={(deviceStatus || listError) && <>{listError && <StatusNote tone="warn">{listError}</StatusNote>}{deviceStatus && <StatusNote tone={deviceStatus.tone}>{deviceStatus.text}</StatusNote>}</>}/>
              <Reveal open={device.value === CHOOSE_GPUS && device.canSplit}>{device.value === CHOOSE_GPUS && device.canSplit && <SettingRow label="GPUs" description="Choose 2 or more.">
                <div className="api-key-models" role="group" aria-label="GPUs to split across">
                  {deviceListed.map(gpu => {
                    const checked = !!device.split?.gpus.includes(gpu.name);
                    return <div className="api-key-model-group" key={gpu.name}><button type="button" role="checkbox" aria-checked={checked} className="api-key-model" onClick={() => edit('device', toggleGpu(settings.device, gpu.name, deviceListed))}>
                      <span className="checkbox-mark" aria-hidden="true"/>
                      <span>{deviceLabel(gpu).label} <small>{deviceLabel(gpu).detail}</small></span>
                    </button></div>;
                  })}
                </div>
              </SettingRow>}</Reveal>
              <Reveal open={splitting}>{splitting && <SettingRow label="Split mode" description="How the model is divided between the GPUs."
                more="Layer gives each GPU whole layers and suits most setups. Row splits each layer across the GPUs, which can help with fast links between them. Passed as --split-mode."
                status={splitModeError && <StatusNote tone="error">Choose Layer or Row.</StatusNote>}
                control={<SegmentedControl label="Split mode">{SPLIT_MODES.map(([value, label]) => <button key={value} type="button" className={settings.split_mode === value ? 'on' : ''} aria-pressed={settings.split_mode === value} onClick={() => edit('split_mode', value)}>{label}</button>)}</SegmentedControl>}/>}</Reveal>
              <Reveal open={splitting}>{splitting && <SettingRow label="Split ratio" htmlFor="load-split-ratio" description="Empty shares by free memory."
                more={`One number per GPU, in the listed order (${device.split!.gpus.join(', ')}). For example 3,1 puts three quarters on the first. Passed as --tensor-split.`}
                control={<input id="load-split-ratio" placeholder="Auto" value={settings.tensor_split} aria-invalid={!!splitRatioError} onChange={e => edit('tensor_split', e.target.value)}/>}
                status={splitRatioError && <StatusNote tone="error">{splitRatioError}</StatusNote>}/>}</Reveal>
              <SettingRow label="GPU layers" description="0 runs on the CPU. 99 puts every layer that fits on the GPU."
                control={<NumberField label="GPU layers" min={-1} value={settings.gpu_layers} onChange={value => edit('gpu_layers', value)}/>}
                status={device.effective === CPU_DEVICE && <StatusNote tone="info">Not used on Device CPU, which always loads with 0.</StatusNote>}/>
              <SettingRow label="Context size" description="Maximum tokens per request. The default is 8192."
                control={<NumberField label="Context size" min={512} max={131072} step={1024} value={settings.context_size} onChange={value => edit('context_size', value)}/>}/>
            </SettingSection>
            <SettingSection title="Behaviour">
              <SettingRow label="Decision mode" descriptionKey={settings.decision_mode} description={(DECISION_MODES.find(([value]) => value === settings.decision_mode) ?? DECISION_MODES[0])[2]}
                control={<SegmentedControl label="Decision mode">{DECISION_MODES.map(([value, label]) => <button key={value} type="button" className={settings.decision_mode === value ? 'on' : ''} aria-pressed={settings.decision_mode === value} onClick={() => edit('decision_mode', value)}>{label}</button>)}</SegmentedControl>}/>
              <SettingRow label="Keep context" description="Whether a newly loaded model starts with Keep context on." more="Loaded models have their own switch on their card below; this does not change them."
                control={<Toggle label={settings.keep_context ? 'On' : 'Off'} ariaLabel="Keep context for new loads" checked={settings.keep_context} onChange={on => edit('keep_context', on)}/>}/>
            </SettingSection>
            <SettingSection title="Runtime">
              <SettingRow label="llama.cpp build" wide description={<>The build and shared-library folders saved on the Runtime tab. <button type="button" className="ghost inline" onClick={() => setTab('runtime')}>Change in Runtime</button></>}
                control={<span className="value-box" title={savedSettings?.runtime_path || undefined}><code>{loadRuntime}</code></span>}
                status={runtimeFormDirty && <StatusNote tone="warn">The Runtime tab has unsaved changes. Loads use the saved build and shared-library folders until you save them there.</StatusNote>}/>
            </SettingSection>
          </div></section>
          <div className={`save-bar load-bar ${loadDirty ? 'dirty' : ''}`}><Reveal open className="hint" swapKey={JSON.stringify([loadDirty, reloadSelected, selectedInstances.length, !!selectedInstance, running])}>{loadDirty ? 'You have unsaved changes. Loading saves them first.' : reloadSelected ? 'The selected model was loaded with other settings. Reload model applies these.' : selectedInstances.length > 1 ? `The selected model is loaded ${selectedInstances.length} times. Eject and Reload act on the instance chosen above.` : selectedInstance ? 'The selected model is loaded with these settings.' : running ? 'Settings apply when you load. Loaded models keep the settings they started with.' : 'Settings apply when you load.'}</Reveal><div className="load-bar-actions">
            <button className="secondary" {...busyButton(modelPending, !loadDirty, isBusy('save-load'))} onClick={() => action('save-load', async () => { await saveLoadSettings(); setNotice({ text: 'Settings saved.', error: false }); })}>Save settings</button>
            {selectedInstance && <button className="secondary" {...busyButton(modelPending, false, false)} title={selectedInstance.ready ? `Unload ${apiName(selectedInstance)}; other loaded models keep running` : undefined} onClick={() => void eject(selectedInstance.id)}>{selectedInstance.ready ? <Unplug size={13}/> : <Square size={13}/>}{selectedInstance.ready ? 'Eject' : 'Cancel loading'}</button>}
            {(!selectedInstance || reloadSelected) && <button className="primary" {...busyButton(modelPending, !model.selected || model.missing || splitBlocked || (!!selectedInstance && !selectedInstance.ready), isBusy('start'))} title={!selectedInstance && running ? 'Loads beside the models already loaded' : undefined} onClick={() => void loadSelected({ replace: replaceOnLoad })}>{showBusy('start') ? <RefreshCw className="spin" size={14}/> : <Play size={14}/>} {showBusy('start') ? 'Starting…' : selectedInstance ? 'Reload model' : running ? 'Load beside' : 'Load model'}</button>}
            {selectedInstance && <button className={reloadSelected ? 'secondary' : 'primary'} {...busyButton(modelPending, !model.selected || model.missing || splitBlocked, isBusy('start'))} title="Loads one more instance of this file beside the loaded one. It answers to the file's names with a suffix such as -i-2." onClick={() => void loadSelected()}>{showBusy('start') ? <RefreshCw className="spin" size={14}/> : <Plus size={14}/>} {showBusy('start') ? 'Starting…' : 'Load another instance'}</button>}
          </div></div>
          <Reveal open={instances.length > 0}>{instances.length > 0 && <><div className="section-heading"><h2>Loaded</h2><span>{instances.length} {instances.length === 1 ? 'model' : 'models'} · each keeps the settings it was loaded with</span></div><div className="loaded-list">{instances.map(instance => <InstanceCard key={instance.id} instance={instance} account={accountFor(instance.path)} runtime={runtimeName(instance.runtime_path || instance.settings.runtime_path)} aliases={aliasesFor(instance.path)} routes={routes} displayName={apiName(instance)}
            stale={differsFrom(instance)} canReload={differsFrom(instance) && instance.ready} starting={showBusy('start')} busy={modelPending} ejecting={isBusy(`eject:${instance.id}`)} reloadDisabled={model.missing || splitBlocked} namesDisabled={!desktop}
            keepContext={keepsContext(instance)} onKeepContext={on => void toggleKeepContext(instance.id, on)} onClearContext={() => void clearContext(instance.id)}
            onEject={() => void eject(instance.id)} onReload={() => void loadSelected({ replace: instance.id })} onSaveAliases={saveAliases} onSaveSuffix={suffix => saveSuffix(instance.id, suffix)} onSaveInstanceAliases={aliases => saveInstanceAliases(instance.id, aliases)} onError={copyError}/>)}</div></>}</Reveal>
        </>}
        {tab === 'api' && <>
          <section className="endpoint-card"><div><span className={`status-dot ${!snapshot?.api_error && desktop ? 'green' : ''}`}/><strong>{apiBase}</strong><button className="icon-button" aria-label="Copy API URL" onClick={() => copy(apiBase, 'api')}>{copied === 'api' ? <Check size={16} weight="bold"/> : <Copy size={16}/>}</button></div><Reveal open swapKey={snapshot?.api_error ? 'error' : String(requireApiKey)}><p>{snapshot?.api_error ?? (requireApiKey ? 'Localhost only. Send a bearer API key with /v1/ and /health requests.' : 'Localhost only. No API key needed. Available while Noullama is open.')}</p></Reveal></section>
          <ModelRouting loadedCount={instances.length} anyReady={instances.some(i => i.ready)} whenBusy={whenBusy} waitMs={shownWait} routes={routes} aliasesFor={aliasesFor} disabled={!desktop} onSave={saveAliases} onError={copyError}/>
          <div className="endpoint-list">{([['GET', '/v1/models', 'List the names that reach a loaded model'], ['POST', '/v1/systemone', 'Native state + typed questions'], ['POST', '/v1/chat/completions', 'OpenAI chat response envelope'], ['GET', '/health', 'Gateway health and model readiness']] as const).map(([method, path, text]) => <div key={path}><span className={`method ${method === 'GET' ? 'get' : ''}`}>{method}</span><code>{path}</code><span>{text}</span><CopyButton value={`${apiOrigin}${path}`} label={`Copy ${apiOrigin}${path}`} size={13} onError={copyError}/></div>)}</div>
          <section className="panel code-panel"><div className="panel-heading"><span>Native decision request</span><button className="icon-button" aria-label="Copy native API example" onClick={() => copy(`curl ${apiBase}/systemone${authHeader} -H 'Content-Type: application/json' -d '${requestJson.replace(/'/g, "'\\''")}'`, 'curl')}>{copied === 'curl' ? <Check size={14} weight="bold"/> : <Copy size={14}/>}</button></div><pre>{`curl ${apiBase}/systemone${authHeader} \\\n  -H 'Content-Type: application/json' \\\n  -d '${requestJson.replace(/'/g, "'\\''")}'`}</pre></section>
          <section className="panel code-panel"><div className="panel-heading"><span>Python · OpenAI SDK</span><div className="heading-actions"><span className="chip">Non-streaming</span><CopyButton value={pythonExample} label="Copy Python example" onError={copyError}/></div></div><pre>{pythonExample}</pre></section>
          <div className="api-note"><BookOpen size={18}/><div><strong>Request format</strong><p>Send one user message containing a JSON request with state and questions. The assistant content contains the native result as JSON. Chat history, tools, and streaming are unsupported. Questions use choice, noul, or score; this runtime does not support extraction or multi-select.</p><a href="https://github.com/ggml-org/llama.cpp/blob/b11938/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api" target="_blank" rel="noreferrer">llama.cpp decision API reference<ExternalLink size={12}/></a></div></div>
        </>}
      </div>
    </main>
    {showDownloadToast && <aside className="download-toast" aria-label="Downloads" ref={toastRef}>
      <div className="toast-head"><strong>Downloads{toastDownloads.length > 1 ? ` (${toastDownloads.length})` : ''}</strong><button className="icon-button" aria-label={toastCollapsed ? 'Show downloads' : 'Collapse downloads'} aria-expanded={!toastCollapsed} onClick={() => setToastCollapsed(v => !v)}>{toastCollapsed ? <ChevronUp size={15}/> : <ChevronDown size={15}/>}</button></div>
      <Reveal open={!toastCollapsed}>{!toastCollapsed && <div className="toast-list">{toastDownloads.map(item => <div className="toast-item" key={item.id}>
        <div className="toast-row"><button className="toast-open" onClick={() => { setTab('hub'); openRepo(item.repo); }}><strong>{item.file}</strong><small>{item.active ? `${item.repo} / ${item.progress === null ? 'Connecting' : `${item.progress.toFixed(0)}%`}` : `${item.repo} / Downloaded`}</small></button>{item.active ? <button className="icon-button" aria-label={`Cancel download of ${item.file}`} title="Cancel download" {...busyButton(isBusy('cancel'))} onClick={() => cancelDownload(item.id)}><X size={15}/></button> : <><CircleCheck className="toast-done" size={17}/><button className="icon-button" aria-label={`Dismiss ${item.file}`} title="Dismiss" onClick={() => void dismissDownload(item.id)}><X size={15}/></button></>}</div>
        {item.active && <div className="download-track" aria-hidden="true"><i style={{ width: `${Math.max(0, Math.min(100, item.progress ?? 0))}%` }}/></div>}
      </div>)}</div>}</Reveal>
    </aside>}
    <MemoryWarningDialog warning={crowdedWarning} pending={isBusy('memory-warning', 'start')} hide={hideCrowded} onHide={setHideCrowded}
      onCancel={() => void action('memory-warning', async () => { if (hideCrowded) await invoke('set_memory_preference', { key: 'hide_crowded_load_warning', value: true }); setCrowdedWarning(null); })}
      onEject={id => { void (async () => { try { if (hideCrowded) await invoke('set_memory_preference', { key: 'hide_crowded_load_warning', value: true }); const replace = crowdedWarning?.replace ?? null; setCrowdedWarning(null); await loadSelected({ replace, ejectOther: id }); } catch (e) { setNotice({ text: errorText(e), error: true }); } })(); }}/>
    {updates && <UpdatesDialog open={updatesOpen} view={{ ...updates, ...preferenceWrites.drafts.prefs }} onClose={() => setUpdatesOpen(false)} onCheck={() => { void updateCommand('check_for_update'); }} onInstall={() => { void updateCommand('install_update'); }} onPrefs={changeUpdatePrefs} prefsSaving={!!preferenceWrites.drafts.prefs}/>}
    <DeletePresetDialog name={presetToDelete?.name ?? null} pending={isBusy('preset')} onCancel={() => setPresetToDelete(null)} onConfirm={() => { if (presetToDelete) void trashPreset(presetToDelete.id); }}/>
    <QuitDialog open={quitPrompt} pending={isBusy('quit') || isBusy('cancel-quit')} loaded={snapshot ? snapshot.instances.length : null} onCancel={() => { setQuitPrompt(false); void action('cancel-quit', async () => { await invoke('cancel_quit'); }); }} onConfirm={() => { void action('quit', async () => { await invoke('quit_app'); }); }}/>
    <Reveal open={showLogs} className="log-reveal">{showLogs && <div className="log-drawer"><div className="panel-heading"><span><Terminal size={16}/> Runtime activity</span><button className="icon-button" aria-label="Close runtime logs" onClick={() => setShowLogs(false)}><X size={17}/></button></div><pre>{snapshot?.logs.join('\n') || 'No runtime activity yet.'}</pre></div>}</Reveal>
  </div>;
}

const keyboardControls = 'button, a[href], summary, input, select, textarea, [tabindex], [contenteditable="true"]';
window.addEventListener('keydown', event => {
  const focused = document.activeElement?.matches(keyboardControls) ?? false;
  if (enablesKeyboardFocus(event, focused)) document.documentElement.classList.add('keyboard-nav');
}, { capture: true });
window.addEventListener('pointerdown', () => {
  document.documentElement.classList.remove('keyboard-nav');
}, { capture: true });

createRoot(document.getElementById('root')!).render(<IconContext.Provider value={{ weight: 'duotone', 'aria-hidden': true }}><App/></IconContext.Provider>);
