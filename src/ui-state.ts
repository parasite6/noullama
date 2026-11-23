export type Notice = { text: string; error: boolean };
export function visibleBanner(notice: Notice | null, apiError: string | null | undefined, dismissedApi: string | null, desktop: boolean, previewDismissed: boolean): Notice | null {
  if (notice) return notice;
  if (apiError && apiError !== dismissedApi) return { text: `Local API unavailable: ${apiError}. Change its port in Runtime and restart the app.`, error: true };
  if (!desktop && !previewDismissed) return { text: 'Interface preview. Start npm run desktop to connect to the Rust backend.', error: false };
  return null;
}

// A wait as Settings and the any-available 503 message state it: whole
// seconds as seconds, anything else in milliseconds (wait_text in runtime.rs).
export function waitText(ms: number): string {
  if (ms === 1000) return '1 second';
  return ms % 1000 === 0 ? `${ms / 1000} seconds` : `${ms} ms`;
}

// Just what the load decisions below read of a loaded instance (`Instance`
// in main.tsx), so they can be tested without React.
export type LoadedInstance = { id: string; slot: number; path: string };

// The slot a load passes to `start_runtime`: a reload (`replace`, the id of
// the instance it replaces) keeps that instance's slot, so its suffixed names
// stay. A fresh load, "Load another instance", or a reload of an instance that
// stopped on its own meanwhile passes null and takes the lowest free slot.
export function reloadSlot(instances: readonly LoadedInstance[], replace: string | null): number | null {
  if (!replace) return null;
  return instances.find(i => i.id === replace)?.slot ?? null;
}

// The selected file's loaded instances, lowest slot first, and the one the
// load bar acts on: the "Instance" picker's choice (`target`, also set to the
// newest load) while it is one of them, else the lowest slot.
export function instancesOf<T extends LoadedInstance>(instances: readonly T[], modelPath: string, target: string | null): { list: T[]; selected: T | null } {
  const list = instances.filter(i => i.path === modelPath).sort((a, b) => a.slot - b.slot);
  return { list, selected: list.find(i => i.id === target) ?? list[0] ?? null };
}

// The Playground's model after a load that started `loaded`: it stays on the
// model chosen there (`current`) while that one is still loaded and is not the
// instance the load replaced or ejected; otherwise it moves to the new one.
export function playModelAfterLoad(current: string | null, loaded: string, instances: readonly LoadedInstance[], replace: string | null, ejected?: string): string {
  return current && current !== replace && current !== ejected && instances.some(i => i.id === current) ? current : loaded;
}

// One device of the GPU runtime's `--list-devices` (RuntimeDevice in devices.rs).
export type RuntimeDevice = { name: string; description: string; total_mib: number | null; free_mib: number | null };
// The `device` value that loads on the CPU runtime.
export const CPU_DEVICE = 'CPU';

// The device's own name as the option's label ("NVIDIA GeForce
// RTX 3080"), with its llama.cpp name and memory as the detail ("CUDA0 · 10 GB").
export function deviceLabel(device: RuntimeDevice): { label: string; detail: string } {
  const size = device.total_mib ? `${Math.round(device.total_mib / 1024)} GB` : '';
  return device.description
    ? { label: device.description, detail: [device.name, size].filter(Boolean).join(' · ') }
    : { label: device.name, detail: size };
}

// Saved `device` values that split a model across GPUs (devices.rs): every
// GPU listed at load time, or a comma list such as "CUDA0,CUDA1". A chosen
// set with fewer than 2 GPUs keeps a comma ("CUDA0,") so it stays a set.
export const ALL_GPUS = 'gpus:all';
// The Device select's value for a chosen set; never saved.
export const CHOOSE_GPUS = 'gpus:choose';
export const SPLIT_MODES = [['layer', 'Layer'], ['row', 'Row']] as const;
export const isSplit = (device: string) => device.trim() === ALL_GPUS || device.includes(',');
export const splitNames = (device: string) => device.split(',').map(name => name.trim()).filter(Boolean);

// A chosen set as saved: the checked names in listed order.
export function chosenGpus(names: readonly string[], devices: readonly RuntimeDevice[]): string {
  const ordered = devices.map(d => d.name).filter(name => names.includes(name));
  return ordered.length < 2 ? `${ordered.join(',')},` : ordered.join(',');
}

// A chosen set with `name` checked or unchecked.
export function toggleGpu(saved: string, name: string, devices: readonly RuntimeDevice[]): string {
  const names = splitNames(saved);
  return chosenGpus(names.includes(name) ? names.filter(n => n !== name) : [...names, name], devices);
}

// How a split is shown, as the instance card shows it: "CUDA0 + CUDA1" or
// "All GPUs (2)", with the GPUs' summed memory when each reports it.
export function splitLabel(gpus: readonly string[], all: boolean, devices: readonly RuntimeDevice[]): string {
  const label = all ? `All GPUs (${gpus.length})` : gpus.join(' + ');
  const totals = gpus.map(name => devices.find(d => d.name === name)?.total_mib ?? null);
  return totals.every(mib => mib !== null) && totals.length ? `${label} · ${Math.round(totals.reduce<number>((a, b) => a + (b ?? 0), 0) / 1024)} GB` : label;
}

// What is wrong with a split ratio for `count` GPUs, as the backend says it;
// null when it is empty (shared by free memory) or valid.
export function ratioError(text: string, count: number): string | null {
  if (!text.trim()) return null;
  const parts = text.split(',').map(part => part.trim());
  const weights = parts.map(Number);
  const maxFloat = Math.fround(3.4028234663852886e38);
  if (!parts.every(part => /^\+?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(part))
      || weights.some(weight => weight !== 0 && weight < 2 ** -126)
      || weights.reduce((sum, weight) => sum + weight, 0) > maxFloat) return 'Split ratio takes numbers of 0 or more separated by commas, such as 3,1.';
  if (parts.length !== count) return `Split ratio needs one number per GPU: ${count} GPUs are in use, but it has ${parts.length}.`;
  if (weights.every(weight => weight === 0)) return 'Split ratio needs at least one number above 0.';
  return null;
}

// The load panel's "Device" select for the saved `device` value. `devices`
// is the GPU runtime's list, or null until it has been read. `effective` is
// the device a load would use (an empty value means the first GPU, else the
// CPU; a split is its GPUs as a comma list in listed order), or null while
// the list is unknown. A saved GPU the runtime does not report stays
// selected and is marked "(not found)"; the load refuses it. With 2 or more
// GPUs listed there are two ways to split: All GPUs and Choose GPUs….
// `split` is the split in use, if any, and `problem` why it cannot load.
export function deviceChoice(saved: string, devices: readonly RuntimeDevice[] | null) {
  const listed = devices ?? [];
  const names = listed.map(d => d.name);
  const canSplit = listed.length >= 2;
  const options: { value: string; label: string; detail?: string }[] = [...listed.map(d => ({ value: d.name, ...deviceLabel(d) }))];
  if (canSplit) options.push({ value: ALL_GPUS, label: 'All GPUs', detail: splitLabel(names, true, listed) }, { value: CHOOSE_GPUS, label: 'Choose GPUs…', detail: '2 or more' });
  options.push({ value: CPU_DEVICE, label: 'CPU', detail: 'CPU runtime' });
  if (isSplit(saved)) {
    const all = saved.trim() === ALL_GPUS;
    const wanted = all ? names : splitNames(saved);
    const unlisted = wanted.filter(name => !names.includes(name));
    const gpus = names.filter(name => wanted.includes(name));
    const value = all ? ALL_GPUS : CHOOSE_GPUS;
    const problem = !devices ? null
      : all && !canSplit ? `All GPUs needs at least 2 GPUs, but the llama.cpp build in Runtime reports ${listed.length}. Choose another device.`
      : unlisted.length ? `${unlisted.join(', ')} ${unlisted.length === 1 ? 'was' : 'were'} not found: the llama.cpp build in Runtime does not report ${unlisted.length === 1 ? 'it' : 'them'}. Loads are refused until you choose other GPUs.`
      : gpus.length < 2 ? 'Choose at least 2 GPUs to split a model.' : null;
    if (!canSplit) options.unshift({ value, label: all ? 'All GPUs (not available)' : `${splitNames(saved).join(' + ') || 'Chosen GPUs'} (not found)` });
    if (!devices) options.unshift({ value: '', label: 'Checking devices…' });
    const effective = devices ? gpus.join(',') : null;
    return { value: devices ? value : '', effective, missing: unlisted.length > 0 && !!devices, options, split: { all, gpus, label: splitLabel(gpus, all, listed) }, problem, canSplit };
  }
  const effective = saved || (devices ? listed[0]?.name ?? CPU_DEVICE : null);
  const known = !saved || saved === CPU_DEVICE || listed.some(d => d.name === saved);
  const missing = !!devices && !known;
  if (!known) options.unshift({ value: saved, label: devices ? `${saved} (not found)` : saved });
  if (effective === null) options.unshift({ value: '', label: 'Checking devices…' });
  return { value: effective ?? '', effective, missing, options, split: null, problem: null, canSplit };
}

// The instance the load panel's main load button replaces: the "Instance"
// picker's choice when it needs a reload (the button reads "Reload model").
// Otherwise the panel's load is a fresh one (Load model, Load beside, Load
// another instance) and replaces none. The fit badge leaves out this one.
export function panelReplace(selected: LoadedInstance | null, reloadSelected: boolean): string | null {
  return selected && reloadSelected ? selected.id : null;
}

// Runs `run` once its arguments have been the same for `delay` ms, one call
// at a time: arguments that arrive while a call is running wait for it, and
// only the newest are sent. A result is passed to `apply` only when no newer
// arguments came in meanwhile, so the caller keeps showing the last result
// until the current one arrives.
export function latestOnly<A, R>(delay: number, run: (args: A) => Promise<R>, apply: (result: { ok: true; value: R } | { ok: false; error: unknown }) => void) {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let running = false;
  let waiting: { args: A } | null = null;
  let generation = 0;
  const start = async () => {
    if (running || !waiting) return;
    const { args } = waiting;
    waiting = null;
    running = true;
    const mine = generation;
    let result: { ok: true; value: R } | { ok: false; error: unknown };
    try { result = { ok: true, value: await run(args) }; } catch (error) { result = { ok: false, error }; }
    running = false;
    if (mine === generation && !waiting && !timer) apply(result);
    void start();
  };
  return {
    request(args: A) {
      generation += 1;
      waiting = null;
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => { timer = null; waiting = { args }; void start(); }, delay);
    },
    cancel() {
      generation += 1;
      if (timer) clearTimeout(timer);
      timer = null;
      waiting = null;
    },
  };
}

// The quit prompt's text for `loaded` loaded instances (null before the
// first snapshot).
export function quitText(loaded: number | null): string {
  const models = loaded === null ? 'The local API and any loaded models will stop.'
    : loaded === 0 ? 'The local API will stop.'
    : loaded === 1 ? 'The local API and the loaded model will stop.'
    : `The local API and all ${loaded} loaded models will stop.`;
  return `${models} Active downloads will also stop.`;
}
export const apiKeyExpiryOptions = [
  { value: 'never', label: 'Never' },
  { value: '1', label: '1 day' },
  { value: '7', label: '7 days' },
  { value: '30', label: '30 days' },
  { value: '90', label: '90 days' },
  { value: '365', label: '1 year' },
  { value: 'custom', label: 'Custom date' },
] as const;

export function apiKeyExpiry(preset: string, date: string, now = Date.now()): number | null {
  if (preset === 'never') return null;
  if (preset !== 'custom') {
    if (!apiKeyExpiryOptions.some(option => option.value === preset)) throw new Error('Choose an expiry.');
    return Math.floor(now / 1000) + Number(preset) * 86400;
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error('Choose an expiry date.');
  const [year, month, day] = date.split('-').map(Number);
  const end = new Date(year, month - 1, day);
  if (end.getFullYear() !== year || end.getMonth() !== month - 1 || end.getDate() !== day) {
    throw new Error('Choose an expiry date.');
  }
  end.setDate(day + 1);
  if (end.getTime() <= now) throw new Error('Choose today or a future date.');
  return end.getTime() / 1000;
}

export function apiKeyAccessSummary(models: { name: string; aliases: string[] }[] | null): string {
  if (models === null) return 'Any';
  return models.map(model => model.aliases.length ? `${model.name} (aliases at creation: ${model.aliases.join(', ')})` : model.name).join(' · ');
}

export function apiKeyModelOptions(models: { path: string; name: string; aliases: string[] }[]) {
  return models.flatMap(model => [model.name, ...model.aliases].map((name, index) => ({
    id: JSON.stringify([model.path, model.name, name]),
    label: index === 0 ? name : `${name} → ${model.name}`,
    selection: { path: model.path, name },
  })));
}

const navigationKeys = new Set(['Tab', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown']);

export function enablesKeyboardFocus(event: Pick<KeyboardEvent, 'key' | 'altKey' | 'ctrlKey' | 'metaKey'>, onControl: boolean): boolean {
  if (event.altKey || event.ctrlKey || event.metaKey) return false;
  return navigationKeys.has(event.key) || (onControl && (event.key === 'Enter' || event.key === ' '));
}


export function matchingProbe(result: { path: string; libraryPath: string; output: string } | null, path: string, libraryPath: string) {
  return result?.path === path && result.libraryPath === libraryPath ? result.output : null;
}


export function settingsGroupMatches<T>(draft: T, saved: T, fields: readonly (keyof T)[], saving: boolean) {
  return !saving && fields.every(key => draft[key] === saved[key]);
}

export function sanitizeTabOrder<T extends string>(stored: unknown, defaults: readonly T[]): T[] {
  const known = new Set<unknown>(defaults);
  const saved = Array.isArray(stored) ? stored.filter((id): id is T => known.has(id)) : [];
  return [...new Set([...saved, ...defaults])];
}

export function moveTab<T>(order: readonly T[], from: number, to: number): readonly T[] {
  if (from === to || from < 0 || to < 0 || from >= order.length || to >= order.length) return order;
  const moved = [...order];
  const [tab] = moved.splice(from, 1);
  moved.splice(to, 0, tab);
  return moved;
}
