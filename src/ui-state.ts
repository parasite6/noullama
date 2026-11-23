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

// The load panel's "Device" select for the saved `device` value. `devices`
// is the GPU runtime's list, or null until it has been read. `effective` is
// the device a load would use (an empty value means the first GPU, else the
// CPU), or null while the list is unknown. A saved GPU the runtime does not
// report stays selected and is marked "(not found)"; the load refuses it.
export function deviceChoice(saved: string, devices: readonly RuntimeDevice[] | null) {
  const listed = devices ?? [];
  const effective = saved || (devices ? listed[0]?.name ?? CPU_DEVICE : null);
  const known = !saved || saved === CPU_DEVICE || listed.some(d => d.name === saved);
  const missing = !!devices && !known;
  const options: { value: string; label: string; detail?: string }[] = [...listed.map(d => ({ value: d.name, ...deviceLabel(d) })), { value: CPU_DEVICE, label: 'CPU', detail: 'CPU runtime' }];
  if (!known) options.unshift({ value: saved, label: devices ? `${saved} (not found)` : saved });
  if (effective === null) options.unshift({ value: '', label: 'Checking devices…' });
  return { value: effective ?? '', effective, missing, options };
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
