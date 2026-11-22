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
