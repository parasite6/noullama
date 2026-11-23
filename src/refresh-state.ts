import { useEffect, useState } from 'react';

// Ownership rejects duplicate clicks before React commits pending state.
export class PendingActions {
  private names = new Set<string>();
  begin(name: string): boolean {
    if (this.names.has(name)) return false;
    this.names.add(name);
    return true;
  }
  end(name: string) { this.names.delete(name); }
  has(name: string) { return this.names.has(name); }
  snapshot() { return new Set(this.names); }
}

export class LatestRequest {
  private generation = 0;
  begin() { return ++this.generation; }
  isLatest(generation: number) { return generation === this.generation; }
}

// Snapshots are JSON values; unchanged polls need not render again.
export function retainEqual<T>(current: T, next: T): T {
  return JSON.stringify(current) === JSON.stringify(next) ? current : next;
}

// False while the window is hidden to the tray, so polls that only feed the
// screen can stop.
export function usePageVisible(): boolean {
  const [visible, setVisible] = useState(() => typeof document === 'undefined' || !document.hidden);
  useEffect(() => {
    const update = () => setVisible(!document.hidden);
    document.addEventListener('visibilitychange', update);
    return () => document.removeEventListener('visibilitychange', update);
  }, []);
  return visible;
}

// IPC values are JSON-compatible. Retain equal sections without serializing them.
export function equalJson(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const left = a as Record<string, unknown>;
  const right = b as Record<string, unknown>;
  const keys = Object.keys(left);
  return keys.length === Object.keys(right).length &&
    keys.every(key => Object.hasOwn(right, key) && equalJson(left[key], right[key]));
}

export function retainSnapshotSections<T extends object>(previous: T | null, next: T): T {
  if (!previous) return next;
  for (const key of Object.keys(next) as (keyof T)[]) {
    if (equalJson(previous[key], next[key])) next[key] = previous[key];
  }
  return next;
}

export function shouldScanModels(
  previous: { models: boolean; folder: string | undefined; finished: readonly string[] } | null,
  current: { models: boolean; folder: string | undefined; finished: readonly string[] },
): boolean {
  return (current.models && !previous?.models) || (!!previous && (
    (previous.folder !== undefined && current.folder !== undefined && previous.folder !== current.folder) ||
    current.finished.some(path => !previous.finished.includes(path))
  ));
}
