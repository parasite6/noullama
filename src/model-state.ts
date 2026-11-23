// `loaded`: the loaded model for the selected file, if it is loaded. Other
// models can be loaded beside it.
// `missing`: the selected file is no longer on disk (deleted, or its drive is
// unmounted). The selection is kept so it returns when the drive does.
export function modelStatus(selected: string, loaded: { path: string; device: string; ready: boolean; verifying?: Verifying | null } | null, missing = false) {
  const checking = loaded && !loaded.ready ? verifyingText(loaded.verifying) : null;
  return {
    selected,
    loaded: loaded?.path ?? '',
    device: loaded?.device ?? '',
    ready: !!loaded?.ready,
    missing: !loaded && !!selected && missing,
    verifying: !!checking,
    label: loaded ? loaded.ready ? 'Loaded' : checking ?? 'Loading' : selected ? missing ? 'File not found' : 'Selected · not loaded' : 'No model selected',
  };
}

// The file a loading instance is hashing before launch, from the snapshot.
export type Verifying = { file: string; slot: 'model' | 'mtp' | 'mmproj'; done_bytes: number; total_bytes: number };

// "Checking sha256… 42%", naming the file when it is not the model; null
// when nothing is being checked.
export function verifyingText(verifying: Verifying | null | undefined): string | null {
  if (!verifying) return null;
  const of = verifying.slot === 'mtp' ? ' of head' : verifying.slot === 'mmproj' ? ' of projector' : '';
  const percent = verifying.total_bytes > 0 ? ` ${Math.min(100, Math.floor(verifying.done_bytes / verifying.total_bytes * 100))}%` : '';
  return `Checking sha256${of}…${percent}`;
}
