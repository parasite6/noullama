// `loaded`: the loaded model for the selected file, if it is loaded. Other
// models can be loaded beside it.
// `missing`: the selected file is no longer on disk (deleted, or its drive is
// unmounted). The selection is kept so it returns when the drive does.
export function modelStatus(selected: string, loaded: { path: string; device: string; ready: boolean } | null, missing = false) {
  return {
    selected,
    loaded: loaded?.path ?? '',
    device: loaded?.device ?? '',
    ready: !!loaded?.ready,
    missing: !loaded && !!selected && missing,
    label: loaded ? loaded.ready ? 'Loaded' : 'Loading' : selected ? missing ? 'File not found' : 'Selected · not loaded' : 'No model selected',
  };
}
