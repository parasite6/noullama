// `missing`: the selected file is no longer on disk (deleted, or its drive is
// unmounted). The selection is kept so it returns when the drive does.
export function modelStatus(selected: string, loaded: { path: string; device: string } | null, running: boolean, ready: boolean, missing = false) {
  return {
    selected,
    loaded: running ? loaded?.path ?? '' : '',
    device: running ? loaded?.device ?? '' : '',
    switching: running && !!loaded && selected !== loaded.path,
    missing: !running && !!selected && missing,
    label: running ? ready ? 'Loaded' : 'Loading' : selected ? missing ? 'File not found' : 'Selected · not loaded' : 'No model selected',
  };
}
