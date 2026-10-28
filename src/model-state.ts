export function modelStatus(selected: string, loaded: { path: string; device: string } | null, running: boolean, ready: boolean) {
  return {
    selected,
    loaded: running ? loaded?.path ?? '' : '',
    device: running ? loaded?.device ?? '' : '',
    switching: running && !!loaded && selected !== loaded.path,
    label: running ? ready ? 'Loaded' : 'Loading' : selected ? 'Selected · not loaded' : 'No model selected',
  };
}
