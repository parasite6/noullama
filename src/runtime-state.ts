function normalize(path: string) {
  return path.replace(/\\/g, '/').replace(/\/+$/, '').replace(/\/llama-server(?:\.exe)?$/i, '');
}

export function runtimeStatus(selected: string, active: string | null, running: boolean, dataDir: string) {
  const path = running && active ? active : selected;
  const root = normalize(dataDir) + '/runtimes/';
  const normalized = normalize(path);
  const downloaded = !!dataDir && normalized.startsWith(root);
  const name = downloaded ? normalized.slice(root.length).split('/')[0] : normalized.split('/').pop() || 'No runtime selected';
  return {
    path,
    name,
    source: downloaded ? 'Downloaded' : 'Custom',
    inUse: running && !!active,
    pending: running && !!active && normalize(selected) !== normalize(active),
  };
}
