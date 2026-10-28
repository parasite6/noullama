import { Download, FolderOpen, ArrowRight, X } from 'lucide-react';

export type DownloadState = { active: boolean; cancelled: boolean; repo: string; file: string; message: string; path: string | null; error: string | null; progress: number | null; downloaded_bytes: number; transferred_bytes: number; total_bytes: number | null };
const bytes = (value: number) => value >= 1e9 ? `${(value / 1e9).toFixed(2)} GB` : `${(value / 1e6).toFixed(1)} MB`;

export function DownloadPanel({ download, onSelect, onReveal, onCancel, busy }: { download: DownloadState; onSelect: () => void; onCancel: () => void; onReveal: () => void; busy: boolean }) {
  const percent = download.progress === null ? null : Math.max(0, Math.min(100, download.progress));
  const done = !download.active && !download.error && !!download.path;
  const received = done ? 100 : download.total_bytes ? Math.max(0, Math.min(100, download.transferred_bytes / download.total_bytes * 100)) : 0;
  const written = done ? 100 : percent ?? 0;
  return <section className={`download-panel ${download.error ? 'download-error' : ''}`} aria-label={`Download ${download.file}`}>
    <div><Download size={16}/><strong>{download.active ? 'Downloading' : download.error ? 'Download failed' : download.cancelled ? 'Download cancelled' : 'Downloaded'}</strong><span className="pill">{download.active ? percent === null ? 'Connecting' : `${percent.toFixed(0)}%` : download.error ? 'Failed' : download.cancelled ? 'Cancelled' : 'Complete'}</span></div>
    <p>{download.error ?? download.message}</p>
    {!download.error && <div className="download-track dual" role="progressbar" aria-label="File download progress" title="Received from the network (left) and written to the file (right)" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent ?? undefined}><i className="track-received" style={{ width: `${received / 2}%` }}/><i className="track-written" style={{ width: `${written / 2}%` }}/></div>}
    {download.active && <div className="download-actions"><button className="secondary compact" onClick={onCancel}><X size={14}/>Cancel download</button></div>}
    <div className="download-counters"><span>{bytes(download.downloaded_bytes)}{download.total_bytes !== null ? ` / ${bytes(download.total_bytes)}` : ''} written</span>{download.transferred_bytes > 0 && <span>{bytes(download.transferred_bytes)} received</span>}</div>
    {download.path && <div className="download-done"><code>{download.path}</code><button className="secondary compact" disabled={busy} onClick={onReveal}><FolderOpen size={14}/>Show in folder</button><button className="secondary compact" onClick={onSelect}>{download.file.toLowerCase().includes('mmproj') ? 'Select projector' : 'Select model'}<ArrowRight size={14}/></button></div>}
  </section>;
}
