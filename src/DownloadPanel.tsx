import { Download, FolderOpen, ArrowRight } from 'lucide-react';

export type DownloadState = { active: boolean; repo: string; file: string; message: string; path: string | null; error: string | null; progress: number | null; downloaded_bytes: number; transferred_bytes: number; total_bytes: number | null };
const bytes = (value: number) => value >= 1e9 ? `${(value / 1e9).toFixed(2)} GB` : `${(value / 1e6).toFixed(1)} MB`;

export function DownloadPanel({ download, onSelect, onReveal, busy }: { download: DownloadState; onSelect: () => void; onReveal: () => void; busy: boolean }) {
  const percent = download.progress === null ? null : Math.max(0, Math.min(100, download.progress));
  return <section className={`download-panel ${download.error ? 'download-error' : ''}`} aria-label={`Download ${download.file}`}>
    <div><Download size={16}/><strong>{download.active ? 'Downloading' : download.error ? 'Download failed' : 'Downloaded'}</strong><span className="pill">{download.active ? percent === null ? 'Connecting' : `${percent.toFixed(0)}%` : download.error ? 'Failed' : 'Complete'}</span></div>
    <p>{download.error ?? download.message}</p>
    {!download.error && <div className="download-track" role="progressbar" aria-label="File download progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent ?? undefined}><i style={{ width: `${percent ?? 0}%` }}/></div>}
    <div className="download-counters"><span>{bytes(download.downloaded_bytes)}{download.total_bytes !== null ? ` / ${bytes(download.total_bytes)}` : ''} written</span>{download.transferred_bytes > 0 && <span>{bytes(download.transferred_bytes)} received</span>}</div>
    {download.path && <div className="download-done"><code>{download.path}</code><button className="secondary compact" disabled={busy} onClick={onReveal}><FolderOpen size={14}/>Show in folder</button><button className="secondary compact" onClick={onSelect}>{download.file.toLowerCase().includes('mmproj') ? 'Select projector' : 'Select model'}<ArrowRight size={14}/></button></div>}
  </section>;
}
