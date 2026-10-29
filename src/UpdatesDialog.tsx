import React, { useEffect, useRef } from 'react';
import { AlertCircle, CircleCheck, ExternalLink, RefreshCw } from 'lucide-react';
import { Toggle } from './Toggle';

export type UpdateStatus =
  | { kind: 'idle' } | { kind: 'checking' } | { kind: 'up_to_date' } | { kind: 'available' }
  | { kind: 'downloading'; received: number; total: number | null } | { kind: 'installing' }
  | { kind: 'error'; message: string };
export type UpdateOffer = { version: string; notes: string; page: string; prerelease: boolean };
export type UpdateView = {
  current_version: string; include_prereleases: boolean; check_on_launch: boolean; unsupported: string | null;
  status: UpdateStatus; offer: UpdateOffer | null; checked_at: number | null;
};

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`;
function checkedAgo(seconds: number) {
  const minutes = Math.round((Date.now() / 1000 - seconds) / 60);
  return minutes < 1 ? 'just now' : minutes < 60 ? `${minutes} min ago` : `${Math.round(minutes / 60)} h ago`;
}

export function UpdatesDialog({ open, view, onClose, onCheck, onInstall, onPrefs }: {
  open: boolean; view: UpdateView; onClose: () => void; onCheck: () => void; onInstall: () => void;
  onPrefs: (includePrereleases: boolean, checkOnLaunch: boolean) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const { status, offer } = view;
  const working = status.kind === 'checking' || status.kind === 'downloading' || status.kind === 'installing';
  const installing = status.kind === 'downloading' || status.kind === 'installing';
  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (open && !element.open) element.showModal();
    else if (!open && element.open) element.close();
  }, [open]);

  let line: React.ReactNode;
  if (status.kind === 'checking') line = <><RefreshCw className="spin" size={14}/>Checking for updates…</>;
  else if (status.kind === 'downloading') line = <><RefreshCw className="spin" size={14}/>Downloading {status.total ? `${Math.floor(status.received / status.total * 100)}% of ${mb(status.total)}` : mb(status.received)}…</>;
  else if (status.kind === 'installing') line = <><RefreshCw className="spin" size={14}/>Installing. Approve the system password prompt to continue.</>;
  else if (status.kind === 'error') line = <span className="update-error"><AlertCircle size={14}/>{status.message}</span>;
  else if (offer) line = <>Version {offer.version} is available.</>;
  else if (status.kind === 'up_to_date') line = <><CircleCheck size={14}/>You have the latest version{view.checked_at ? `, checked ${checkedAgo(view.checked_at)}` : ''}.</>;
  else line = <>Not checked yet.</>;

  return <dialog ref={dialog} className="updates-dialog" aria-labelledby="updates-title" onCancel={event => { event.preventDefault(); if (!installing) onClose(); }} onClick={event => {
    if (event.target !== dialog.current || installing) return;
    const bounds = dialog.current.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) onClose();
  }}>
    <h2 id="updates-title">Updates</h2>
    <p className="updates-installed">noullama {view.current_version} is installed.</p>
    <p className="updates-status" role="status">{line}</p>
    {offer && <div className="update-offer">
      <div className="update-offer-head"><strong>noullama {offer.version}</strong>{offer.prerelease && <span className="chip">Pre-release</span>}{offer.page && <a href={offer.page} target="_blank" rel="noreferrer">Release page<ExternalLink size={12}/></a>}</div>
      {offer.notes.trim() && <pre className="update-notes">{offer.notes.trim()}</pre>}
    </div>}
    {view.unsupported && <p className="update-unsupported">{view.unsupported}</p>}
    <div className="updates-prefs">
      <Toggle checked={view.include_prereleases} disabled={working} onChange={value => onPrefs(value, view.check_on_launch)} label="Include pre-releases"/>
      <Toggle checked={view.check_on_launch} disabled={working} onChange={value => onPrefs(view.include_prereleases, value)} label="Check when noullama starts"/>
    </div>
    {offer && !view.unsupported && <p className="updates-hint">Installing unloads the model, stops the local API, and restarts noullama.</p>}
    <div className="quit-dialog-actions">
      <button className="secondary" disabled={installing} onClick={onClose}>Close</button>
      <button className="secondary" disabled={working} onClick={onCheck}>Check now</button>
      {offer && !view.unsupported && <button className="primary" disabled={working} onClick={onInstall}>{installing ? 'Installing…' : 'Install and restart'}</button>}
    </div>
  </dialog>;
}
