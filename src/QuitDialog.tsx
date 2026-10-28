import React, { useEffect, useRef } from 'react';
import { Power } from 'lucide-react';

export function QuitDialog({ open, pending, onCancel, onConfirm }: { open: boolean; pending: boolean; onCancel: () => void; onConfirm: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const cancel = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (open && !element.open) { element.showModal(); cancel.current?.focus(); }
    else if (!open && element.open) element.close();
  }, [open]);
  return <dialog ref={dialog} className="quit-dialog" aria-labelledby="quit-title" aria-describedby="quit-description" onCancel={event => { event.preventDefault(); if (!pending) onCancel(); }} onClick={event => {
    if (event.target !== dialog.current || pending) return;
    const bounds = dialog.current.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) onCancel();
  }}>
    <Power size={24} aria-hidden="true"/>
    <h2 id="quit-title">Quit Noullama?</h2>
    <p id="quit-description">The local API and loaded model will stop. Active downloads will also stop.</p>
    <div className="quit-dialog-actions"><button ref={cancel} className="secondary" disabled={pending} onClick={onCancel}>Keep running</button><button className="primary" disabled={pending} onClick={onConfirm}>{pending ? 'Quitting…' : 'Quit Noullama'}</button></div>
  </dialog>;
}
