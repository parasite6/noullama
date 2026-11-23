import React, { useEffect, useRef } from 'react';
import { TrashIcon as Trash2 } from '@phosphor-icons/react/dist/icons/Trash';

// Asks before deleting a saved custom preset. Cancel has focus, and Escape or a
// click on the backdrop cancels.
export function DeletePresetDialog({ name, pending, onCancel, onConfirm }: { name: string | null; pending: boolean; onCancel: () => void; onConfirm: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const cancel = useRef<HTMLButtonElement>(null);
  const open = name !== null;
  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (open && !element.open) { element.showModal(); cancel.current?.focus(); }
    else if (!open && element.open) element.close();
  }, [open]);
  return <dialog ref={dialog} className="quit-dialog" aria-labelledby="delete-preset-title" aria-describedby="delete-preset-description" onCancel={event => { event.preventDefault(); if (!pending) onCancel(); }} onClick={event => {
    if (event.target !== dialog.current || pending) return;
    const bounds = dialog.current.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) onCancel();
  }}>
    <Trash2 size={24} aria-hidden="true"/>
    <h2 id="delete-preset-title">Delete “{name}”?</h2>
    <p id="delete-preset-description">This removes the saved questions and state for this preset. It cannot be undone.</p>
    <div className="quit-dialog-actions"><button ref={cancel} className="secondary" disabled={pending} onClick={onCancel}>Keep preset</button><button className="danger" disabled={pending} onClick={onConfirm}>{pending ? 'Deleting…' : 'Delete preset'}</button></div>
  </dialog>;
}
