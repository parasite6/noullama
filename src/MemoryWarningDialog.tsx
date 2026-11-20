import React, { useEffect, useRef } from 'react';

export type CrowdedWarning = { reason: string; models: { id: string; name: string }[] };

export function MemoryWarningDialog({ warning, pending, hide, onHide, onCancel, onEject }: {
  warning: CrowdedWarning | null;
  pending: boolean;
  hide: boolean;
  onHide: (value: boolean) => void;
  onCancel: () => void;
  onEject: (id: string) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const cancel = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (warning && !element.open) { element.showModal(); cancel.current?.focus(); }
    else if (!warning && element.open) element.close();
  }, [warning]);
  return <dialog ref={dialog} className="quit-dialog memory-dialog" aria-labelledby="memory-warning-title" aria-describedby="memory-warning-description" onCancel={event => { event.preventDefault(); if (!pending) onCancel(); }}>
    <h2 id="memory-warning-title">This probably won't fit</h2>
    <p id="memory-warning-description">{warning?.reason} Would you like to cancel or eject another model and try again?</p>
    <label><input type="checkbox" checked={hide} disabled={pending} onChange={e => onHide(e.target.checked)}/> Don't show this again</label>
    <p>This hides only this message. OOM protection stays enabled.</p>
    <div className="quit-dialog-actions"><button ref={cancel} className="secondary" disabled={pending} onClick={onCancel}>Cancel</button>
      {warning?.models.map(model => <button key={model.id} className="secondary" disabled={pending} onClick={() => onEject(model.id)}>Eject {model.name}</button>)}
    </div>
  </dialog>;
}
