import React, { useState } from 'react';
import { Plus, X } from 'lucide-react';
import { CopyButton } from './CopyButton';

// The names the local API accepts for the loaded model. "noullama" and the
// file name always work; custom aliases are saved with the settings.
export function ModelNames({ fileName, aliases, disabled, onSave, onError }: { fileName: string | null; aliases: string[]; disabled?: boolean; onSave: (aliases: string[]) => Promise<void>; onError: (message: string) => void }) {
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);
  async function save(next: string[]) {
    setSaving(true);
    try { await onSave(next); setDraft(''); } catch (e) { onError(e instanceof Error ? e.message : String(e)); } finally { setSaving(false); }
  }
  const name = (value: string, note: string, remove?: () => void) => <span className="model-name" key={value}>
    <code>{value}</code><small>{note}</small><CopyButton value={value} label={`Copy model name ${value}`} size={12} onError={onError}/>
    {remove && <button type="button" className="icon-button" aria-label={`Remove ${value}`} title="Remove" disabled={disabled || saving} onClick={remove}><X size={12}/></button>}
  </span>;
  return <section className="panel model-names">
    <div className="panel-heading"><span>Model names</span><span className="muted normal-weight">Any of these works as "model" in a request</span></div>
    <div className="model-names-body">
      <div className="model-name-list">
        {name('noullama', 'Always')}
        {fileName && name(fileName, 'File name')}
        {aliases.map(alias => name(alias, 'Alias', () => void save(aliases.filter(other => other !== alias))))}
      </div>
      <form className="preset-save" onSubmit={event => { event.preventDefault(); if (draft.trim()) void save([...aliases, draft.trim()]); }}>
        <input aria-label="New model alias" placeholder="Add an alias, such as gpt-4o-mini or triage" maxLength={64} value={draft} onChange={event => setDraft(event.target.value)}/>
        <button className="secondary compact" disabled={disabled || saving || !draft.trim()}><Plus size={13}/>Add alias</button>
      </form>
      <small>Letters, digits and . _ : / - only. /v1/models lists every name while a model is loaded; responses echo the name you sent.</small>
    </div>
  </section>;
}
