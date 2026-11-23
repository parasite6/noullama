import React, { useEffect, useState } from 'react';
import { Check, Play, Plus, RefreshCw, RotateCcw, Square, Unplug } from 'lucide-react';
import { AuthorLogo } from './AuthorLogo';
import { CopyButton } from './CopyButton';
import { ModelAliases, Name, type Route } from './ModelNames';
import { KeepContext } from './KeepContext';
import { verifyingText, type Verifying } from './model-state';

// What a card needs from one loaded instance. Everything shown is what the
// instance was loaded with; edits in the load panel only reach it by reloading.
export type CardInstance = {
  id: string; name: string; path: string; device: string; ready: boolean;
  // The file it is hashing before launch, while it does.
  verifying: Verifying | null;
  slot: number; suffix: string; default_suffix: string;
  // Exact extra names for this instance only, held while it is loaded.
  instance_aliases: string[];
  projector: boolean; mmproj_path: string; mtp: boolean; mtp_path: string; layout: string | null;
  context_size: number; gpu_layers: number;
  // settings.device: the device it was loaded on ("CPU", a name such as CUDA0, or a split such as CUDA0,CUDA1); `device` is how it is shown ("All GPUs (2) · 34 GB").
  settings: { decision_mode: string; device: string };
  // This instance's own Keep context switch and history counts.
  keep_context: boolean; context: { entries: number; last_flushed: number };
};

const basename = (path: string) => path.split(/[\\/]/).pop() || path;
// UI labels for the stored decision_mode values and the layout a load resolved to.
export const modeLabels: Record<string, string> = { auto: 'Auto', native: 'Built-in', winnow: 'Prompt layout' };
const layoutLabel = (layout: string) => layout === 'native' ? 'Built-in' : layout === 'winnow-v1' ? 'Prompt layout (Winnow)' : layout;

// One loaded instance on the Loading tab: status, load-time settings, the
// names the API routes to it, and its own Reload and Eject. Keyed by instance
// id so two instances of one file stay apart.
const errorText = (e: unknown) => e instanceof Error ? e.message : String(e);

// The instance suffix: fixed empty for instance 1, editable from instance 2 on.
// It is held by the backend only while the instance is loaded.
function SuffixEditor({ instance, disabled, onSave, onError }: { instance: CardInstance; disabled: boolean; onSave: (suffix: string) => Promise<void>; onError: (message: string) => void }) {
  const [draft, setDraft] = useState(instance.suffix);
  const [saving, setSaving] = useState(false);
  useEffect(() => { setDraft(instance.suffix); }, [instance.suffix]);
  if (instance.slot <= 1) return <div className="instance-suffix"><span className="instance-names-label">Instance suffix</span><small>None. Instance 1 answers to the plain names.</small></div>;
  async function save(value: string) {
    if (saving) return;
    setSaving(true);
    try { await onSave(value); } catch (e) { onError(errorText(e)); setDraft(instance.suffix); } finally { setSaving(false); }
  }
  const changed = draft.trim() !== instance.suffix;
  return <form className="instance-suffix" onSubmit={event => { event.preventDefault(); if (changed) void save(draft); }}>
    <span className="instance-names-label">Instance suffix</span>
    <input aria-label={`Suffix for instance ${instance.slot} of ${instance.name}`} placeholder={instance.default_suffix} maxLength={64} value={draft} disabled={disabled} onChange={event => setDraft(event.target.value)}/>
    <button className="secondary compact" disabled={disabled || !changed}><Check size={13}/>Save suffix</button>
    {instance.suffix !== instance.default_suffix && <button type="button" className="text-button" disabled={disabled} title={`Go back to ${instance.default_suffix}`} onClick={() => void save(instance.default_suffix)}><RotateCcw size={13}/>Use {instance.default_suffix}</button>}
    <small>Added to each of this file's names and aliases. Kept only until this instance is ejected; loading it again starts at {instance.default_suffix}.</small>
  </form>;
}

// This instance's own aliases: exact names (no suffix) that reach only it.
// They are kept apart from the file's models.yml aliases above and are never saved.
function InstanceAliasEditor({ instance, disabled, onSave, onError }: { instance: CardInstance; disabled: boolean; onSave: (aliases: string[]) => Promise<void>; onError: (message: string) => void }) {
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);
  const aliases = instance.instance_aliases;
  async function save(next: string[], clear: boolean) {
    if (saving) return;
    setSaving(true);
    try { await onSave(next); if (clear) setDraft(''); } catch (e) { onError(errorText(e)); } finally { setSaving(false); }
  }
  return <div className="instance-aliases">
    <span className="instance-names-label">Instance aliases</span>
    {aliases.length > 0 ? <div className="model-name-list">{aliases.map(alias => <Name key={alias} value={alias} note="This instance only" removeTitle="Remove instance alias" disabled={disabled} onRemove={() => void save(aliases.filter(other => other !== alias), false)} onError={onError}/>)}</div> : <small>None.</small>}
    <form className="alias-form" onSubmit={event => { event.preventDefault(); if (draft.trim()) void save([...aliases, draft.trim()], true); }}>
      <input aria-label={`New instance alias for instance ${instance.slot} of ${instance.name}`} placeholder="Add a name for this instance only, such as billing" maxLength={64} value={draft} disabled={disabled} onChange={event => setDraft(event.target.value)}/>
      <button className="secondary compact" disabled={disabled || !draft.trim()}><Plus size={13}/>Add instance alias</button>
    </form>
    <small>Exact names that reach only this instance; no suffix is added. Kept only until this instance is ejected or reloaded, and never saved.</small>
  </div>;
}

export function InstanceCard({ instance, account, runtime, aliases, routes, displayName, stale, canReload, starting, ejecting, reloadDisabled, namesDisabled, keepContext, onEject, onReload, onSaveAliases, onSaveSuffix, onSaveInstanceAliases, onKeepContext, onClearContext, onError }: {
  instance: CardInstance; account: string | null; runtime: string; aliases: string[]; routes: Route[];
  // The instance's first API identifier, such as qwen-i-2.
  displayName: string;
  onSaveSuffix: (suffix: string) => Promise<void>;
  onSaveInstanceAliases: (aliases: string[]) => Promise<void>;
  // The switch as shown: the snapshot's value, or the one just chosen while it saves.
  keepContext: boolean; onKeepContext: (on: boolean) => void; onClearContext: () => void;
  // The load panel's settings differ from this instance's (same file selected there).
  stale: boolean; canReload: boolean; starting: boolean; ejecting: boolean; reloadDisabled: boolean; namesDisabled: boolean;
  onEject: () => void; onReload: () => void;
  onSaveAliases: (path: string, aliases: string[]) => Promise<void>; onError: (message: string) => void;
}) {
  const settings: [string, string, string?][] = [
    ['Device', instance.device || instance.settings.device],
    ['Vision', instance.projector ? basename(instance.mmproj_path) : 'Off', instance.projector ? instance.mmproj_path : undefined],
    ['MTP', instance.mtp ? instance.mtp_path ? basename(instance.mtp_path) : 'Built-in layers' : 'Off', instance.mtp && instance.mtp_path ? instance.mtp_path : undefined],
    // Auto shows the layout it resolved to; a forced mode is its own layout.
    ['Decision mode', `${modeLabels[instance.settings.decision_mode] ?? instance.settings.decision_mode}${instance.layout && !['native', 'winnow'].includes(instance.settings.decision_mode) ? ` · ${layoutLabel(instance.layout)}` : ''}`],
    ['Context', `${instance.context_size.toLocaleString()} tokens`],
    ['GPU layers', instance.gpu_layers === 0 ? 'None (CPU)' : String(instance.gpu_layers)],
    ['llama.cpp', runtime],
  ];
  return <section className="panel loaded-model instance-card" aria-label={`Loaded model ${displayName}`}>
    <div className="loaded-head">
      <AuthorLogo author={account} size={34}/>
      <span className="loaded-name">
        <span className="identity-line"><strong title={instance.path}>{displayName}</strong><CopyButton value={displayName} label={`Copy model name ${displayName}`} size={12} onError={onError}/></span>
        <small>{instance.ready ? instance.device : 'Loading…'} · Instance {instance.slot}</small>
      </span>
      <span className={`chip ${instance.ready ? 'success' : ''}`}><span className={`status-dot ${instance.ready ? 'green' : 'amber'}`}/>{instance.ready ? 'Ready' : verifyingText(instance.verifying) ?? 'Loading'}</span>
      {canReload && <button className="secondary compact" disabled={reloadDisabled} title="Eject this model and load it again with the load panel's settings" onClick={onReload}>{starting ? <RefreshCw className="spin" size={13}/> : <Play size={13}/>}{starting ? 'Starting…' : 'Reload'}</button>}
      <button className="secondary compact" disabled={ejecting} title={instance.ready ? `Unload ${displayName}; other loaded models keep running` : undefined} onClick={onEject}>{instance.ready ? <><Unplug size={13}/>Eject</> : <><Square size={13}/>Cancel loading</>}</button>
    </div>
    <dl className="instance-settings">{settings.map(([label, value, title]) => <div key={label}><dt>{label}</dt><dd title={title}>{value}</dd></div>)}</dl>
    {stale && <p className="instance-stale">The load panel's settings differ from the ones this model was loaded with. They apply only after a reload.</p>}
    <div className="instance-names"><span className="instance-names-label">API names</span>
      <ModelAliases path={instance.path} instance={instance.id} aliases={aliases} routes={routes} disabled={namesDisabled} onSave={onSaveAliases} onError={onError}/>
    </div>
    <SuffixEditor instance={instance} disabled={namesDisabled} onSave={onSaveSuffix} onError={onError}/>
    <InstanceAliasEditor instance={instance} disabled={namesDisabled} onSave={onSaveInstanceAliases} onError={onError}/>
    <div className="instance-keep"><KeepContext on={keepContext} entries={instance.context.entries} lastFlushed={instance.context.last_flushed} disabled={namesDisabled} onToggle={onKeepContext} onClear={onClearContext}/></div>
  </section>;
}
