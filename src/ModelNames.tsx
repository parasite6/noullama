import React, { useState } from 'react';
import { Plus, X } from 'lucide-react';
import { CopyButton } from './CopyButton';

export type Route = { name: string; kind: 'name' | 'alias'; path: string; instance: string | null; ready: boolean; conflict: boolean };
export type LoadedInstance = { id: string; name: string; path: string; device: string; ready: boolean };

const errorText = (e: unknown) => e instanceof Error ? e.message : String(e);
const stemOf = (path: string) => (path.split(/[\\/]/).pop() ?? path).replace(/\.gguf$/i, '');

function Name({ value, note, warn, disabled, onRemove, onError }: { value: string; note: string; warn?: boolean; disabled?: boolean; onRemove?: () => void; onError: (message: string) => void }) {
  return <span className={`model-name ${warn ? 'warn' : ''}`}>
    <code>{value}</code><small>{note}</small><CopyButton value={value} label={`Copy model name ${value}`} size={12} onError={onError}/>
    {onRemove && <button type="button" className="icon-button" aria-label={`Remove alias ${value}`} title="Remove alias" disabled={disabled} onClick={onRemove}><X size={12}/></button>}
  </span>;
}

// The names the local API routes to one model file: its file name while it
// is loaded, and any aliases. An alias belongs to one file and keeps working
// across reloads.
export function ModelAliases({ path, aliases, routes, disabled, onSave, onError }: { path: string; aliases: string[]; routes?: Route[]; disabled?: boolean; onSave: (path: string, aliases: string[]) => Promise<void>; onError: (message: string) => void }) {
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);
  async function save(next: string[]) {
    setSaving(true);
    try { await onSave(path, next); setDraft(''); } catch (e) { onError(errorText(e)); } finally { setSaving(false); }
  }
  const mine = (routes ?? []).filter(route => route.path === path && route.kind !== 'alias');
  return <div className="model-aliases">
    <div className="model-name-list">
      {mine.map(route => <Name key={`${route.kind}:${route.name}`} value={route.name} note={route.conflict ? 'Shared; rename it in Library' : 'Name'} warn={route.conflict} onError={onError}/>)}
      {!mine.length && <Name value={stemOf(path)} note="Name" onError={onError}/>}
      {aliases.map(alias => <Name key={alias} value={alias} note="Alias" disabled={disabled || saving} onRemove={() => void save(aliases.filter(other => other !== alias))} onError={onError}/>)}
    </div>
    <form className="alias-form" onSubmit={event => { event.preventDefault(); if (draft.trim()) void save([...aliases, draft.trim()]); }}>
      <input aria-label={`New alias for ${stemOf(path)}`} placeholder="Add an alias, such as triage or gpt-4o-mini" maxLength={64} value={draft} onChange={event => setDraft(event.target.value)}/>
      <button className="secondary compact" disabled={disabled || saving || !draft.trim()}><Plus size={13}/>Add alias</button>
    </form>
  </div>;
}

// Local API: which name reaches which loaded model, with aliases editable in place.
export function ModelRouting({ instances, routes, aliasesFor, disabled, onSave, onError }: { instances: LoadedInstance[]; routes: Route[]; aliasesFor: (path: string) => string[]; disabled?: boolean; onSave: (path: string, aliases: string[]) => Promise<void>; onError: (message: string) => void }) {
  const unloaded = routes.filter(route => !route.instance);
    return <section className="panel model-names">
    <div className="panel-heading"><span>Model routing</span><span className="muted normal-weight">Every request sends one of these names as "model"</span></div>
    <div className="model-names-body">
      {instances.length === 0 && <p className="hint">Load a model to route requests to it. Several can be loaded at once; each answers to its name and aliases from Library.</p>}
      {instances.map(instance => <div className="route-group" key={instance.id}>
        <div className="route-model"><span className={`status-dot ${instance.ready ? 'green' : 'amber'}`}/><strong title={instance.path}>{instance.name}</strong><small>{instance.ready ? instance.device || 'Ready' : 'Loading'}</small></div>
        <ModelAliases path={instance.path} aliases={aliasesFor(instance.path)} routes={routes} disabled={disabled} onSave={onSave} onError={onError}/>
      </div>)}
      {unloaded.length > 0 && <div className="route-group">
        <div className="route-model"><span className="status-dot"/><strong>Not loaded</strong><small>These names answer 503 until their model is loaded</small></div>
        <div className="model-name-list">{unloaded.map(route => <Name key={`${route.path}:${route.name}`} value={route.name} note={route.kind === 'name' ? 'Name' : `Alias of ${unloaded.find(r => r.path === route.path && r.kind === 'name')?.name ?? stemOf(route.path)}`} disabled={disabled} onRemove={route.kind === 'alias' ? () => void onSave(route.path, aliasesFor(route.path).filter(other => other !== route.name)).catch(e => onError(errorText(e))) : undefined} onError={onError}/>)}</div>
      </div>}
      <small>"model" is required and case-sensitive: send a name exactly as shown. A request without one gets 400; an unknown name gets 404. Names and aliases live in models.yml (Library); each belongs to one model file. /v1/models lists the names that reach a ready model; responses echo the name you sent.</small>
    </div>
  </section>;
}
