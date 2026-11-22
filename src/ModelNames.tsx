import React, { useState } from 'react';
import { Plus, X } from 'lucide-react';
import { CopyButton } from './CopyButton';

// `name` is the identifier a request sends: `base` plus the instance's suffix.
// An instance alias belongs to one loaded instance and is never suffixed.
export type Route = { name: string; kind: 'name' | 'alias' | 'instance_alias'; base: string; path: string; instance: string | null; slot: number; suffix: string; ready: boolean; fallthrough: boolean; conflict: boolean };

const errorText = (e: unknown) => e instanceof Error ? e.message : String(e);
const stemOf = (path: string) => (path.split(/[\\/]/).pop() ?? path).replace(/\.gguf$/i, '');

// The identifier a request sends as "model" to reach any free instance.
export const ANY_AVAILABLE = 'any-available';

export function Name({ value, note, warn, disabled, removeTitle = 'Remove alias', onRemove, onError }: { value: string; note: string; warn?: boolean; disabled?: boolean; removeTitle?: string; onRemove?: () => void; onError: (message: string) => void }) {
  return <span className={`model-name ${warn ? 'warn' : ''}`}>
    <code>{value}</code><small>{note}</small><CopyButton value={value} label={`Copy model name ${value}`} size={12} onError={onError}/>
    {onRemove && <button type="button" className="icon-button" aria-label={`${removeTitle} ${value}`} title={removeTitle} disabled={disabled} onClick={onRemove}><X size={12}/></button>}
  </span>;
}

// The names the local API routes to one model file: its file name while it
// is loaded, and any aliases. An alias belongs to one file and keeps working
// across reloads. With `instance` (an instance card), the list shows that
// instance's identifiers (names plus its suffix); aliases are still edited
// per file, without the suffix.
export function ModelAliases({ path, instance, aliases, routes, disabled, onSave, onError }: { path: string; instance?: string; aliases: string[]; routes?: Route[]; disabled?: boolean; onSave: (path: string, aliases: string[]) => Promise<void>; onError: (message: string) => void }) {
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);
  async function save(next: string[]) {
    setSaving(true);
    try { await onSave(path, next); setDraft(''); } catch (e) { onError(errorText(e)); } finally { setSaving(false); }
  }
  const ownRoutes = (routes ?? []).filter(route => route.instance === instance);
  const conflictNote = (route: Route) => route.kind === 'instance_alias' ? 'Shared; remove this instance alias' : route.slot > 1 ? 'Shared; change the suffix or rename it in Library' : 'Shared; rename it in Library';
  const plainNote = (route: Route) => route.kind === 'name' ? 'Name' : route.kind === 'alias' ? 'Alias' : 'Instance alias';
  // Without an instance, only the plain (unsuffixed) names, once each.
  const mine = (routes ?? []).filter(route => route.path === path && route.kind === 'name' && route.suffix === '').filter((route, index, all) => all.findIndex(other => other.name === route.name) === index);
  return <div className="model-aliases">
    {instance ? <div className="model-name-list">
      {ownRoutes.map(route => <Name key={`${route.kind}:${route.name}`} value={route.name} warn={route.conflict}
        note={route.conflict ? conflictNote(route) : route.fallthrough ? `${plainNote(route)} · falls through` : plainNote(route)}
        disabled={disabled || saving} onRemove={route.kind === 'alias' && !route.fallthrough ? () => void save(aliases.filter(other => other !== route.base)) : undefined} onError={onError}/>)}
    </div> : <div className="model-name-list">
      {mine.map(route => <Name key={`${route.kind}:${route.name}`} value={route.name} note={route.conflict ? 'Shared; rename it in Library' : 'Name'} warn={route.conflict} onError={onError}/>)}
      {!mine.length && <Name value={stemOf(path)} note="Name" onError={onError}/>}
      {aliases.map(alias => <Name key={alias} value={alias} note="Alias" disabled={disabled || saving} onRemove={() => void save(aliases.filter(other => other !== alias))} onError={onError}/>)}
    </div>}
    <form className="alias-form" onSubmit={event => { event.preventDefault(); if (draft.trim()) void save([...aliases, draft.trim()]); }}>
      <input aria-label={`New alias for ${stemOf(path)}`} placeholder="Add an alias, such as triage or gpt-4o-mini" maxLength={64} value={draft} onChange={event => setDraft(event.target.value)}/>
      <button className="secondary compact" disabled={disabled || saving || !draft.trim()}><Plus size={13}/>Add alias</button>
    </form>
  </div>;
}

// Local API: the rules for "model" and the names that do not reach a loaded
// model yet. A loaded model's own names are edited on its card.
export function ModelRouting({ loadedCount, anyReady, whenBusy, routes, aliasesFor, disabled, onSave, onError }: { loadedCount: number; anyReady: boolean; whenBusy: string; routes: Route[]; aliasesFor: (path: string) => string[]; disabled?: boolean; onSave: (path: string, aliases: string[]) => Promise<void>; onError: (message: string) => void }) {
  const unloaded = routes.filter(route => !route.instance && !route.fallthrough);
  const withoutFirst = routes.filter(route => !route.instance && route.fallthrough);
    return <section className="panel model-names">
    <div className="panel-heading"><span>Model routing</span><span className="muted normal-weight">Every request sends one of these names as "model"</span></div>
    <div className="model-names-body">
      <p className="hint">{loadedCount === 0 ? 'Load a model to route requests to it. Several can be loaded at once; each answers to its name and aliases from Library.' : 'Each loaded model answers to the names on its card under Loaded.'}</p>
      <div className="route-group">
        <div className="route-model"><span className={`status-dot ${anyReady ? 'green' : ''}`}/><strong>Any free instance</strong><small>{anyReady ? 'Reaches a ready instance now' : 'Answers 503 until an instance is ready'}</small></div>
        <div className="model-name-list"><Name value={ANY_AVAILABLE} note="Any model" onError={onError}/></div>
        <small>Sends the request to whichever ready instance, of any model, is not running a request; the one idle longest goes first. When every instance is busy it {whenBusy === 'wait' ? 'waits up to 120 seconds for the first to finish' : 'gets 503'} (When every instance is busy, in Settings). Each instance keeps its own Keep context history.</small>
      </div>
      {unloaded.length > 0 && <div className="route-group">
        <div className="route-model"><span className="status-dot"/><strong>Not loaded</strong><small>These names answer 503 until their model is loaded</small></div>
        <div className="model-name-list">{unloaded.map(route => <Name key={`${route.path}:${route.name}`} value={route.name} note={route.kind === 'name' ? 'Name' : `Alias of ${unloaded.find(r => r.path === route.path && r.kind === 'name')?.name ?? stemOf(route.path)}`} disabled={disabled} onRemove={route.kind === 'alias' ? () => void onSave(route.path, aliasesFor(route.path).filter(other => other !== route.name)).catch(e => onError(errorText(e))) : undefined} onError={onError}/>)}</div>
      </div>}
      {withoutFirst.length > 0 && <div className="route-group">
        <div className="route-model"><span className="status-dot amber"/><strong>Instance 1 not loaded</strong><small>These plain names answer 503 while other instances of their file are loaded. Fall through to another instance in Settings sends them to the lowest-numbered one.</small></div>
        <div className="model-name-list">{withoutFirst.map(route => <Name key={`${route.path}:${route.name}`} value={route.name} note={route.kind === 'name' ? 'Name' : 'Alias'} onError={onError}/>)}</div>
      </div>}
      <small>"model" is required and case-sensitive: send a name exactly as shown. A request without one gets 400; an unknown name gets 404. Names and aliases live in models.yml (Library); each belongs to one model file. A second instance of a file answers to the same names with its suffix, such as -i-2. /v1/models lists the names that reach a ready model; responses echo the name you sent.</small>
    </div>
  </section>;
}
