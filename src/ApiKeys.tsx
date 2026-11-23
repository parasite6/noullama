import { Reveal } from './Reveal';
import { busyButton, useDelayedPending } from './busy-button';
import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { KeyIcon as KeyRound } from '@phosphor-icons/react/dist/icons/Key';
import { TrashIcon as Trash2 } from '@phosphor-icons/react/dist/icons/Trash';
import { CopyButton } from './CopyButton';
import type { LibraryEntry } from './Library';
import { Select } from './Select';
import { SettingRow, SettingSection, StatusNote } from './SettingsLayout';
import { Toggle } from './Toggle';
import { apiKeyAccessSummary, apiKeyExpiry, apiKeyExpiryOptions, apiKeyModelOptions } from './ui-state';

type KeyInfo = {
  id: string;
  name: string;
  prefix: string;
  created: number;
  expires: number | null;
  models: { name: string; aliases: string[] }[] | null;
};

type CreatedKey = { key: string; info: KeyInfo };

export function ApiKeys({ active, desktop, required, models, onToggle, onCopyError, onOpenLibrary }: {
  active: boolean;
  desktop: boolean;
  required: boolean;
  models: LibraryEntry[];
  onToggle: (on: boolean) => void;
  onCopyError: (message: string) => void;
  onOpenLibrary: () => void;
}) {
  const [keys, setKeys] = useState<KeyInfo[] | null>(null);
  const [name, setName] = useState('');
  const [expiry, setExpiry] = useState('never');
  const [date, setDate] = useState('');
  const [specific, setSpecific] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [created, setCreated] = useState<CreatedKey | null>(null);
  const [pending, setPending] = useState<'create' | 'delete' | null>(null);
  const visiblePending = useDelayedPending(pending);
  const [error, setError] = useState('');
  const [deleteTarget, setDeleteTarget] = useState<KeyInfo | null>(null);
  const deleteDialog = useRef<HTMLDialogElement>(null);
  const cancelDelete = useRef<HTMLButtonElement>(null);
  const deleteTrigger = useRef<HTMLButtonElement>(null);
  const panelTitle = useRef<HTMLHeadingElement>(null);
  const groups = models.map(model => apiKeyModelOptions([model]));
  const selectedModels = groups.flat().filter(option => selected.includes(option.id));

  useEffect(() => {
    if (!active || !desktop || keys !== null) return;
    let stale = false;
    invoke<KeyInfo[]>('list_api_keys').then(value => {
      if (!stale) setKeys(value);
    }).catch(reason => {
      if (!stale) setError(String(reason));
    });
    return () => { stale = true; };
  }, [active, desktop, keys]);

  useLayoutEffect(() => {
    const dialog = deleteDialog.current;
    if (!dialog) return;
    if (deleteTarget && !dialog.open) {
      dialog.showModal();
      cancelDelete.current?.focus();
    } else if (!deleteTarget && dialog.open) {
      dialog.close();
      if (deleteTrigger.current?.isConnected) deleteTrigger.current.focus();
      else panelTitle.current?.focus();
      deleteTrigger.current = null;
    }
  }, [deleteTarget]);

  async function create(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending || created || keys === null) return;
    setPending('create');
    setError('');
    try {
      const result = await invoke<CreatedKey>('create_api_key', {
        name,
        expires: apiKeyExpiry(expiry, date),
        models: specific ? selectedModels.map(option => option.selection) : null,
      });
      setKeys(current => [...(current ?? []), result.info]);
      setCreated(result);
      setName('');
      setSelected([]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setPending(null);
    }
  }

  async function remove() {
    if (!deleteTarget || pending) return;
    setPending('delete');
    setError('');
    try {
      const key = deleteTarget;
      setKeys(await invoke<KeyInfo[]>('delete_api_key', { id: key.id }));
      setCreated(current => current?.info.id === key.id ? null : current);
      setDeleteTarget(null);
    } catch (reason) {
      setError(String(reason));
    } finally {
      setPending(null);
    }
  }

  const disabled = !desktop || keys === null || pending !== null || created !== null;
  const dateText = (seconds: number) => new Date(seconds * 1000).toLocaleString();
  function modelOption(option: (typeof groups)[number][number], alias: boolean) {
    const checked = selected.includes(option.id);
    return <button key={option.id} type="button" role="checkbox" aria-checked={checked}
      aria-label={option.label} className={alias ? 'api-key-alias' : 'api-key-model'}
      {...busyButton(!!pending, !desktop, false)} onClick={() => setSelected(current => checked ? current.filter(id => id !== option.id) : [...current, option.id])}>
      <span className="checkbox-mark" aria-hidden="true"/>
      <span>{option.selection.name}</span>
    </button>;
  }

  return <section hidden={!active} className="panel settings-panel api-keys" aria-labelledby="api-keys-title">
    <header className="panel-heading"><span><KeyRound size={16}/><h2 id="api-keys-title" ref={panelTitle} tabIndex={-1}>API keys</h2></span></header>
    <div className="settings-body">
      <div className="setting-rows">
        <SettingRow label="Require API key" description="Require a bearer key for /v1/ and /health."
          control={<Toggle checked={required} label={required ? 'On' : 'Off'} ariaLabel="Require API key" disabled={!desktop} onChange={onToggle}/>}/>
      </div>
      <form onSubmit={create} aria-label="Create API key">
        <SettingSection title="New key">
          <SettingRow label="Name" htmlFor="api-key-name"
            control={<input id="api-key-name" required value={name} disabled={disabled} onChange={event => setName(event.target.value)}/>}/>
          <SettingRow label="Expiry" control={<Select label="Expiry" value={expiry} options={[...apiKeyExpiryOptions]} onChange={setExpiry} busy={!!pending} disabled={!desktop}/>}/>
          <Reveal open={expiry === 'custom'}><SettingRow label="Expiry date" htmlFor="api-key-date" description="Expires at the end of this day, local time."
            control={<input id="api-key-date" type="date" required value={date} disabled={disabled || expiry !== 'custom'} onChange={event => setDate(event.target.value)}/>}/></Reveal>
          <SettingRow label="Access" control={<Select label="Access" value={specific ? 'specific' : 'any'} busy={!!pending} disabled={!desktop}
            options={[{ value: 'any', label: 'Any' }, { value: 'specific', label: 'Specific models' }]} onChange={value => setSpecific(value === 'specific')}/>}/>
          <Reveal open={specific}>{specific && <SettingRow label="Models">
            {groups.length > 0 ? <div className="api-key-models" role="group" aria-label="Specific models">
              {groups.map(([model, ...aliases]) => <div className="api-key-model-group" key={model.id}>
                {modelOption(model, false)}
                {aliases.length > 0 && <div className="api-key-aliases">{aliases.map(alias => modelOption(alias, true))}</div>}
              </div>)}
            </div> : <StatusNote tone="info">Add a model to Library first.</StatusNote>}
            {groups.length > 0 && models.every(model => model.aliases.length === 0) && <div className="api-key-library">
              <StatusNote tone="info">Aliases come from Library. These models have none yet.</StatusNote>
              <button type="button" className="ghost" onClick={onOpenLibrary}>Open Library</button>
            </div>}
          </SettingRow>}</Reveal>
          <SettingRow label=""
            status={visiblePending === 'create' && <StatusNote tone="pending">{specific ? 'Checking model hashes and creating key…' : 'Creating key…'}</StatusNote>}
            control={<button type="submit" className="primary" {...busyButton(!!pending, !desktop || !name.trim() || (specific && !selectedModels.length), pending === 'create')}>Create key</button>}/>
        </SettingSection>
      </form>
      <Reveal open={created} className="settings-spaced">{created && <SettingSection title="New API key" description="Copy this key now. You can only see it here.">
        <SettingRow label={<code className="api-key-secret">{created.key}</code>}
          control={<><CopyButton value={created.key} label="Copy API key" onError={onCopyError}/><button type="button" className="secondary" onClick={() => setCreated(null)}>Done</button></>}/>
      </SettingSection>}</Reveal>
      <Reveal open={error && !deleteTarget} className="settings-spaced">{error && !deleteTarget && <StatusNote tone="error"><span role="alert">{error}</span></StatusNote>}</Reveal>
      <SettingSection title="Keys">
        <Reveal open={keys?.length === 0}>{keys?.length === 0 && <SettingRow label="No API keys"/>}</Reveal>
        {keys?.map(key => {
          const expired = key.expires !== null && key.expires * 1000 <= Date.now();
          const expiryText = expired ? 'Expired' : key.expires === null ? 'Never expires' : `Expires ${dateText(key.expires)}`;
          return <SettingRow key={key.id} label={<span className="api-key-identity">{key.name}<span className={`chip ${expired ? 'warning' : ''}`} title={expired && key.expires !== null ? dateText(key.expires) : undefined}>{expiryText}</span></span>}
            description={<><span className="api-key-meta"><code>{key.prefix}…</code><span>Created {dateText(key.created)}</span></span><span className="api-key-access">{apiKeyAccessSummary(key.models)}</span></>}
            control={<button type="button" className="secondary" aria-label={`Delete API key ${key.name}`} aria-haspopup="dialog" {...busyButton(!!pending, false, false)} onClick={event => {
              deleteTrigger.current = event.currentTarget;
              setError('');
              setDeleteTarget(key);
            }}>Delete</button>}/>;
        })}
      </SettingSection>
    </div>
    <dialog ref={deleteDialog} className="quit-dialog" aria-labelledby="delete-api-key-title" aria-describedby="delete-api-key-description" onCancel={event => {
      event.preventDefault();
      if (!pending) setDeleteTarget(null);
    }}>
      <Trash2 size={24} aria-hidden="true"/>
      <h2 id="delete-api-key-title">Delete API key?</h2>
      <p id="delete-api-key-description" style={{ overflowWrap: 'anywhere' }}>{deleteTarget?.name} (<code>{deleteTarget?.prefix}…</code>) will no longer start requests. Inference already running can finish.</p>
      <Reveal open={error && deleteTarget}>{error && deleteTarget && <StatusNote tone="error"><span role="alert">{error}</span></StatusNote>}</Reveal>
      <div className="quit-dialog-actions">
        <button ref={cancelDelete} type="button" className="secondary" {...busyButton(!!pending, false, false)} onClick={() => setDeleteTarget(null)}>Cancel</button>
        <button type="button" className="danger" {...busyButton(!!pending)} onClick={() => void remove()}>{visiblePending === 'delete' ? 'Deleting…' : 'Delete'}</button>
      </div>
    </dialog>
  </section>;
}
