import React, { useEffect, useState } from 'react';
import { AlertCircle, ArchiveRestore, FileCog, FolderOpen, Plus, RefreshCw, Save, Trash2, X } from 'lucide-react';
import { AuthorLogo, authorFromPath } from './AuthorLogo';
import { CopyButton } from './CopyButton';

// One entry of models.yml, as the editor holds it.
export type LibraryEntry = { name: string; aliases: string[]; path: string; account: string; sha256: string };
// As the backend reports it: the entry plus what it found about the file.
export type LibraryRow = LibraryEntry & { missing: boolean; hashing: boolean; mismatch: string | null };
export type LibraryState = { entries: LibraryRow[]; error: string | null; revision: number; file: string; backups: string };

const errorText = (e: unknown) => e instanceof Error ? e.message : String(e);
const stemOf = (path: string) => (path.split(/[\\/]/).pop() ?? path).replace(/\.gguf$/i, '');
const strip = ({ name, aliases, path, account, sha256 }: LibraryEntry): LibraryEntry => ({ name, aliases, path, account, sha256 });
const same = (a: LibraryEntry[], b: LibraryEntry[]) => JSON.stringify(a.map(strip)) === JSON.stringify(b.map(strip));

function uniqueName(base: string, entries: LibraryEntry[]) {
  const taken = new Set(entries.flatMap(e => [e.name, ...e.aliases]).map(n => n.toLowerCase()));
  const clean = base.replace(/[^A-Za-z0-9._:/-]/g, '-').slice(0, 60) || 'model';
  for (let n = 1; ; n++) { const name = n === 1 ? clean : `${clean}-${n}`; if (!taken.has(name.toLowerCase())) return name; }
}

function AliasInput({ aliases, onChange, label }: { aliases: string[]; onChange: (aliases: string[]) => void; label: string }) {
  const [draft, setDraft] = useState('');
  const add = () => { const value = draft.trim(); if (value && !aliases.includes(value)) onChange([...aliases, value]); setDraft(''); };
  return <div className="alias-input">
    {aliases.map(alias => <span className="model-name" key={alias}><code>{alias}</code><button type="button" className="icon-button" aria-label={`Remove alias ${alias}`} title="Remove alias" onClick={() => onChange(aliases.filter(a => a !== alias))}><X size={12}/></button></span>)}
    <input aria-label={label} placeholder={aliases.length ? 'Add another' : 'Add an alias'} maxLength={64} value={draft} onChange={e => setDraft(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); add(); } }} onBlur={add}/>
  </div>;
}

type Props = {
  library: LibraryState | null; loadedPaths: string[]; desktop: boolean;
  onSave: (entries: LibraryEntry[], revision: number) => Promise<void>;
  onReload: () => Promise<void>; onOpenFile: () => Promise<void>; onOpenBackups: () => Promise<void>;
  onRecordSha: (path: string) => Promise<void>; onBrowse: () => Promise<string | null>;
  onError: (message: string) => void;
};

// The Library tab: every model in models.yml, editable field by field. Saving
// writes the file (backing up the old one); edits made in a text editor show
// up here as soon as they are saved.
export function Library({ library, loadedPaths, desktop, onSave, onReload, onOpenFile, onOpenBackups, onRecordSha, onBrowse, onError }: Props) {
  const [draft, setDraft] = useState<LibraryEntry[]>([]);
  const [base, setBase] = useState<{ revision: number; entries: LibraryEntry[] } | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const rows = library?.entries ?? [];
  const dirty = !!base && !same(draft, base.entries);
  // Take the file's newest contents whenever there is nothing unsaved to lose.
  const latest = JSON.stringify(rows.map(strip));
  useEffect(() => {
    if (!library || dirty || (base && base.revision === library.revision && same(base.entries, library.entries))) return;
    setDraft(library.entries.map(strip)); setBase({ revision: library.revision, entries: library.entries.map(strip) });
  }, [latest, library?.revision, dirty]);
  const stale = !!library && !!base && dirty && base.revision !== library.revision;
  const accounts = [...new Set([...draft.map(e => e.account), ...draft.map(e => authorFromPath(e.path) ?? '')].filter(Boolean))].sort();
  const edit = (index: number, change: Partial<LibraryEntry>) => setDraft(entries => entries.map((e, i) => i === index ? { ...e, ...change } : e));
  async function run(name: string, task: () => Promise<void>) {
    setPending(name);
    try { await task(); } catch (e) { onError(errorText(e)); } finally { setPending(null); }
  }
  async function addModel() {
    const path = await onBrowse();
    if (!path) return;
    if (draft.some(e => e.path === path)) { onError(`${stemOf(path)} is already in models.yml.`); return; }
    setDraft(entries => [...entries, { name: uniqueName(stemOf(path), entries), aliases: [], path, account: authorFromPath(path) ?? '', sha256: '' }]);
  }
  function discard() {
    if (!library) return;
    setDraft(library.entries.map(strip)); setBase({ revision: library.revision, entries: library.entries.map(strip) });
  }
  // After a save the backend's tidied copy (trimmed names and so on) replaces the draft.
  // A sha256 recorded in the background while you edited is kept, unless you changed that field yourself.
  const save = () => run('save', async () => {
    if (!base) return;
    const merged = draft.map(entry => {
      const before = base.entries.find(e => e.path === entry.path);
      const now = rows.find(r => r.path === entry.path);
      return before && now && before.sha256 === entry.sha256 ? { ...entry, sha256: now.sha256 } : entry;
    });
    await onSave(merged, base.revision); setBase(null);
  });

  return <div className="library">
    <section className="panel library-toolbar">
      <div className="library-file"><FileCog size={17}/><span className="identity-line"><code title={library?.file}>{library?.file ?? 'models.yml'}</code>{library?.file && <CopyButton value={library.file} label="Copy models.yml path" size={12} onError={onError}/>}</span></div>
      <div className="library-actions">
        <button className="secondary compact" disabled={!desktop || !!pending} onClick={() => void run('add', addModel)}><Plus size={13}/>Add model</button>
        <button className="secondary compact" disabled={!desktop || !!pending} title="Open models.yml in your default editor" onClick={() => void run('open', onOpenFile)}><FolderOpen size={13}/>Edit models.yml</button>
        <button className="secondary compact" disabled={!desktop || !!pending} title="A copy is kept before every change; the newest 20 stay" onClick={() => void run('backups', onOpenBackups)}><ArchiveRestore size={13}/>Backups</button>
        <button className="text-button" disabled={!desktop || !!pending || dirty} title={dirty ? 'Save or discard your changes first' : 'Read models.yml again'} onClick={() => void run('reload', onReload)}><RefreshCw size={13} className={pending === 'reload' ? 'spin' : ''}/>Reload</button>
      </div>
    </section>
    {library?.error && <div className="load-error" role="alert"><AlertCircle size={18}/><div><strong>models.yml has an error</strong><pre>{library.error}</pre></div></div>}
    {stale && <div className="notice error" role="alert"><span className="notice-text"><AlertCircle size={16}/><span>models.yml changed on disk while you were editing. Discard your changes to see it, or copy them over afterwards.</span></span></div>}
    {draft.length === 0 && <div className="local-empty"><FileCog size={18}/><p>No models yet. Downloads and models found on this machine are added automatically, or click <strong>Add model</strong>.</p></div>}
    <div className="library-list">{draft.map((entry, index) => {
      const row = rows.find(r => r.path === entry.path);
      const loaded = loadedPaths.includes(entry.path);
      const account = entry.account || authorFromPath(entry.path);
      return <section className="panel library-entry" key={index}>
        <div className="library-head">
          <AuthorLogo author={account || null} size={36}/>
          <input className="library-name" aria-label={`Name of ${stemOf(entry.path)}`} title="The model's API ID and display name; requests send it as model, exact case" value={entry.name} maxLength={64} onChange={e => edit(index, { name: e.target.value })}/>
          <span className="library-chips">
            {loaded && <span className="chip success"><span className="status-dot green"/>Loaded</span>}
            {row?.missing && <span className="chip warning" title="No file at this path. Fix the path, or remove the entry.">File missing</span>}
            {row?.hashing && <span className="chip"><RefreshCw className="spin" size={12}/>Checking sha256</span>}
            {row?.mismatch && <span className="chip warning" title={`The file's sha256 is now ${row.mismatch}`}>sha256 changed</span>}
          </span>
          <button className="icon-button" aria-label={`Remove ${entry.name} from models.yml`} title="Remove from models.yml (the file stays on disk)" onClick={() => setDraft(entries => entries.filter((_, i) => i !== index))}><Trash2 size={15}/></button>
        </div>
        <div className="library-fields">
          <label>Aliases<AliasInput label={`Aliases of ${entry.name}`} aliases={entry.aliases} onChange={aliases => edit(index, { aliases })}/></label>
          <label>File<div className="file-input-row"><input aria-label={`Path of ${entry.name}`} value={entry.path} onChange={e => edit(index, { path: e.target.value })}/><button type="button" className="secondary" disabled={!desktop} onClick={() => void run('browse', async () => { const path = await onBrowse(); if (path) edit(index, { path, account: entry.account || authorFromPath(path) || '' }); })}><FolderOpen size={14}/>Browse</button></div></label>
          <label>Hugging Face account<input list="library-accounts" aria-label={`Account of ${entry.name}`} placeholder={authorFromPath(entry.path) ?? 'For the avatar, such as LiquidAI'} value={entry.account} maxLength={96} onChange={e => edit(index, { account: e.target.value })}/></label>
          <div className="library-sha"><span className="field-label">sha256</span>
            <div className="sha-row">{entry.sha256 ? <><code title={entry.sha256}>{entry.sha256.slice(0, 16)}…{entry.sha256.slice(-8)}</code><CopyButton value={entry.sha256} label={`Copy sha256 of ${entry.name}`} size={12} onError={onError}/></> : <span className="muted">{row?.hashing ? 'Reading the file…' : 'Not recorded yet'}</span>}
              {row?.mismatch
                ? <button type="button" className="secondary compact" disabled={!desktop || dirty || !!pending} title={dirty ? 'Save or discard your changes first' : `Record ${row.mismatch} as the expected sha256`} onClick={() => void run('sha', () => onRecordSha(entry.path))}>Accept new sha256</button>
                : <button type="button" className="text-button" disabled={!desktop || dirty || !!pending || !!row?.missing || !row} title={dirty ? 'Save or discard your changes first' : 'Read the whole file and record its sha256'} onClick={() => void run('sha', () => onRecordSha(entry.path))}>{pending === 'sha' ? 'Reading…' : entry.sha256 ? 'Recompute' : 'Record now'}</button>}
              {entry.sha256 && <button type="button" className="text-button" title="Forget the recorded sha256" onClick={() => edit(index, { sha256: '' })}>Clear</button>}
            </div>
          </div>
        </div>
      </section>;
    })}</div>
    <datalist id="library-accounts">{accounts.map(account => <option key={account} value={account}/>)}</datalist>
    <div className={`save-bar ${dirty ? 'dirty' : ''}`}><span className="hint">{dirty ? 'Unsaved changes to models.yml. Saving backs up the current file first.' : 'models.yml is up to date. Edits made in a text editor appear here when saved.'}</span><span className="save-actions">{dirty && <button className="text-button" onClick={discard}>Discard</button>}<button className={dirty ? 'primary' : 'secondary'} disabled={!dirty || !desktop || !!pending} onClick={save}><Save size={14}/>Save models.yml</button></span></div>
  </div>;
}

// The Settings tab's sha256 choice.
export const SHA_CHECKS: { value: string; label: string; detail: string }[] = [
  { value: 'once', label: 'Hash once', detail: 'Each file is hashed once in the background. Before a load the hash is read again only if the file\'s size or modified time changed. A mismatch stops the load until you accept the new sha256 in Library.' },
  { value: 'every_load', label: 'Every load', detail: 'The whole file is hashed before every load. Safest, but a large model takes 10–30 seconds longer to load.' },
  { value: 'warn', label: 'Warn only', detail: 'Hashed as with Hash once, but a mismatch only adds a warning to the log and the model loads anyway.' },
  { value: 'off', label: 'Off', detail: 'No hashing. sha256 values already in models.yml are kept but not checked.' },
];
