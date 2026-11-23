import { Reveal } from './Reveal';
import { busyButton, useDelayedPending } from './busy-button';
import { LibraryList, type LibraryScroll } from './LibraryList';
import { equalJson } from './refresh-state';
import React, { createContext, memo, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { WarningCircleIcon as AlertCircle } from '@phosphor-icons/react/dist/icons/WarningCircle';
import { ArchiveIcon as ArchiveRestore } from '@phosphor-icons/react/dist/icons/Archive';
import { FileCodeIcon as FileCog } from '@phosphor-icons/react/dist/icons/FileCode';
import { FolderOpenIcon as FolderOpen } from '@phosphor-icons/react/dist/icons/FolderOpen';
import { PlusIcon as Plus } from '@phosphor-icons/react/dist/icons/Plus';
import { ArrowsClockwiseIcon as RefreshCw } from '@phosphor-icons/react/dist/icons/ArrowsClockwise';
import { FloppyDiskIcon as Save } from '@phosphor-icons/react/dist/icons/FloppyDisk';
import { TrashIcon as Trash2 } from '@phosphor-icons/react/dist/icons/Trash';
import { XIcon as X } from '@phosphor-icons/react/dist/icons/X';
import { AuthorLogo, authorFromPath } from './AuthorLogo';
import { CopyButton } from './CopyButton';
import { ComboInput } from './Select';

// One entry of models.yml, as the editor holds it.
export type LibraryEntry = { name: string; aliases: string[]; path: string; account: string; sha256: string; mtp: string; mtp_sha256: string; mmproj: string; mmproj_sha256: string };
export type Slot = 'model' | 'mtp' | 'mmproj';
type FileStatus = { missing: boolean; hashing: boolean; mismatch: string | null };
// As the backend reports it: the entry plus what it found about each file.
export type LibraryRow = LibraryEntry & { status: Record<Slot, FileStatus> };
// Heads and projectors found on this machine, offered in the fields' lists.
export type Companion = { path: string; kind: 'mtp' | 'mmproj' };
const SUM: Record<Slot, 'sha256' | 'mtp_sha256' | 'mmproj_sha256'> = { model: 'sha256', mtp: 'mtp_sha256', mmproj: 'mmproj_sha256' };
export type LibraryState = { entries: LibraryRow[]; error: string | null; revision: number; file: string; backups: string };

const errorText = (e: unknown) => e instanceof Error ? e.message : String(e);
const stemOf = (path: string) => (path.split(/[\\/]/).pop() ?? path).replace(/\.gguf$/i, '');
const strip = ({ name, aliases, path, account, sha256, mtp, mtp_sha256, mmproj, mmproj_sha256 }: LibraryEntry): LibraryEntry => ({ name, aliases, path, account, sha256, mtp: mtp ?? '', mtp_sha256: mtp_sha256 ?? '', mmproj: mmproj ?? '', mmproj_sha256: mmproj_sha256 ?? '' });
const folderOf = (path: string) => path.replace(/[\\/][^\\/]*$/, '');
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

// A file's recorded sha256 with the buttons that change it.
const LibraryDirty = createContext(false);

function ShaRow({ label, sum, status, canRecord, blocked, pending, onRecord, onClear, onError }: { label: string; sum: string; status?: FileStatus; canRecord: boolean; blocked: string | null; pending: boolean; onRecord: () => void; onClear: () => void; onError: (message: string) => void }) {
  const reading = useDelayedPending(pending);
  if (useContext(LibraryDirty) && !blocked) blocked = 'Save or discard your changes first';
  return <div className="sha-row">{sum ? <><code title={sum}>{sum.slice(0, 16)}…{sum.slice(-8)}</code><CopyButton value={sum} label={`Copy sha256 of ${label}`} size={12} onError={onError}/></> : <span className="muted">{status?.hashing ? 'Reading the file…' : 'Not recorded yet'}</span>}
    {status?.mismatch
      ? <button type="button" className="secondary compact" {...busyButton(pending, !!blocked)} title={blocked ?? `Record ${status.mismatch} as the expected sha256`} onClick={onRecord}>Accept new sha256</button>
      : <button type="button" className="ghost" {...busyButton(pending, !!blocked || !canRecord)} title={blocked ?? 'Read the whole file and record its sha256'} onClick={onRecord}>{reading ? 'Reading…' : sum ? 'Recompute' : 'Record now'}</button>}
    {sum && <button type="button" className="ghost" title="Forget the recorded sha256" onClick={onClear}>Clear</button>}
  </div>;
}

type Props = {
  library: LibraryState | null; loadedPaths: string[]; companions: Companion[]; desktop: boolean; scrollPosition: React.RefObject<LibraryScroll>;
  onSave: (entries: LibraryEntry[], revision: number) => Promise<void>;
  onReload: () => Promise<void>; onOpenFile: () => Promise<void>; onOpenBackups: () => Promise<void>;
  onRecordSha: (path: string, slot: Slot) => Promise<void>; onBrowse: () => Promise<string | null>;
  onError: (message: string) => void;
};

type DraftEntry = LibraryEntry & { id: number };
type CompanionGroup = { all: string[]; folders: Map<string, string[]> };
type CompanionGroups = Record<Companion['kind'], CompanionGroup>;
const emptyRows: LibraryRow[] = [];

const LibraryCard = memo(function LibraryCard({
  entry, row, loaded, accounts, companions, desktop, edit, remove, pending, run, onBrowse, onRecordSha, onError,
}: {
  entry: DraftEntry; row?: LibraryRow; loaded: boolean; accounts: string[]; companions: CompanionGroups;
  desktop: boolean; edit: (id: number, change: Partial<LibraryEntry>) => void; remove: (id: number) => void;
  pending: string | null; run: (name: string, task: () => Promise<void>) => Promise<void>;
} & Pick<Props, 'onBrowse' | 'onRecordSha' | 'onError'>) {
      const account = entry.account || authorFromPath(entry.path);
      return <section className="panel library-entry">
        <div className="library-head">
          <AuthorLogo author={account || null} size={36}/>
          <input className="library-name" aria-label={`Name of ${stemOf(entry.path)}`} title="The model's API ID and display name; requests send it as model, exact case" value={entry.name} maxLength={64} onChange={e => edit(entry.id, { name: e.target.value })}/>
          <span className="library-chips">
            {loaded && <span className="chip success"><span className="status-dot green"/>Loaded</span>}
            {row?.status.model.missing && <span className="chip warning" title="No file at this path. Fix the path, or remove the entry.">File missing</span>}
            {(row?.status.mtp.missing || row?.status.mmproj.missing) && <span className="chip warning" title="A head or projector listed for this model is missing. Fix its path, or clear it to look in the model's folder.">Companion missing</span>}
            {row && Object.values(row.status).some(s => s.hashing) && <span className="chip"><RefreshCw className="spin" size={12}/>Checking sha256</span>}
            {row && Object.values(row.status).some(s => s.mismatch) && <span className="chip warning" title="A file no longer matches its recorded sha256">sha256 changed</span>}
          </span>
          <button className="icon-button" aria-label={`Remove ${entry.name} from models.yml`} title="Remove from models.yml (the file stays on disk)" onClick={() => remove(entry.id)}><Trash2 size={15}/></button>
        </div>
        <div className="library-fields">
          <label>Aliases<AliasInput label={`Aliases of ${entry.name}`} aliases={entry.aliases} onChange={aliases => edit(entry.id, { aliases })}/></label>
          <label>File<div className="file-input-row"><input aria-label={`Path of ${entry.name}`} value={entry.path} onChange={e => edit(entry.id, { path: e.target.value })}/><button type="button" className="secondary" disabled={!desktop} onClick={() => void run('browse', async () => { const path = await onBrowse(); if (path) edit(entry.id, { path, account: entry.account || authorFromPath(path) || '' }); })}><FolderOpen size={14}/>Browse</button></div></label>
          <label>Hugging Face account<ComboInput label={`Account of ${entry.name}`} options={accounts} placeholder={authorFromPath(entry.path) ?? 'For the avatar, such as LiquidAI'} value={entry.account} maxLength={96} onChange={account => edit(entry.id, { account })}/></label>
          <div className="library-sha"><span className="field-label">sha256</span>
            <ShaRow label={entry.name} sum={entry.sha256} status={row?.status.model} canRecord={!!row && !row.status.model.missing} blocked={!desktop ? 'Needs the desktop app' : null} pending={pending === `sha:${entry.id}:model`} onRecord={() => void run(`sha:${entry.id}:model`, () => onRecordSha(entry.path, 'model'))} onClear={() => edit(entry.id, { sha256: '' })} onError={onError}/>
          </div>
          {(['mtp', 'mmproj'] as const).map(slot => {
            const listed = row?.[slot] ?? '';
            const options = () => {
              const folder = folderOf(entry.path);
              const { all, folders } = companions[slot];
              const nearby = folders.get(folder);
              return nearby ? [...nearby, ...all.filter(path => folderOf(path) !== folder)] : all;
            };
            const title = slot === 'mtp' ? 'MTP head' : 'Projector (mmproj)';
            return <div className="library-companion" key={slot}>
              <label>{title}<div className="file-input-row"><ComboInput label={`${title} of ${entry.name}`} options={options} placeholder="Empty: look in the model's folder" value={entry[slot]} onChange={value => edit(entry.id, { [slot]: value })}/><button type="button" className="secondary" disabled={!desktop} onClick={() => void run('browse', async () => { const path = await onBrowse(); if (path) edit(entry.id, { [slot]: path }); })}><FolderOpen size={14}/>Browse</button></div></label>
              <Reveal open={entry[slot]}>{entry[slot] && <><span className="field-label">{title} sha256</span>
                <ShaRow label={`${entry.name} ${title}`} sum={entry[SUM[slot]]} status={row?.status[slot]} canRecord={!!listed && listed === entry[slot] && !row?.status[slot].missing} blocked={!desktop ? 'Needs the desktop app' : null} pending={pending === `sha:${entry.id}:${slot}`} onRecord={() => void run(`sha:${entry.id}:${slot}`, () => onRecordSha(entry.path, slot))} onClear={() => edit(entry.id, { [SUM[slot]]: '' })} onError={onError}/></>}</Reveal>
            </div>;
          })}
        </div>
      </section>;
});

// The Library tab: every model in models.yml, editable field by field. Saving
// writes the file (backing up the old one); edits made in a text editor show
// up here as soon as they are saved.
export const Library = memo(function Library({ library, loadedPaths, companions, desktop, scrollPosition, onSave, onReload, onOpenFile, onOpenBackups, onRecordSha, onBrowse, onError }: Props) {
  const [draft, setDraft] = useState<DraftEntry[]>([]);
  const nextId = useRef(0);
  const resetDraft = (entries: LibraryEntry[]) => {
    const previous = new Map(draft.map(entry => [entry.path, entry]));
    setDraft(entries.map(entry => {
      const existing = previous.get(entry.path);
      const next = strip(entry);
      if (existing && equalJson(strip(existing), next)) return existing;
      return { ...next, id: existing?.id ?? ++nextId.current };
    }));
  };
  const [base, setBase] = useState<{ revision: number; entries: LibraryEntry[] } | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const visiblePending = useDelayedPending(pending);
  const rows = library?.entries ?? emptyRows;
  const byPath = useMemo(() => new Map(rows.map(row => [row.path, row])), [rows]);
  const loaded = useMemo(() => new Set(loadedPaths), [loadedPaths]);
  const groups = useMemo(() => {
    const result: CompanionGroups = { mtp: { all: [], folders: new Map() }, mmproj: { all: [], folders: new Map() } };
    for (const { path, kind } of companions) {
      const group = result[kind];
      const folder = folderOf(path);
      group.all.push(path);
      const nearby = group.folders.get(folder);
      if (nearby) nearby.push(path);
      else group.folders.set(folder, [path]);
    }
    return result;
  }, [companions]);
  const dirty = !!base && !same(draft, base.entries);
  // Take the file's newest contents whenever there is nothing unsaved to lose.
  const latest = useMemo(() => JSON.stringify(rows.map(strip)), [rows]);
  useEffect(() => {
    if (!library || dirty || (base && base.revision === library.revision && same(base.entries, library.entries))) return;
    resetDraft(library.entries); setBase({ revision: library.revision, entries: library.entries.map(strip) });
  }, [latest, library?.revision, dirty]);
  const stale = !!library && !!base && dirty && base.revision !== library.revision;
  // Suggestions come from the saved file, so a half-typed value is never offered back.
  const accountSuggestions = JSON.stringify([...new Set([...rows.map(e => e.account), ...draft.map(e => authorFromPath(e.path) ?? '')].filter(Boolean))].sort());
  const accounts = useMemo(() => JSON.parse(accountSuggestions) as string[], [accountSuggestions]);
  const edit = useCallback((id: number, change: Partial<LibraryEntry>) => setDraft(entries =>
    entries.map(entry => entry.id === id ? { ...entry, ...change } : entry)), []);
  const remove = useCallback((id: number) => setDraft(entries => entries.filter(entry => entry.id !== id)), []);
  const run = useCallback(async (name: string, task: () => Promise<void>) => {
    setPending(name);
    try { await task(); } catch (e) { onError(errorText(e)); } finally { setPending(null); }
  }, [onError]);
  async function addModel() {
    const path = await onBrowse();
    if (!path) return;
    if (draft.some(e => e.path === path)) { onError(`${stemOf(path)} is already in models.yml.`); return; }
    const id = ++nextId.current;
    setDraft(entries => [...entries, { id, name: uniqueName(stemOf(path), entries), aliases: [], path, account: authorFromPath(path) ?? '', sha256: '', mtp: '', mtp_sha256: '', mmproj: '', mmproj_sha256: '' }]);
  }
  function discard() {
    if (!library) return;
    resetDraft(library.entries); setBase({ revision: library.revision, entries: library.entries.map(strip) });
  }
  // After a save the backend's tidied copy (trimmed names and so on) replaces the draft.
  // A sha256 recorded in the background while you edited is kept, unless you changed that field yourself.
  const save = () => run('save', async () => {
    if (!base) return;
    const original = new Map(base.entries.map(entry => [entry.path, entry]));
    const merged = draft.map(draftEntry => {
      const entry = strip(draftEntry);
      const before = original.get(entry.path);
      const now = byPath.get(entry.path);
      if (!before || !now) return entry;
      const kept = { ...entry };
      for (const slot of ['model', 'mtp', 'mmproj'] as Slot[]) {
        const field = SUM[slot];
        const sameFile = slot === 'model' || (before[slot] === entry[slot] && now[slot] === entry[slot]);
        if (sameFile && before[field] === entry[field]) kept[field] = now[field];
      }
      return kept;
    });
    await onSave(merged, base.revision); setBase(null);
  });

  return <div className="library">
    <section className="panel library-toolbar">
      <div className="library-file"><FileCog size={17}/><span className="identity-line"><code title={library?.file}>{library?.file ?? 'models.yml'}</code>{library?.file && <CopyButton value={library.file} label="Copy models.yml path" size={12} onError={onError}/>}</span></div>
      <div className="library-actions">
        <button className="secondary compact" {...busyButton(!!pending, !desktop, pending === 'add')} onClick={() => void run('add', addModel)}><Plus size={13}/>Add model</button>
        <button className="secondary compact" {...busyButton(!!pending, !desktop, pending === 'open')} title="Open models.yml in your default editor" onClick={() => void run('open', onOpenFile)}><FolderOpen size={13}/>Edit models.yml</button>
        <button className="secondary compact" {...busyButton(!!pending, !desktop, pending === 'backups')} title="A copy is kept before every change; the newest 20 stay" onClick={() => void run('backups', onOpenBackups)}><ArchiveRestore size={13}/>Backups</button>
        <button className="ghost" {...busyButton(!!pending, !desktop || dirty, pending === 'reload')} title={dirty ? 'Save or discard your changes first' : 'Read models.yml again'} onClick={() => void run('reload', onReload)}><RefreshCw size={13} className={visiblePending === 'reload' ? 'spin' : ''}/>Reload</button>
      </div>
    </section>
    <Reveal open={library?.error} className="library-spaced">{library?.error && <div className="load-error" role="alert"><AlertCircle size={18}/><div><strong>models.yml has an error</strong><pre>{library.error}</pre></div></div>}</Reveal>
    <Reveal open={stale} className="library-spaced">{stale && <div className="notice error" role="alert"><span className="notice-text"><AlertCircle size={16}/><span>models.yml changed on disk while you were editing. Discard your changes to see it, or copy them over afterwards.</span></span></div>}</Reveal>
    <Reveal open={draft.length === 0} className="library-spaced">{draft.length === 0 && <div className="local-empty"><FileCog size={18}/><p>No models yet. Downloads and models found on this machine are added automatically, or click <strong>Add model</strong>.</p></div>}</Reveal>
    <LibraryDirty.Provider value={dirty}>
      <LibraryList scrollPosition={scrollPosition} ids={draft.map(entry => entry.id)} renderRow={index => {
        const entry = draft[index];
        return <LibraryCard entry={entry} row={byPath.get(entry.path)} loaded={loaded.has(entry.path)}
          accounts={accounts} companions={groups} desktop={desktop} edit={edit} remove={remove}
          pending={pending?.startsWith('sha:' + entry.id + ':') ? pending : null}
          run={run} onBrowse={onBrowse} onRecordSha={onRecordSha} onError={onError}/>;
      }}/>
    </LibraryDirty.Provider>
    <div className={`save-bar ${dirty ? 'dirty' : ''}`}><Reveal open className="hint" swapKey={String(dirty)}>{dirty ? 'Unsaved changes to models.yml. Saving backs up the current file first.' : 'models.yml is up to date. Edits made in a text editor appear here when saved.'}</Reveal><div className="save-actions"><Reveal open={dirty} className="inline-reveal">{dirty && <button className="ghost" onClick={discard}>Discard</button>}</Reveal><button className={dirty ? 'primary' : 'secondary'} {...busyButton(!!pending, !dirty || !desktop, pending === 'save')} onClick={save}><Save size={14}/>Save models.yml</button></div></div>
  </div>;
});

// The Settings tab's sha256 choice.
export const SHA_CHECKS: { value: string; label: string; detail: string }[] = [
  { value: 'once', label: 'Hash once', detail: 'Each file is hashed once in the background. Before a load the hash is read again if the file changed. A mismatch stops the load until you accept the new sha256 in Library.' },
  { value: 'every_load', label: 'Every load', detail: 'The whole file is hashed before every load. Safest, but a large model takes 10–30 seconds longer to load.' },
  { value: 'warn', label: 'Warn only', detail: 'Hashed as with Hash once, but a mismatch only adds a warning to the log and the model loads anyway.' },
  { value: 'off', label: 'Off', detail: 'No hashing. sha256 values already in models.yml are kept but not checked.' },
];
