import React, { useId, useState } from 'react';
import { CircleAlert, CircleCheck, CircleHelp, Info, LoaderCircle, TriangleAlert } from 'lucide-react';

// The shared settings layout used by the Loading tab load panel, the Runtime tab
// and the Settings tab. A section groups rows under a title; a row has its
// label and one short description on the left and its control in a fixed
// control column on the right, so every control lines up on one edge. Extra
// explanation sits behind a "more" button, and live state is a StatusNote,
// never plain help text.

export type Tone = 'ok' | 'info' | 'warn' | 'error' | 'pending';
const ICONS = { ok: CircleCheck, info: Info, warn: TriangleAlert, error: CircleAlert, pending: LoaderCircle };

// Live state (found, missing, checking, refused), coloured by what it means.
export function StatusNote({ tone, children }: { tone: Tone; children: React.ReactNode }) {
  const Icon = ICONS[tone];
  return <p className={`status-note ${tone}`}><Icon size={14} aria-hidden="true" className={tone === 'pending' ? 'spin' : undefined}/><span>{children}</span></p>;
}

// One short line of description, with the rest of the explanation behind a
// button that shows it inline (keyboard reachable, not a hover tooltip).
export function Help({ text, more, topic }: { text?: React.ReactNode; more?: React.ReactNode; topic: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  if (!text && !more) return null;
  return <div className="setting-help">
    <p className="setting-description">{text}{more && <button type="button" className="more-button" aria-expanded={open} aria-controls={id} aria-label={`${open ? 'Hide' : 'Show'} more about ${topic}`} title={open ? 'Show less' : 'Show more'} onClick={() => setOpen(!open)}><CircleHelp size={14}/></button>}</p>
    {more && <p className="setting-more" id={id} hidden={!open}>{more}</p>}
  </div>;
}

export function SettingSection({ title, description, more, aside, children, className }: { title: string; description?: React.ReactNode; more?: React.ReactNode; aside?: React.ReactNode; children: React.ReactNode; className?: string }) {
  const id = useId();
  return <section className={`setting-section ${className ?? ''}`} aria-labelledby={id}>
    <header className="setting-section-head">
      <div className="setting-section-title"><h3 id={id}>{title}</h3><Help text={description} more={more} topic={title}/></div>
      {aside && <div className="setting-section-aside">{aside}</div>}
    </header>
    <div className="setting-rows">{children}</div>
  </section>;
}

// `control` goes in the control column; `wide` gives it more room (Device).
// `children` span the full row under the label (paths, file pickers).
// `htmlFor` makes the label a <label> for a text input in the row.
export function SettingRow({ label, htmlFor, description, more, status, control, wide, children, className }: { label: React.ReactNode; htmlFor?: string; description?: React.ReactNode; more?: React.ReactNode; status?: React.ReactNode; control?: React.ReactNode; wide?: boolean; children?: React.ReactNode; className?: string }) {
  const topic = typeof label === 'string' ? label : 'this setting';
  return <div className={`setting-row ${wide ? 'wide' : ''} ${control ? '' : 'no-control'} ${className ?? ''}`}>
    <div className="setting-text">
      {htmlFor ? <label className="setting-label" htmlFor={htmlFor}>{label}</label> : <span className="setting-label">{label}</span>}
      <Help text={description} more={more} topic={topic}/>
    </div>
    {control && <div className="setting-control">{control}</div>}
    {children && <div className="setting-full">{children}</div>}
    {status && <div className="setting-status">{status}</div>}
  </div>;
}
