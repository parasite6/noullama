import React from 'react';
import { Toggle } from './Toggle';

// One switch shared by the Playground and Local API pages: the backend keeps a
// single history that every decision, from either place, reads and adds to.
export function KeepContext({ on, entries, lastFlushed, disabled, onToggle, onClear }: { on: boolean; entries: number; lastFlushed: number; disabled?: boolean; onToggle: (on: boolean) => void; onClear: () => void }) {
  const status = !on
    ? 'Each decision starts fresh.'
    : entries === 0
      ? 'Earlier decisions will be added to the next requests.'
      : `${entries} earlier ${entries === 1 ? 'decision' : 'decisions'} in context${lastFlushed ? `; the last run flushed the ${lastFlushed === 1 ? 'oldest one' : `${lastFlushed} oldest`} to fit` : ''}.`;
  return <div className="keep-context">
    <div className="keep-context-row">
      <Toggle label="Keep context" title="Add earlier decisions to each request's state. Applies to the Playground and every local API call. The oldest are flushed when the context window is full." checked={on} disabled={disabled} onChange={onToggle}/>
      {on && entries > 0 && <button type="button" className="text-button" onClick={onClear}>Clear</button>}
    </div>
    <p className="keep-context-status">{status}</p>
  </div>;
}
