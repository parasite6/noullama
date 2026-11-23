import React from 'react';
import { Toggle } from './Toggle';

// One loaded instance's Keep context switch, used on its Loading tab card and
// in the Playground rail (for the Playground's instance). Each instance keeps
// its own history, which every decision it makes, from either place, reads
// and adds to. `scope` names whose history the counts describe.
export function KeepContext({ on, entries, lastFlushed, scope, disabled, onToggle, onClear }: { on: boolean; entries: number; lastFlushed: number; scope?: string; disabled?: boolean; onToggle: (on: boolean) => void; onClear: () => void }) {
  const status = !on
    ? 'Each decision starts fresh.'
    : entries === 0
      ? 'Earlier decisions will be added to the next requests.'
      : `${entries} earlier ${entries === 1 ? 'decision' : 'decisions'} in context${scope ? ` ${scope}` : ''}${lastFlushed ? `; the last run flushed the ${lastFlushed === 1 ? 'oldest one' : `${lastFlushed} oldest`} to fit` : ''}.`;
  return <div className="keep-context">
    <div className="keep-context-row">
      <Toggle label="Keep context" title="Add this model's earlier decisions to each request's state. Its history is shared by the Playground and every local API call it answers; other loaded models keep their own. The oldest are flushed when the context window is full." checked={on} disabled={disabled} onChange={onToggle}/>
      {on && entries > 0 && <button type="button" className="text-button" disabled={disabled} onClick={onClear}>Clear</button>}
    </div>
    <p className="keep-context-status">{status}</p>
  </div>;
}
