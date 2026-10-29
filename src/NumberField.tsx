import React, { useEffect, useState } from 'react';
import { Minus, Plus } from 'lucide-react';

// WebKitGTK draws the native number spinner as two tiny system arrows, so the
// spinner is hidden in CSS and replaced by these buttons. Arrow keys still step.
export function NumberField({ value, onChange, min, max, step = 1 }: { value: number; onChange: (value: number) => void; min?: number; max?: number; step?: number }) {
  // The text is kept separately so the field can be empty while retyping.
  const [draft, setDraft] = useState(String(value));
  useEffect(() => { if (Number(draft) !== value || draft === '') setDraft(String(value)); }, [value]);
  const clamp = (next: number) => Math.min(max ?? Infinity, Math.max(min ?? -Infinity, next));
  return <span className="number-field">
    <input type="number" min={min} max={max} step={step} value={draft} onChange={e => { setDraft(e.target.value); if (e.target.value !== '' && Number.isFinite(e.target.valueAsNumber)) onChange(e.target.valueAsNumber); }} onBlur={() => setDraft(String(value))}/>
    <button type="button" className="icon-button" aria-label="Decrease" tabIndex={-1} disabled={min !== undefined && value <= min} onClick={() => onChange(clamp(value - step))}><Minus size={14}/></button>
    <button type="button" className="icon-button" aria-label="Increase" tabIndex={-1} disabled={max !== undefined && value >= max} onClick={() => onChange(clamp(value + step))}><Plus size={14}/></button>
  </span>;
}
