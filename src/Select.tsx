import React, { useEffect, useId, useRef, useState } from 'react';
import { Check, ChevronDown } from 'lucide-react';

// The native <select> popup is drawn by GTK in the desktop build and ignores
// the app's styles, so this draws its own listbox.
export function Select({ value, options, onChange, disabled, label }: { value: string; options: { value: string; label: string }[]; onChange: (value: string) => void; disabled?: boolean; label: string }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const id = useId();
  const selected = options.find(option => option.value === value);

  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, [open]);
  useEffect(() => { if (open) list.current?.children[active]?.scrollIntoView({ block: 'nearest' }); }, [open, active]);

  function show() { setActive(Math.max(0, options.findIndex(option => option.value === value))); setOpen(true); }
  function choose(index: number) { const option = options[index]; if (option) onChange(option.value); setOpen(false); }
  function onKeyDown(event: React.KeyboardEvent) {
    if (disabled) return;
    const keys: Record<string, () => void> = {
      ArrowDown: () => open ? setActive(i => Math.min(options.length - 1, i + 1)) : show(),
      ArrowUp: () => open ? setActive(i => Math.max(0, i - 1)) : show(),
      Home: () => open && setActive(0),
      End: () => open && setActive(options.length - 1),
      Enter: () => open ? choose(active) : show(),
      ' ': () => open ? choose(active) : show(),
      Escape: () => setOpen(false),
      Tab: () => setOpen(false),
    };
    const handler = keys[event.key];
    if (!handler) return;
    if (event.key !== 'Tab') event.preventDefault();
    handler();
  }

  return <div className={`select ${open ? 'open' : ''}`} ref={root}>
    <button type="button" className="select-trigger" role="combobox" aria-label={label} aria-haspopup="listbox" aria-expanded={open} aria-controls={id} aria-activedescendant={open ? `${id}-${active}` : undefined} disabled={disabled} onClick={() => open ? setOpen(false) : show()} onKeyDown={onKeyDown}>
      <span>{selected?.label ?? value}</span><ChevronDown size={15}/>
    </button>
    {open && <ul className="select-menu" role="listbox" id={id} ref={list} aria-label={label}>
      {options.map((option, index) => <li key={option.value} id={`${id}-${index}`} role="option" aria-selected={option.value === value} className={index === active ? 'active' : ''} onPointerEnter={() => setActive(index)} onClick={() => choose(index)}>
        <span>{option.label}</span>{option.value === value && <Check size={14}/>}
      </li>)}
    </ul>}
  </div>;
}
