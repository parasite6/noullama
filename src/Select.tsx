import { busyButton } from './busy-button';
import React, { useEffect, useId, useRef, useState } from 'react';
import { CheckIcon as Check } from '@phosphor-icons/react/dist/icons/Check';
import { CaretDownIcon as ChevronDown } from '@phosphor-icons/react/dist/icons/CaretDown';

// The native <select> popup is drawn by GTK in the desktop build and ignores
// the app's styles, so this draws its own listbox.
export function Select({ value, options, onChange, disabled, busy = false, label }: { value: string; options: { value: string; label: string; detail?: string }[]; onChange: (value: string) => void; disabled?: boolean; busy?: boolean; label: string }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const id = useId();
  const selected = options.find(option => option.value === value);

  useEffect(() => {
    if (busy && open) { setOpen(false); return; }
    if (!open) return;
    const close = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, [open, busy]);
  useEffect(() => { if (open) list.current?.children[active]?.scrollIntoView({ block: 'nearest' }); }, [open, active]);

  function show() { setActive(Math.max(0, options.findIndex(option => option.value === value))); setOpen(true); }
  function choose(index: number) { const option = options[index]; if (option && !busy && !disabled) onChange(option.value); setOpen(false); }
  function onKeyDown(event: React.KeyboardEvent) {
    if (disabled || busy) return;
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
    <button type="button" className="select-trigger" role="combobox" aria-label={label} aria-haspopup="listbox" aria-expanded={open} aria-controls={id} aria-activedescendant={open ? `${id}-${active}` : undefined} {...busyButton(busy, disabled, false)} onClick={() => open ? setOpen(false) : show()} onKeyDown={onKeyDown}>
      <span className="select-value" title={selected ? [selected.label, selected.detail].filter(Boolean).join(' · ') : undefined}><span className="select-label">{selected?.label ?? value}</span>{selected?.detail && <span className="select-detail">{selected.detail}</span>}</span><ChevronDown size={15}/>
    </button>
    {open && <ul className="select-menu" role="listbox" id={id} ref={list} aria-label={label}>
      {options.map((option, index) => <li key={option.value} id={`${id}-${index}`} role="option" aria-selected={option.value === value} className={index === active ? 'active' : ''} onPointerEnter={() => setActive(index)} onClick={() => choose(index)}>
        <span className="select-value"><span className="select-label">{option.label}</span>{option.detail && <span className="select-detail">{option.detail}</span>}</span>
        {option.value === value && <Check className="select-check" size={15} weight="bold" aria-hidden="true"/>}
      </li>)}
    </ul>}
  </div>;
}

// A text field with suggestions. Like Select, it draws its own list because
// the native <datalist> popup is drawn by GTK. Any text can be typed.
export function ComboInput({ value, options, onChange, label, placeholder, maxLength, className }: { value: string; options: string[] | (() => string[]); onChange: (value: string) => void; label: string; placeholder?: string; maxLength?: number; className?: string }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  // Opening shows every option; typing narrows the list to what matches.
  const [typed, setTyped] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const id = useId();
  const needle = typed ? value.trim().toLowerCase() : '';
  const shown: string[] = [];
  if (open) {
    const seen = new Set<string>();
    for (const option of typeof options === 'function' ? options() : options) {
      if (!option || seen.has(option) || (needle && !option.toLowerCase().includes(needle))) continue;
      seen.add(option);
      shown.push(option);
      if (shown.length === 50) break;
    }
  }
  function show() { setTyped(false); setActive(-1); setOpen(true); }
  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, [open]);
  function choose(option: string | undefined) { if (option !== undefined) onChange(option); setOpen(false); setActive(-1); }
  function onKeyDown(event: React.KeyboardEvent) {
    if (event.key === 'ArrowDown') { event.preventDefault(); if (!open) { show(); setActive(0); } else setActive(i => Math.min(shown.length - 1, i + 1)); }
    else if (event.key === 'ArrowUp') { event.preventDefault(); setActive(i => Math.max(0, i - 1)); }
    else if (event.key === 'Enter' && open && active >= 0) { event.preventDefault(); choose(shown[active]); }
    else if (event.key === 'Escape' || event.key === 'Tab') setOpen(false);
  }
  return <div className={`combo ${className ?? ''}`} ref={root}>
    <input role="combobox" aria-label={label} aria-autocomplete="list" aria-expanded={open && shown.length > 0} aria-controls={id} aria-activedescendant={open && active >= 0 && active < shown.length ? `${id}-${active}` : undefined}
      placeholder={placeholder} maxLength={maxLength} spellCheck={false} autoComplete="off" value={value} onFocus={show} onClick={() => { if (!open) show(); }} onChange={event => { onChange(event.target.value); setTyped(true); setOpen(true); setActive(-1); }} onKeyDown={onKeyDown}/>
    {open && shown.length > 0 && <ul className="select-menu" role="listbox" id={id} aria-label={label}>
      {shown.map((option, index) => <li key={option} id={`${id}-${index}`} role="option" aria-selected={index === active} data-current={option === value || undefined} className={index === active ? 'active' : ''} title={option} onPointerEnter={() => setActive(index)} onPointerDown={event => { event.preventDefault(); choose(option); }}>
        <span><bdi dir="ltr">{option}</bdi></span>
        {option === value && <Check className="select-check" size={15} weight="bold" aria-hidden="true"/>}
      </li>)}
    </ul>}
  </div>;
}
