import React, { useEffect, useRef, useState } from 'react';
import { ClipboardPaste, Copy, Scissors, TextSelect } from 'lucide-react';

type Target = { element: HTMLElement | null; editable: boolean; text: string; x: number; y: number };
type Item = { label: string; icon: typeof Copy; shortcut: string; disabled: boolean; run: () => Promise<void> | void };

const isMac = navigator.platform.toLowerCase().includes('mac');
const key = (letter: string) => `${isMac ? '⌘' : 'Ctrl+'}${letter}`;

function editableFrom(node: EventTarget | null): HTMLElement | null {
  if (!(node instanceof Element)) return null;
  const field = node.closest('input, textarea, [contenteditable="true"]');
  if (!(field instanceof HTMLElement)) return null;
  if ((field instanceof HTMLInputElement || field instanceof HTMLTextAreaElement) && (field.readOnly || field.disabled)) return null;
  if (field instanceof HTMLInputElement && !['text', 'search', 'url', 'number', 'email', 'password', ''].includes(field.type)) return null;
  return field;
}

function selectedText(element: HTMLElement | null) {
  if (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement) {
    if (element instanceof HTMLInputElement && element.type === 'password') return '';
    return element.value.slice(element.selectionStart ?? 0, element.selectionEnd ?? 0);
  }
  return window.getSelection()?.toString() ?? '';
}

// The webview's own menu offers Back and Reload, which mean nothing here, and is
// drawn in the system theme. This one covers the edit actions instead. Hold
// Shift while right-clicking to get the native menu (for Inspect Element).
export function ContextMenu() {
  const [target, setTarget] = useState<Target | null>(null);
  const [active, setActive] = useState(0);
  const menu = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const open = (event: MouseEvent) => {
      if (event.shiftKey || event.defaultPrevented) return;
      event.preventDefault();
      if (event.target instanceof Element && event.target.closest('dialog[open]')) { setTarget(null); return; }
      const element = editableFrom(event.target);
      const text = selectedText(element);
      if (!element && !text) { setTarget(null); return; }
      setTarget({ element, editable: !!element, text, x: event.clientX, y: event.clientY });
      setActive(-1);
    };
    window.addEventListener('contextmenu', open);
    return () => window.removeEventListener('contextmenu', open);
  }, []);

  useEffect(() => {
    if (!target) return;
    const close = (event: Event) => { if (!(event.target instanceof Node && menu.current?.contains(event.target))) setTarget(null); };
    const dismiss = () => setTarget(null);
    window.addEventListener('pointerdown', close, true);
    window.addEventListener('scroll', dismiss, true);
    window.addEventListener('resize', dismiss);
    window.addEventListener('blur', dismiss);
    return () => { window.removeEventListener('pointerdown', close, true); window.removeEventListener('scroll', dismiss, true); window.removeEventListener('resize', dismiss); window.removeEventListener('blur', dismiss); };
  }, [target]);

  useEffect(() => {
    if (!target || !menu.current) return;
    const { width, height } = menu.current.getBoundingClientRect();
    setPosition({ x: Math.max(8, Math.min(target.x, innerWidth - width - 8)), y: target.y + height + 8 > innerHeight ? Math.max(8, target.y - height) : target.y });
    menu.current.focus({ preventScroll: true });
  }, [target]);

  if (!target) return null;
  const { element, editable, text } = target;
  const finish = () => { setTarget(null); element?.focus(); };
  const items: Item[] = [
    ...(editable ? [{ label: 'Cut', icon: Scissors, shortcut: key('X'), disabled: !text, run: async () => { await navigator.clipboard.writeText(text); finish(); document.execCommand('delete'); } }] : []),
    { label: 'Copy', icon: Copy, shortcut: key('C'), disabled: !text, run: async () => { await navigator.clipboard.writeText(text); finish(); } },
    ...(editable ? [
      { label: 'Paste', icon: ClipboardPaste, shortcut: key('V'), disabled: false, run: async () => {
        finish();
        try { const pasted = await navigator.clipboard.readText(); if (pasted) document.execCommand('insertText', false, pasted); }
        catch { document.execCommand('paste'); }
      } },
      { label: 'Select all', icon: TextSelect, shortcut: key('A'), disabled: false, run: () => { finish(); document.execCommand('selectAll'); } },
    ] : []),
  ];
  const enabled = items.map((item, index) => item.disabled ? -1 : index).filter(index => index >= 0);
  function onKeyDown(event: React.KeyboardEvent) {
    const at = enabled.indexOf(active);
    if (event.key === 'ArrowDown') setActive(enabled[(at + 1) % enabled.length]);
    else if (event.key === 'ArrowUp') setActive(enabled[(at - 1 + enabled.length) % enabled.length]);
    else if (event.key === 'Enter' && active >= 0) void items[active].run();
    else if (event.key === 'Escape') finish();
    else return;
    event.preventDefault();
  }

  return <div className="context-menu" role="menu" ref={menu} tabIndex={-1} style={{ left: position.x, top: position.y }} onKeyDown={onKeyDown} onMouseDown={event => event.preventDefault()} onContextMenu={event => event.preventDefault()}>
    {items.map((item, index) => { const Icon = item.icon; return <button key={item.label} role="menuitem" className={index === active ? 'active' : ''} disabled={item.disabled} onPointerEnter={() => setActive(item.disabled ? -1 : index)} onClick={() => void item.run()}>
      <Icon size={14}/><span>{item.label}</span><kbd>{item.shortcut}</kbd>
    </button>; })}
  </div>;
}
