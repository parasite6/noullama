import { SelectionIndicator } from './SelectionIndicator';
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { CodeIcon as Code2 } from '@phosphor-icons/react/dist/icons/Code';
import { CpuIcon as Cpu } from '@phosphor-icons/react/dist/icons/Cpu';
import { FlaskIcon as FlaskConical } from '@phosphor-icons/react/dist/icons/Flask';
import { StackIcon as Layers3 } from '@phosphor-icons/react/dist/icons/Stack';
import { BooksIcon as Library } from '@phosphor-icons/react/dist/icons/Books';
import { GraphIcon as Network } from '@phosphor-icons/react/dist/icons/Graph';
import { PlugChargingIcon as PlugZap } from '@phosphor-icons/react/dist/icons/PlugCharging';
import { SlidersHorizontalIcon as Settings2 } from '@phosphor-icons/react/dist/icons/SlidersHorizontal';
import { ContextMenu } from './ContextMenu';
import { moveTab, sanitizeTabOrder } from './ui-state';

const tabs = {
  playground: [FlaskConical, 'Playground'],
  models: [Layers3, 'Models'],
  loading: [PlugZap, 'Loading'],
  hub: [Network, 'Hub'],
  runtime: [Cpu, 'Runtime'],
  api: [Code2, 'Local API'],
  library: [Library, 'Library'],
  settings: [Settings2, 'Settings'],
} as const;
type Tab = keyof typeof tabs;
const defaults = Object.keys(tabs) as Tab[];
const storageKey = 'noullama.sidebarTabOrder';

type Gesture = {
  pointerId: number;
  from: number;
  startY: number;
  slots: { top: number; bottom: number; center: number }[];
  canceled: boolean;
};
type Preview = { from: number; to: number; delta: number; tops: number[] };

function readOrder(): readonly Tab[] {
  try {
    return sanitizeTabOrder(JSON.parse(localStorage.getItem(storageKey) ?? 'null'), defaults);
  } catch {
    return defaults;
  }
}

export function SidebarTabs({ tab, onSelect }: { tab: string; onSelect: (tab: Tab) => void }) {
  const [order, setOrder] = useState(readOrder);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [announcement, setAnnouncement] = useState('');
  const nav = useRef<HTMLElement>(null);
  const gesture = useRef<Gesture | null>(null);
  const suppressClick = useRef(false);
  const restoreFocus = useRef<Tab | null>(null);
  const settleFrom = useRef<Map<HTMLElement, number> | null>(null);
  const settling = useRef(new Map<HTMLElement, Animation>());

  const captureReorder = useCallback(() => {
    const container = nav.current;
    if (!container || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    settleFrom.current = new Map(Array.from(
      container.querySelectorAll<HTMLElement>('.nav-item, .selection-indicator'),
      element => [element, element.getBoundingClientRect().top],
    ));
    container.dataset.settling = '';
  }, []);

  useLayoutEffect(() => {
    const from = settleFrom.current;
    const container = nav.current;
    if (!from || !container) return;
    settleFrom.current = null;
    for (const animation of settling.current.values()) animation.cancel();
    settling.current.clear();
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      delete container.dataset.settling;
      return;
    }

    // The indicator's layout effect already measured the selected row's new slot.
    const destinations = [...from].map(([element, top]) =>
      [element, top - element.getBoundingClientRect().top] as const);
    const easing = getComputedStyle(container).getPropertyValue('--ease-reorder').trim();
    for (const [element, delta] of destinations) {
      if (Math.abs(delta) < .01) continue;
      const animation = element.animate(
        [{ translate: `0 ${delta}px` }, { translate: '0 0' }],
        { duration: 200, easing, fill: 'backwards' },
      );
      settling.current.set(element, animation);
      animation.onfinish = () => {
        if (settling.current.get(element) === animation) settling.current.delete(element);
      };
    }
    delete container.dataset.settling;
  });

  useLayoutEffect(() => {
    if (restoreFocus.current) {
      nav.current?.querySelector<HTMLButtonElement>(`[data-tab="${restoreFocus.current}"]`)?.focus({ preventScroll: true });
      restoreFocus.current = null;
    }
  }, [order]);

  useEffect(() => {
    function cancel() {
      if (!gesture.current || gesture.current.canceled) return;
      if (suppressClick.current) captureReorder();
      gesture.current.canceled = true;
      suppressClick.current = true;
      setPreview(null);
    }
    function onKeyDown(event: globalThis.KeyboardEvent) {
      if (event.key !== 'Escape' || !gesture.current || !suppressClick.current) return;
      event.preventDefault();
      cancel();
    }
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('resize', cancel);
    window.addEventListener('blur', cancel);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('resize', cancel);
      window.removeEventListener('blur', cancel);
    };
  }, [captureReorder]);

  function commit(next: readonly Tab[], message: string) {
    setOrder(next);
    setAnnouncement(message);
    try {
      localStorage.setItem(storageKey, JSON.stringify(next));
    } catch {
      // Keep the in-memory order when storage is unavailable.
    }
  }

  function move(from: number, to: number, keepFocus = false) {
    const next = moveTab(order, from, to);
    if (next === order) return;
    const id = order[from];
    if (keepFocus) {
      captureReorder();
      restoreFocus.current = id;
    }
    commit(next, `${tabs[id][1]} moved to position ${to + 1} of ${order.length}`);
  }

  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>, from: number) {
    if (!event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
    event.preventDefault();
    if (!gesture.current) move(from, from + (event.key === 'ArrowUp' ? -1 : 1), true);
  }

  function onPointerDown(event: PointerEvent<HTMLButtonElement>, from: number) {
    if (!event.isPrimary || event.button !== 0 || gesture.current || !nav.current) return;
    const listTop = nav.current.getBoundingClientRect().top;
    const slots = Array.from(nav.current.querySelectorAll('button'), button => {
      const top = listTop + button.offsetTop;
      const bottom = top + button.offsetHeight;
      return { top, bottom, center: (top + bottom) / 2 };
    });
    event.currentTarget.setPointerCapture(event.pointerId);
    suppressClick.current = false;
    gesture.current = { pointerId: event.pointerId, from, startY: event.clientY, slots, canceled: false };
  }

  function updateDrag(event: PointerEvent<HTMLButtonElement>): Preview | null {
    const drag = gesture.current;
    if (!drag || drag.pointerId !== event.pointerId || drag.canceled) return null;
    const dy = event.clientY - drag.startY;
    if (!suppressClick.current && Math.abs(dy) < 5) return null;
    suppressClick.current = true;
    const { slots, from } = drag;
    const slot = slots[from];
    const delta = Math.max(slots[0].top - slot.top, Math.min(dy, slots[slots.length - 1].bottom - slot.bottom));
    const center = slot.center + delta;
    const to = slots.filter((other, index) => index !== from && (other.center < center || (delta > 0 && other.center === center))).length;
    return { from, to, delta, tops: slots.map(other => other.top) };
  }

  function finishPointer(event: PointerEvent<HTMLButtonElement>, drop: boolean) {
    if (gesture.current?.pointerId !== event.pointerId) return;
    const final = drop ? updateDrag(event) : null;
    if (drop && !suppressClick.current) {
      // Capture retargets pointerup, so hit-test where the pointer actually landed.
      const releasedOver = document.elementFromPoint(event.clientX, event.clientY);
      suppressClick.current = !event.currentTarget.contains(releasedOver);
    }
    if (preview || (final && final.from !== final.to)) captureReorder();
    gesture.current = null;
    setPreview(null);
    if (final) move(final.from, final.to);
  }

  const previewOrder = preview ? moveTab(order, preview.from, preview.to) : order;
  const activeIndex = order.findIndex(id => id === tab);
  function offsetFor(id: Tab, index: number) {
    if (!preview) return 0;
    return preview.from === index ? preview.delta : preview.tops[previewOrder.indexOf(id)] - preview.tops[index];
  }

  return <>
    <nav ref={nav} className={preview ? 'sidebar-tabs reordering' : 'sidebar-tabs'} aria-label="Pages">
      <SelectionIndicator selected='button[aria-current="page"]'
        offsetY={activeIndex < 0 ? 0 : offsetFor(order[activeIndex], activeIndex)}
        dragging={!!preview && preview.from === activeIndex}/>
      {order.map((id, index) => {
        const [Icon, label] = tabs[id];
        const dragging = preview?.from === index;
        const offset = offsetFor(id, index);
        return <button
          key={id}
          type="button"
          data-tab={id}
          className={`nav-item${tab === id ? ' active' : ''}${dragging ? ' dragging' : ''}`}
          style={offset ? { transform: `translateY(${offset}px)` } : undefined}
          aria-current={tab === id ? 'page' : undefined}
          aria-label={label}
          aria-keyshortcuts="Alt+ArrowUp Alt+ArrowDown"
          title={label}
          onClick={event => {
            if (event.detail !== 0 && suppressClick.current) {
              suppressClick.current = false;
              return;
            }
            onSelect(id);
          }}
          onKeyDown={event => onKeyDown(event, index)}
          onPointerDown={event => onPointerDown(event, index)}
          onPointerMove={event => {
            const next = updateDrag(event);
            if (next) setPreview(next);
          }}
          onPointerUp={event => finishPointer(event, true)}
          onPointerCancel={event => finishPointer(event, false)}
          onLostPointerCapture={event => finishPointer(event, false)}
        ><span className="nav-tile"><Icon size={19} weight={tab === id ? 'fill' : 'duotone'}/></span><span>{label}</span></button>;
      })}
    </nav>
    <span className="sr-only" role="status">{announcement}</span>
    <ContextMenu onResetTabOrder={element => {
      if (gesture.current || order.every((id, index) => id === defaults[index])) return;
      captureReorder();
      restoreFocus.current = (element?.dataset.tab as Tab | undefined) ?? null;
      commit(defaults, 'Tab order reset');
    }}/>
  </>;
}
