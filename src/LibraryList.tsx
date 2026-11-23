import { Fragment, useLayoutEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from 'react';
import { libraryUsesWindow } from './ui-state';

const defaultEstimatedHeight = 300;
export type LibraryScroll = { top: number; estimatedHeight: number };
const overscan = 1000;
const focusable = 'button:enabled, input:enabled, select:enabled, textarea:enabled, [tabindex="0"]';

export function LibraryList({ ids, renderRow, scrollPosition }: {
  ids: number[]; renderRow: (index: number) => ReactNode; scrollPosition: RefObject<LibraryScroll>;
}) {
  const windowed = libraryUsesWindow(ids.length);
  const list = useRef<HTMLDivElement>(null);
  const focusNext = useRef<{ index: number; reverse: boolean } | null>(null);
  const estimatedHeight = useRef(scrollPosition.current.estimatedHeight || defaultEstimatedHeight);
  const heights = useRef(new Map<number, number>());
  const [revision, setRevision] = useState(0);
  const [viewport, setViewport] = useState({ top: 0, bottom: 0 });
  const [focused, setFocused] = useState<number | null>(null);
  const offsets = useMemo(() => {
    const result = [0];
    for (const id of ids) result.push(result[result.length - 1] + (heights.current.get(id) ?? estimatedHeight.current));
    return result;
  }, [ids, revision]);
  const firstOffset = offsets.findIndex(offset => offset > viewport.top - overscan);
  const first = firstOffset < 0 ? ids.length : Math.max(0, firstOffset - 1);
  const lastOffset = offsets.findIndex(offset => offset > viewport.bottom + overscan);
  const last = lastOffset < 0 ? ids.length : lastOffset;
  const visible = windowed ? ids.flatMap((id, index) => (index >= first && index < last) || id === focused ? [index] : []) : [];
  const visibleIds = visible.map(index => ids[index]).join(',');

  useLayoutEffect(() => {
    if (!windowed) {
      if (!ids.length) return;
      heights.current.clear();
      estimatedHeight.current = defaultEstimatedHeight;
      if (scrollPosition.current.top) window.scrollTo({ top: 0, behavior: 'instant' });
      scrollPosition.current = { top: 0, estimatedHeight: 0 };
      return;
    }
    const node = list.current;
    if (!node) return;
    let frame = 0;
    const update = () => {
      scrollPosition.current.top = window.scrollY;
      const top = -node.getBoundingClientRect().top;
      const bottom = top + window.innerHeight;
      setViewport(current => current.top === top && current.bottom === bottom ? current : { top, bottom });
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(() => { frame = 0; update(); });
    };
    if (scrollPosition.current.top) window.scrollTo({ top: scrollPosition.current.top, behavior: 'instant' });
    update();
    const resize = new ResizeObserver(schedule);
    resize.observe(node);
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      window.removeEventListener('scroll', schedule);
      window.removeEventListener('resize', schedule);
    };
  }, [windowed, ids.length === 0, scrollPosition]);

  useLayoutEffect(() => {
    if (!windowed || !list.current) return;
    const node = list.current;
    const rows = node.querySelectorAll<HTMLElement>('[data-library-row]');
    const measure = () => {
      let changed = false;
      let correction = 0;
      const needsEstimate = heights.current.size === 0 && window.scrollY === 0;
      for (const child of rows) {
        const id = Number(child.dataset.libraryRow);
        const height = child.offsetHeight;
        const previous = heights.current.get(id) ?? estimatedHeight.current;
        if (height === previous) continue;
        if (child.getBoundingClientRect().bottom <= 0) correction += height - previous;
        heights.current.set(id, height);
        changed = true;
      }
      if (needsEstimate && heights.current.size) {
        estimatedHeight.current = [...heights.current.values()].reduce((sum, height) => sum + height, 0) / heights.current.size;
        scrollPosition.current.estimatedHeight = estimatedHeight.current;
      }
      if (correction) window.scrollBy({ top: correction, behavior: 'instant' });
      if (changed) setRevision(value => value + 1);
    };
    const resize = new ResizeObserver(measure);
    for (const row of rows) resize.observe(row);
    measure();
    return () => resize.disconnect();
  }, [windowed, visibleIds, first, last]);

  useLayoutEffect(() => {
    const next = focusNext.current;
    if (!next || !list.current) return;
    const row = list.current.querySelector('[data-index="' + next.index + '"]');
    const controls = row?.querySelectorAll<HTMLElement>(focusable);
    if (!controls?.length) return;
    focusNext.current = null;
    controls[next.reverse ? controls.length - 1 : 0].focus();
  });

  if (!windowed) return <div className="library-list">{ids.map((id, index) =>
    <Fragment key={id}>{renderRow(index)}</Fragment>)}</div>;

  const children: ReactNode[] = [];
  let previous = 0;
  for (const index of visible) {
    if (index > previous) children.push(<div key={'gap-' + ids[index]} aria-hidden="true" style={{ height: offsets[index] - offsets[previous] }}/>);
    children.push(<div className="library-window-row" data-library-row={ids[index]} data-index={index} key={ids[index]}>
      {renderRow(index)}
    </div>);
    previous = index + 1;
  }
  if (previous < ids.length) children.push(<div key="end" aria-hidden="true" style={{ height: offsets[ids.length] - offsets[previous] - 10 }}/>);
  return <div className="library-window" ref={list}
    onKeyDown={event => {
      if (event.key !== 'Tab' || event.defaultPrevented) return;
      const row = (event.target as HTMLElement).closest<HTMLElement>('[data-library-row]');
      if (!row) return;
      const controls = row.querySelectorAll<HTMLElement>(focusable);
      if (event.target !== controls[event.shiftKey ? 0 : controls.length - 1]) return;
      const next = Number(row.dataset.index) + (event.shiftKey ? -1 : 1);
      if (next < 0 || next >= ids.length || visible.includes(next)) return;
      event.preventDefault();
      focusNext.current = { index: next, reverse: event.shiftKey };
      setFocused(ids[next]);
    }}
    onFocusCapture={event => {
      const row = (event.target as HTMLElement).closest<HTMLElement>('[data-library-row]');
      setFocused(row ? Number(row.dataset.libraryRow) : null);
    }}
    onBlurCapture={event => {
      if (!event.currentTarget.contains(event.relatedTarget)) setFocused(null);
    }}>{children}</div>;
}
