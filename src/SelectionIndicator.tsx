import { useLayoutEffect, useRef, type RefObject } from 'react';

type SelectionIndicatorProps = {
  track: RefObject<HTMLElement | null>;
  selected: string;
  offsetY?: number;
  dragging?: boolean;
};

export function SelectionIndicator({ track, selected, offsetY = 0, dragging = false }: SelectionIndicatorProps) {
  const indicator = useRef<HTMLSpanElement>(null);

  useLayoutEffect(() => {
    const container = track.current;
    const layer = indicator.current;
    if (!container || !layer) return;
    const option = container.querySelector<HTMLElement>(selected);
    if (!option) {
      layer.hidden = true;
      return;
    }

    function position() {
      if (!layer || !option) return;
      layer.style.width = `${option.offsetWidth}px`;
      layer.style.height = `${option.offsetHeight}px`;
      layer.style.transform = `translate3d(${option.offsetLeft}px, ${option.offsetTop + offsetY}px, 0)`;
      layer.hidden = false;
    }
    position();

    const observer = new ResizeObserver(position);
    observer.observe(container);
    for (const child of container.querySelectorAll('button')) observer.observe(child);
    return () => observer.disconnect();
  });

  return <span ref={indicator} className="selection-indicator" data-dragging={dragging || undefined} aria-hidden="true" hidden/>;
}
