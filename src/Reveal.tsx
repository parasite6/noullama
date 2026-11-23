import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';

type Props = {
  open: unknown;
  swapKey?: string;
  id?: string;
  className?: string;
  children: ReactNode;
};

export function Reveal({ open, swapKey = '', id, className = '', children }: Props) {
  const visible = Boolean(open);
  const [retained, setRetained] = useState(visible);
  const [displayedKey, setDisplayedKey] = useState(swapKey);
  const root = useRef<HTMLDivElement>(null);
  const body = useRef<HTMLDivElement>(null);
  const content = useRef(children);
  const wasVisible = useRef(visible);
  const swapHeight = useRef<number | null>(null);
  const motions = useRef<Animation[]>([]);
  const current = visible && displayedKey === swapKey;

  useLayoutEffect(() => {
    if (current) content.current = children;
  });

  useLayoutEffect(() => {
    const node = root.current;
    const inner = body.current;
    const previouslyVisible = wasVisible.current;
    wasVisible.current = visible;
    if (!node || !inner) return;

    const running = motions.current.length > 0;
    const height = running ? node.getBoundingClientRect().height : undefined;
    const style = getComputedStyle(inner);
    const opacity = running ? style.opacity : undefined;
    const transform = running ? style.transform : undefined;
    const enterFrom = swapHeight.current;
    swapHeight.current = null;

    const cancel = () => {
      motions.current.forEach(animation => animation.cancel());
      motions.current = [];
    };
    cancel();

    if (matchMedia('(prefers-reduced-motion: reduce)').matches || !node.getClientRects().length) {
      setDisplayedKey(swapKey);
      setRetained(visible);
      return;
    }

    const swap = visible && displayedKey !== swapKey;
    if (swap && previouslyVisible) {
      const fade = inner.animate([
        { opacity: opacity ?? 1, transform: transform ?? 'none' },
        { opacity: 0, transform: 'translateY(-2px)' },
      ], { duration: 90, easing: 'ease-in', fill: 'both' });
      motions.current = [fade];
      fade.onfinish = () => {
        swapHeight.current = node.getBoundingClientRect().height;
        setDisplayedKey(swapKey);
      };
      return;
    }

    if (swap) {
      swapHeight.current = height ?? 0;
      setDisplayedKey(swapKey);
      return;
    }

    if (!running && enterFrom === null && visible === previouslyVisible) return;

    const naturalHeight = inner.getBoundingClientRect().height;
    const from = enterFrom ?? height ?? (previouslyVisible ? naturalHeight : 0);
    const rootStyle = getComputedStyle(node);
    const time = rootStyle.getPropertyValue('--motion-enter').trim();
    const duration = enterFrom === null ? parseFloat(time) * (time.endsWith('ms') ? 1 : 1000) : 120;
    const timing = { duration, easing: rootStyle.getPropertyValue('--ease-reorder').trim(), fill: 'both' } as const;
    const size = node.animate([
      { height: from + 'px', overflow: 'hidden' },
      { height: (visible ? naturalHeight : 0) + 'px', overflow: 'hidden' },
    ], timing);
    const fade = inner.animate([
      { opacity: enterFrom !== null ? 0 : opacity ?? (previouslyVisible ? 1 : 0), transform: enterFrom !== null ? 'translateY(2px)' : transform ?? (previouslyVisible ? 'none' : 'translateY(-3px)') },
      { opacity: visible ? 1 : 0, transform: visible ? 'none' : 'translateY(-3px)' },
    ], timing);
    motions.current = [size, fade];
    if (visible) setRetained(true);
    size.onfinish = () => {
      if (visible) cancel();
      else {
        motions.current = [];
        setRetained(false);
      }
    };
  }, [visible, swapKey, displayedKey]);

  useLayoutEffect(() => () => {
    motions.current.forEach(animation => animation.cancel());
    motions.current = [];
  }, []);

  const present = visible || retained;
  if (!present && !id) return null;
  return <div ref={root} id={id} className={`reveal ${className}`} hidden={!present} inert={!visible}>
    {present && <div ref={body} className="reveal-content">{current ? children : content.current}</div>}
  </div>;
}
