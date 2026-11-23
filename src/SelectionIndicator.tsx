import { useLayoutEffect, useRef } from 'react';

type SelectionIndicatorProps = {
  selected: string;
  offsetY?: number;
  dragging?: boolean;
  followLayout?: boolean;
};

type Geometry = { x: number; y: number; width: number; height: number };
const reducedMotionQuery = '(prefers-reduced-motion: reduce)';

export function SelectionIndicator({ selected, offsetY = 0, dragging = false, followLayout = false }: SelectionIndicatorProps) {
  const indicator = useRef<HTMLSpanElement>(null);
  const observer = useRef<ResizeObserver | null>(null);
  const preference = useRef<MediaQueryList | null>(null);
  const observed = useRef(new Set<Element>());
  const position = useRef(() => {});
  const active = useRef<HTMLElement | null>(null);
  const painted = useRef<Geometry | null>(null);
  const motion = useRef<Animation | null>(null);

  useLayoutEffect(() => {
    const layer = indicator.current!;
    const resize = new ResizeObserver(() => position.current());
    const visibility = new IntersectionObserver(() => position.current());
    observer.current = resize;
    visibility.observe(layer.parentElement!);
    const media = matchMedia(reducedMotionQuery);
    preference.current = media;
    const remeasure = () => position.current();
    media.addEventListener('change', remeasure);
    let mounted = true;
    void document.fonts.ready.then(() => { if (mounted) remeasure(); });

    return () => {
      mounted = false;
      resize.disconnect();
      visibility.disconnect();
      media.removeEventListener('change', remeasure);
      observer.current = null;
      observed.current.clear();
      motion.current?.cancel();
      motion.current = null;
    };
  }, []);

  useLayoutEffect(() => {
    const layer = indicator.current!;
    // The child's ref is already attached here. The parent's ref may still be null.
    const container = layer.parentElement!;
    const resize = observer.current!;
    const targets = new Set<Element>([container, ...container.querySelectorAll('button')]);
    for (const target of observed.current) if (!targets.has(target)) resize.unobserve(target);
    for (const target of targets) if (!observed.current.has(target)) resize.observe(target, { box: 'border-box' });
    observed.current = targets;

    position.current = () => {
      const option = container.querySelector<HTMLElement>(selected);
      const width = option?.offsetWidth ?? 0;
      const height = option?.offsetHeight ?? 0;
      if (!option || !width || !height || option.checkVisibility?.() === false) {
        container.removeAttribute('data-indicator-ready');
        container.removeAttribute('data-indicator-moving');
        painted.current = null;
        motion.current?.cancel();
        motion.current = null;
        return;
      }

      const changed = option !== active.current;
      active.current = option;
      const target = { width, height, x: option.offsetLeft, y: option.offsetTop + offsetY };
      if (followLayout) {
        const rect = option.getBoundingClientRect();
        target.width = parseFloat(getComputedStyle(option).width);
        // Use the center to retain fractional positions without including the press scale.
        target.x = rect.left + (rect.width - target.width) / 2
          - container.getBoundingClientRect().left - container.clientLeft;
      }

      const previous = painted.current;
      const sameTarget = previous && target.x === previous.x && target.y === previous.y
        && target.width === previous.width && target.height === previous.height;
      const reduce = preference.current!.matches;
      if (sameTarget && !reduce) return;

      const style = getComputedStyle(layer);
      const start = { transform: style.transform, width: style.width, height: style.height };
      const finish = {
        transform: `translate3d(${target.x}px, ${target.y}px, 0)`,
        width: target.width + 'px',
        height: target.height + 'px',
      };
      const glide = followLayout && previous && (changed || motion.current) && !reduce;
      const time = glide ? style.getPropertyValue('--motion-selection').trim() : '0ms';
      const duration = parseFloat(time) * (time.endsWith('ms') ? 1 : 1000);
      const easing = glide ? style.getPropertyValue('--ease-selection').trim() : '';
      motion.current?.cancel();
      motion.current = null;
      container.removeAttribute('data-indicator-moving');
      Object.assign(layer.style, finish);
      painted.current = target;
      if (glide) {
        const animation = layer.animate([start, finish], { duration, easing, fill: 'both' });
        animation.startTime = document.timeline.currentTime;
        motion.current = animation;
        container.setAttribute('data-indicator-moving', '');
        animation.onfinish = () => {
          if (motion.current !== animation) return;
          animation.cancel();
          motion.current = null;
          container.removeAttribute('data-indicator-moving');
        };
      }
      if (!container.hasAttribute('data-indicator-ready')) container.setAttribute('data-indicator-ready', '');
    };
    position.current();
  });

  return <span ref={indicator} className="selection-indicator" data-dragging={dragging || undefined} aria-hidden="true"/>;
}
