import { useEffect, useState, type MouseEvent } from 'react';

export const busyDelay = 300;

function stopBusyClick(event: MouseEvent<HTMLButtonElement>) {
  event.preventDefault();
  event.stopPropagation();
}

/** Keep temporary locks focusable without permitting another activation. */
export function busyButton(busy: boolean, unavailable = false, indicate = busy) {
  return {
    disabled: unavailable && !busy,
    'aria-disabled': busy || undefined,
    'aria-busy': indicate || undefined,
    onClickCapture: busy ? stopBusyClick : undefined,
  };
}

export function useDelayedPending<T extends string | boolean | null>(pending: T): T | null {
  const [visible, setVisible] = useState<T | null>(null);
  useEffect(() => {
    if (!pending) return;
    const timer = window.setTimeout(() => setVisible(pending), busyDelay);
    return () => {
      window.clearTimeout(timer);
      setVisible(null);
    };
  }, [pending]);
  return visible === pending ? visible : null;
}
