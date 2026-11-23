import React, { useEffect, useRef, useState } from 'react';
import { WarningCircleIcon as AlertCircle } from '@phosphor-icons/react/dist/icons/WarningCircle';
import { XIcon as X } from '@phosphor-icons/react/dist/icons/X';

type Banner = { text: string; error: boolean };

// The app's notification: floats over the page instead of pushing it down,
// slides in, and fades out when dismissed. Errors stay until dismissed;
// other messages leave on their own after a few seconds.
export function Toast({ banner, onDismiss }: { banner: Banner | null; onDismiss: () => void }) {
  const [shown, setShown] = useState<Banner | null>(banner);
  const [leaving, setLeaving] = useState(false);
  const dismiss = useRef(onDismiss);
  dismiss.current = onDismiss;
  useEffect(() => {
    if (banner) { setShown(banner); setLeaving(false); return; }
    if (!shown) return;
    setLeaving(true);
    const timer = window.setTimeout(() => { setShown(null); setLeaving(false); }, 180);
    return () => window.clearTimeout(timer);
  }, [banner?.text, banner?.error]);
  useEffect(() => {
    if (!banner || banner.error) return;
    const timer = window.setTimeout(() => dismiss.current(), 6000);
    return () => window.clearTimeout(timer);
  }, [banner?.text, banner?.error]);
  if (!shown) return null;
  return <div className="toast-region">
    <div key={shown.text} className={`notice toast ${shown.error ? 'error' : ''} ${leaving ? 'leaving' : ''}`} role={shown.error ? 'alert' : 'status'}>
      <span className="notice-text">{shown.error && <AlertCircle size={16}/>}<span>{shown.text}</span></span>
      <button className="icon-button small" aria-label="Dismiss notification" onClick={() => dismiss.current()}><X size={16}/></button>
    </div>
  </div>;
}
