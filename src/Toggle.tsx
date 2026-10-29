import React from 'react';

export function Toggle({ checked, onChange, label, ariaLabel, title, disabled }: { checked: boolean; onChange: (checked: boolean) => void; label: string; ariaLabel?: string; title?: string; disabled?: boolean }) {
  return <button type="button" role="switch" aria-checked={checked} aria-label={ariaLabel} className={`toggle ${checked ? 'on' : ''}`} title={title} disabled={disabled} onClick={() => onChange(!checked)}>
    <span className="toggle-track"><span className="toggle-thumb"/></span>{label}
  </button>;
}
