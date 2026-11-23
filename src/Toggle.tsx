export function Toggle({ checked, onChange, label, ariaLabel, title, disabled, locked = false }: { checked: boolean; onChange: (checked: boolean) => void; label: string; ariaLabel?: string; title?: string; disabled?: boolean; locked?: boolean }) {
  return <button type="button" role="switch" aria-checked={checked} aria-label={ariaLabel ?? label} className={`toggle ${checked ? 'on' : ''}`} title={title} disabled={disabled} aria-disabled={locked || undefined} onClick={() => { if (!locked) onChange(!checked); }}>
    <span className="toggle-track" aria-hidden="true"><span className="toggle-thumb"/></span>
    {label !== 'On' && label !== 'Off' && <span>{label}</span>}
  </button>;
}
