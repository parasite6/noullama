import React, { useEffect, useRef, useState } from 'react';
import { CheckIcon as Check } from '@phosphor-icons/react/dist/icons/Check';
import { CopyIcon as Copy } from '@phosphor-icons/react/dist/icons/Copy';

// Most text in the app is not selectable, so identifiers worth reusing (model
// names, paths, repository ids, URLs) get one of these instead.
export function CopyButton({ value, label, size = 14, className = '', weight = 'duotone', onError }: { value: string; label: string; size?: number; className?: string; weight?: 'duotone' | 'fill'; onError?: (message: string) => void }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);
  async function copy(event: React.MouseEvent) {
    event.stopPropagation();
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true); clearTimeout(timer.current); timer.current = setTimeout(() => setCopied(false), 1800);
    } catch (e) { onError?.(`Could not copy: ${e instanceof Error ? e.message : String(e)}`); }
  }
  return <button type="button" className={`icon-button small copy-button ${copied ? 'copied' : ''} ${className}`} aria-label={copied ? 'Copied' : label} title={copied ? 'Copied' : label} onClick={copy}>
    {copied ? <Check size={size} weight="fill"/> : <Copy size={size} weight={weight}/>}
  </button>;
}
