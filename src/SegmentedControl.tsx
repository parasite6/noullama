import { useRef, type ReactNode } from 'react';
import { SelectionIndicator } from './SelectionIndicator';

export function SegmentedControl({ label, children }: { label: string; children: ReactNode }) {
  const track = useRef<HTMLDivElement>(null);
  return <div ref={track} className="segmented" role="group" aria-label={label}>
    <SelectionIndicator track={track} selected="button.on"/>
    {children}
  </div>;
}
