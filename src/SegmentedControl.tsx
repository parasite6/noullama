import type { ReactNode } from 'react';
import { SelectionIndicator } from './SelectionIndicator';

export function SegmentedControl({ label, children }: { label: string; children: ReactNode }) {
  return <div className="segmented" role="group" aria-label={label}>
    <SelectionIndicator selected="button.on" followLayout/>
    {children}
  </div>;
}
