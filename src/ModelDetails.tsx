import React from 'react';

export type MemoryUse = { process_bytes: number | null; used_bytes: number | null; total_bytes: number | null };
type Props = {
  name: string; status: string; path: string; device: string | null; projector: string | null;
  contextSize: number; gpuLayers: number; runtime: string; endpoint: string | null;
  memory: MemoryUse | null; estimate: number | null; gpuName: string | null;
};
const gb = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(1)} GB`;

function Meter({ label, value, total, strong }: { label: string; value: number; total: number; strong?: boolean }) {
  return <div className="meter">
    <div className="meter-label"><span>{label}</span><span><strong>{gb(value)}</strong> of {gb(total)}</span></div>
    <div className="meter-track"><i className={strong ? 'strong' : ''} style={{ width: `${Math.min(100, value / total * 100)}%` }}/></div>
  </div>;
}

// Shown while the sidebar runtime card is hovered or focused.
export function ModelDetails(props: Props) {
  const { memory } = props;
  const total = memory?.total_bytes ?? null;
  return <div className="model-details" role="tooltip">
    <div className="model-details-head"><strong title={props.path}>{props.name}</strong><span>{props.status}</span></div>
    <dl>
      {props.device && <><dt>Device</dt><dd>{props.device}</dd></>}
      <dt>Vision</dt><dd title={props.projector ?? undefined}>{props.projector ?? 'Off'}</dd>
      <dt>Context</dt><dd>{props.contextSize.toLocaleString()} tokens</dd>
      <dt>GPU layers</dt><dd>{props.gpuLayers === 0 ? 'None (CPU)' : props.gpuLayers}</dd>
      <dt>llama.cpp</dt><dd>{props.runtime}</dd>
      {props.endpoint && <><dt>API</dt><dd><code>{props.endpoint}</code></dd></>}
    </dl>
    <div className="model-details-memory">
      <div className="model-details-title">Graphics memory{props.gpuName ? <span> / {props.gpuName}</span> : null}</div>
      {total ? <>
        {memory?.process_bytes != null && <Meter label="This model" value={memory.process_bytes} total={total} strong/>}
        {memory?.used_bytes != null && <Meter label="In use on the GPU" value={memory.used_bytes} total={total}/>}
      </> : <p>{!memory ? 'Reading usage…' : props.gpuName ? 'Live usage is not available for this GPU.' : 'No graphics card found. The model runs from system memory.'}</p>}
      {props.estimate != null && <p>Estimated need for this model: about {gb(props.estimate)}.</p>}
    </div>
  </div>;
}
