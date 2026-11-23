import React, { useEffect, useState } from 'react';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { SpinnerGapIcon as Spinner } from '@phosphor-icons/react/dist/icons/SpinnerGap';
import { authorImageSources } from './author-image';

const avatars = new Map<string, Promise<string | null>>();
// Answers already in, so a logo seen before renders at once with no spinner.
const resolved = new Map<string, string | null>();
// Image URLs that have finished loading, so a cached image skips the spinner.
const painted = new Set<string>();

export function authorFromPath(path: string): string | null {
  return path.split(/[\\/]/).reverse().map(part => part.replace(/^models--/, '').match(/^([a-zA-Z0-9_.-]+?)--.+$/)?.[1]).find(Boolean) ?? null;
}

// Avatars come from the Hugging Face profile for the repository owner, looked up
// the first time that author appears. While the lookup or the image is still
// loading, a small spinner holds the spot; the image then fades in. When the
// profile has no usable image, the author's initials are shown.
export function AuthorLogo({ author, size = 28 }: { author: string | null; size?: number }) {
  const pixels = Math.max(24, Math.min(192, Math.round((size - 2) * window.devicePixelRatio)));
  const key = author ? `${author}:${pixels}` : '';
  const lookup = !!author && isTauri();
  const [url, setUrl] = useState<string | null | undefined>(() => lookup ? resolved.get(key) : null);
  const [failed, setFailed] = useState(false);
  const [useAvatar, setUseAvatar] = useState(false);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    setFailed(false); setUseAvatar(false);
    if (!lookup) { setUrl(null); return; }
    if (resolved.has(key)) { setUrl(resolved.get(key)); return; }
    setUrl(undefined);
    let cancelled = false;
    let request = avatars.get(key);
    if (!request) {
      const oldest = avatars.keys().next().value;
      if (avatars.size >= 64 && oldest) { avatars.delete(oldest); resolved.delete(oldest); }
      request = invoke<string | null>('author_avatar', { author, pixels }).catch(() => null);
      avatars.set(key, request);
    }
    request.then(found => { resolved.set(key, found); if (!cancelled) setUrl(found); });
    return () => { cancelled = true; };
  }, [key, lookup]);
  const sources = url ? authorImageSources(url) : null;
  const src = sources && !failed ? (useAvatar ? sources.fallback : sources.primary) : null;
  useEffect(() => { setLoaded(!!src && painted.has(src)); }, [src]);
  const ready = loaded || (!!src && painted.has(src));
  const waiting = url === undefined || (!!src && !ready);
  return <span className={`author-logo${waiting ? ' waiting' : ''}`} style={{ width: size, height: size }} title={author ?? 'Local model'} aria-label={author ? `${author} logo` : 'Local model'} aria-busy={waiting || undefined}>
    {src && <img className={ready ? 'ready' : ''} src={src} alt="" width={size} height={size} decoding="async" referrerPolicy="no-referrer"
      onLoad={() => { painted.add(src); setLoaded(true); }}
      onError={() => { if (!useAvatar && sources && sources.primary !== sources.fallback) setUseAvatar(true); else setFailed(true); }}/>}
    {waiting && <Spinner className="spin author-spinner" size={Math.max(12, Math.round(size * .45))} weight="bold" aria-hidden="true"/>}
    {!src && !waiting && <span aria-hidden="true">{author?.slice(0, 2).toUpperCase() ?? 'LM'}</span>}
  </span>;
}
