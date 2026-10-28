import React, { useEffect, useState } from 'react';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { authorImageSources } from './author-image';

const avatars = new Map<string, Promise<string | null>>();
const liquidLogo = new URL('../assets/authors/liquid-ai.png', import.meta.url).href;

export function authorFromPath(path: string): string | null {
  return path.split(/[\\/]/).reverse().map(part => part.replace(/^models--/, '').match(/^([a-zA-Z0-9_.-]+?)--.+$/)?.[1]).find(Boolean) ?? null;
}

export function AuthorLogo({ author, size = 28 }: { author: string | null; size?: number }) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const [useAvatar, setUseAvatar] = useState(false);
  const liquid = author?.toLowerCase() === 'liquidai';
  const pixels = Math.max(24, Math.min(192, Math.round((size - 2) * window.devicePixelRatio)));
  useEffect(() => {
    setUrl(null); setFailed(false); setUseAvatar(false);
    let cancelled = false;
    if (author && isTauri()) {
      const key = `${author}:${pixels}`;
      let request = avatars.get(key);
      if (!request) {
        const oldest = avatars.keys().next().value;
        if (avatars.size >= 64 && oldest) avatars.delete(oldest);
        request = invoke<string | null>('author_avatar', { author, pixels }).catch(() => null); avatars.set(key, request);
      }
      request.then(url => { if (!cancelled) setUrl(url); });
    }
    return () => { cancelled = true; };
  }, [author, pixels]);
  const sources = url ? authorImageSources(url) : liquid ? { primary: liquidLogo, fallback: liquidLogo } : null;
  return <span className="author-logo" style={{ width: size, height: size }} title={author ?? 'Local model'} aria-label={author ? `${author} logo` : 'Local model'}>
    {sources && !failed ? <img src={useAvatar ? sources.fallback : sources.primary} alt="" width={size} height={size} loading="lazy" decoding="async" referrerPolicy="no-referrer" onError={() => { if (!useAvatar && sources.primary !== sources.fallback) setUseAvatar(true); else setFailed(true); }}/> : <span aria-hidden="true">{author?.slice(0, 2).toUpperCase() ?? 'LM'}</span>}
  </span>;
}
