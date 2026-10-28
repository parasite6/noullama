export function authorImageSources(avatar: string) {
  try {
    const url = new URL(avatar);
    if (url.protocol === 'https:' && url.hostname === 'cdn-avatars.huggingface.co' && url.pathname.startsWith('/v1/production/uploads/')) {
      return { primary: `https://cdn-uploads.huggingface.co${url.pathname.slice(3)}`, fallback: avatar };
    }
  } catch { /* Keep malformed URLs on the normal image-error fallback path. */ }
  return { primary: avatar, fallback: avatar };
}
