import React, { useRef } from 'react';
import { ImagePlus, X } from 'lucide-react';

export type Attachment = { name: string; url: string; bytes: number };

// llama.cpp decodes images with stb_image; anything else the webview can show
// (WebP, AVIF, SVG) is redrawn as PNG before it is sent.
const native = ['image/png', 'image/jpeg', 'image/gif', 'image/bmp'];
const MAX_BYTES = 20 * 1024 * 1024;
// Everything attached travels as base64 through IPC in one request.
export const MAX_TOTAL_BYTES = 60 * 1024 * 1024;

const readAsDataUrl = (blob: Blob) => new Promise<string>((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(String(reader.result));
  reader.onerror = () => reject(reader.error ?? new Error('Could not read the file.'));
  reader.readAsDataURL(blob);
});

async function toPng(url: string, name: string) {
  const image = new Image();
  image.src = url;
  try { await image.decode(); } catch { throw new Error(`${name} is not an image this app can read.`); }
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
  canvas.getContext('2d')!.drawImage(image, 0, 0);
  const png = canvas.toDataURL('image/png');
  // An image with no size (some SVGs) or one past the canvas limit encodes as "data:,".
  if (!png.startsWith('data:image/png;base64,') || png.length <= 'data:image/png;base64,'.length) throw new Error(`${name} could not be converted to PNG.`);
  return png;
}

export async function readImages(files: Iterable<File>): Promise<Attachment[]> {
  const images: Attachment[] = [];
  for (const file of files) {
    if (!file.type.startsWith('image/')) throw new Error(`${file.name || 'That file'} is not an image.`);
    if (file.size > MAX_BYTES) throw new Error(`${file.name} is larger than 20 MB.`);
    const read = await readAsDataUrl(file);
    const url = native.includes(file.type) ? read : await toPng(read, file.name);
    // Base64 is 4/3 of the bytes, and a converted PNG can be far larger than the original.
    const bytes = Math.ceil((url.length - url.indexOf(',') - 1) * 3 / 4);
    if (bytes > MAX_BYTES) throw new Error(`${file.name || 'That image'} is larger than 20 MB${url === read ? '' : ' after converting it to PNG'}.`);
    images.push({ name: file.name || 'Pasted image', url, bytes });
  }
  return images;
}

export function ImageAttachments({ images, onFiles, onRemove, disabled }: { images: Attachment[]; onFiles: (files: File[]) => void; onRemove: (index: number) => void; disabled?: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  return <div className="image-strip">
    {images.map((image, index) => <figure className="image-thumb" key={`${image.name}-${index}`} title={image.name}>
      <img src={image.url} alt={image.name}/>
      <button type="button" className="image-remove" aria-label={`Remove ${image.name}`} onClick={() => onRemove(index)}><X size={12}/></button>
    </figure>)}
    <button type="button" className="image-add" disabled={disabled} onClick={() => input.current?.click()}><ImagePlus size={18}/><span>{images.length ? 'Add more' : 'Add images'}</span></button>
    {!images.length && <small>Or drop or paste images here. They are sent with the state.</small>}
    <input ref={input} type="file" accept="image/*" multiple hidden onChange={event => {
      const files = [...(event.target.files ?? [])];
      event.target.value = '';
      onFiles(files);
    }}/>
  </div>;
}
