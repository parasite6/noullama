import { useRef, useState } from 'react';

// Publish the draft immediately and send writes to the same destination in
// click order. Compare write identity so A → B → A keeps the last A visible
// even when the first A settles.
export function useOptimisticWrites<Values extends object>(onError: (reason: unknown) => void) {
  const [drafts, setDrafts] = useState<Partial<Values>>({});
  const pending = useRef(new Map<keyof Values, Promise<void>>());

  function write<K extends keyof Values>(key: K, value: Values[K], save: () => Promise<void>) {
    setDrafts(current => ({ ...current, [key]: value }));
    const previous = pending.current.get(key) ?? Promise.resolve();
    const request = previous.then(save).catch(onError).finally(() => {
      if (pending.current.get(key) !== request) return;
      pending.current.delete(key);
      setDrafts(current => {
        const next = { ...current };
        delete next[key];
        return next;
      });
    });
    pending.current.set(key, request);
  }

  return { drafts, write };
}
