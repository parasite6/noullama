// Browsers chain wheel scrolling from a field to the page once the field hits
// an edge, which makes the page jump while editing. Stop that scroll at the
// edge of whichever scrollable box the pointer is over.
export function containScrollChaining() {
  const onWheel = (event: WheelEvent) => {
    if (event.defaultPrevented || event.ctrlKey || event.deltaY === 0) return;
    let node = event.target instanceof Element ? event.target : null;
    while (node && node !== document.body && node !== document.documentElement) {
      const scrollable = /(auto|scroll)/.test(getComputedStyle(node).overflowY) && node.scrollHeight > node.clientHeight + 1;
      if (scrollable) {
        const atTop = node.scrollTop <= 0;
        const atBottom = node.scrollTop + node.clientHeight >= node.scrollHeight - 1;
        if ((event.deltaY < 0 && atTop) || (event.deltaY > 0 && atBottom)) event.preventDefault();
        return;
      }
      node = node.parentElement;
    }
  };
  window.addEventListener('wheel', onWheel, { passive: false });
  return () => window.removeEventListener('wheel', onWheel);
}
