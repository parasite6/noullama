import { useEffect, useRef } from 'react';
import { EditorState } from '@codemirror/state';
import { EditorView, drawSelection, keymap, lineNumbers, highlightActiveLineGutter } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import { bracketMatching, foldGutter, HighlightStyle, indentOnInput, syntaxHighlighting } from '@codemirror/language';
import { json, jsonParseLinter } from '@codemirror/lang-json';
import { linter, lintGutter } from '@codemirror/lint';
import { tags } from '@lezer/highlight';

const colors = HighlightStyle.define([
  { tag: tags.propertyName, color: '#7cb7ff' },
  { tag: tags.string, color: '#9ece6a' },
  { tag: tags.number, color: '#ff9e64' },
  { tag: [tags.bool, tags.null], color: '#f7768e' },
  { tag: [tags.punctuation, tags.bracket, tags.separator], color: '#8a96a3' },
  { tag: tags.invalid, color: '#ff5f5f', textDecoration: 'underline wavy' },
]);
const theme = EditorView.theme({
  '&': { backgroundColor: '#19191b', color: '#c9c9cf', height: '100%', fontSize: '13px' },
  '.cm-scroller': { fontFamily: '"Geist Mono Variable", "SFMono-Regular", Consolas, monospace', lineHeight: '1.7', overflow: 'auto', overscrollBehavior: 'contain' },
  '.cm-content': { padding: '12px 0', caretColor: '#ececee' },
  '.cm-line': { padding: '0 12px' },
  '.cm-gutters': { backgroundColor: '#19191b', color: '#5a5a61', borderRight: '0' },
  '.cm-activeLineGutter': { backgroundColor: '#232325', color: '#b6b6bd' },
  '.cm-cursor': { borderLeftColor: '#ececee' },
  '&.cm-focused': { outline: 'none' },
  // Drawn by drawSelection(), so it shows in WebKitGTK where the native selection does not.
  '.cm-selectionBackground': { backgroundColor: '#2a3a55' },
  '&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground': { backgroundColor: '#33507d' },
  '.cm-selectionMatch': { backgroundColor: '#24324a' },
  '.cm-matchingBracket, &.cm-focused .cm-matchingBracket': { backgroundColor: '#2d3b2a', color: '#e0e6e3' },
  '.cm-tooltip': { backgroundColor: '#2b2b2e', color: '#ececee', border: '1px solid #46464c', borderRadius: '8px' },
  '.cm-diagnostic': { fontSize: '11px' },
  '.cm-foldGutter .cm-gutterElement': { padding: '0 4px 0 2px', cursor: 'pointer' },
  '.cm-fold-marker': { display: 'inline-flex', verticalAlign: 'middle', color: '#6a6a72', transition: 'transform .15s' },
  '.cm-fold-marker:hover': { color: '#ececee' },
  '.cm-fold-marker.open': { transform: 'rotate(90deg)' },
  '.cm-foldPlaceholder': { backgroundColor: '#2b2b2e', border: '0', borderRadius: '6px', color: '#b6b6bd', padding: '0 6px' },
}, { dark: true });

// The default fold markers are tiny text glyphs; draw the app's chevron instead.
function foldMarker(open: boolean) {
  const marker = document.createElement('span');
  marker.className = open ? 'cm-fold-marker open' : 'cm-fold-marker';
  marker.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>';
  return marker;
}

export function JsonEditor({ value, onChange, label, readOnly = false }: { value: string; onChange?: (value: string) => void; label: string; readOnly?: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const editor = useRef<EditorView | null>(null);
  const change = useRef(onChange);
  change.current = onChange;
  useEffect(() => {
    if (!host.current) return;
    const view = new EditorView({ parent: host.current, state: EditorState.create({ doc: value, extensions: [
      json(), theme, EditorView.lineWrapping, syntaxHighlighting(colors), lineNumbers(), highlightActiveLineGutter(),
      drawSelection(), bracketMatching(), foldGutter({ markerDOM: foldMarker }), indentOnInput(), history(),
      linter(jsonParseLinter(), { delay: 200 }), lintGutter(),
      keymap.of([indentWithTab, ...defaultKeymap, ...historyKeymap]),
      EditorState.readOnly.of(readOnly), EditorView.editable.of(!readOnly),
      EditorView.contentAttributes.of({ 'aria-label': label }),
      EditorView.updateListener.of(update => { if (update.docChanged) change.current?.(update.state.doc.toString()); }),
    ] }) });
    editor.current = view;
    return () => { editor.current = null; view.destroy(); };
  }, [label, readOnly]);
  useEffect(() => {
    const view = editor.current;
    if (view && value !== view.state.doc.toString()) view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: value } });
  }, [value]);
  return <div className="json-editor" ref={host}/>;
}
