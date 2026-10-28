import { useEffect, useRef } from 'react';
import { EditorState } from '@codemirror/state';
import { EditorView, keymap, lineNumbers, highlightActiveLineGutter } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import { bracketMatching, foldGutter, HighlightStyle, indentOnInput, syntaxHighlighting } from '@codemirror/language';
import { json, jsonParseLinter } from '@codemirror/lang-json';
import { linter, lintGutter } from '@codemirror/lint';
import { tags } from '@lezer/highlight';

const colors = HighlightStyle.define([
  { tag: tags.propertyName, color: '#b9a8f0' },
  { tag: tags.string, color: '#a7b3ad' },
  { tag: tags.number, color: '#b9b199' },
  { tag: [tags.bool, tags.null], color: '#b9a8f0' },
  { tag: [tags.punctuation, tags.bracket], color: '#7d8995' },
]);
const theme = EditorView.theme({
  '&': { backgroundColor: '#0a0e14', color: '#adb6be', height: '100%', fontSize: '13px' },
  '.cm-scroller': { fontFamily: '"SFMono-Regular", Consolas, monospace', lineHeight: '1.7', overflow: 'auto', overscrollBehavior: 'contain' },
  '.cm-content': { padding: '12px 0', caretColor: '#c3cac4' },
  '.cm-line': { padding: '0 12px' },
  '.cm-gutters': { backgroundColor: '#0a0e14', color: '#596570', borderRight: '1px solid #202830' },
  '.cm-activeLineGutter': { backgroundColor: '#171d25' },
  '.cm-cursor': { borderLeftColor: '#c3cac4' },
  '&.cm-focused': { outline: 'none' },
  '&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection': { backgroundColor: '#2c3044' },
  '.cm-tooltip': { backgroundColor: '#171d25', color: '#c3cac4', border: '1px solid #2b333b' },
  '.cm-diagnostic': { fontSize: '11px' },
}, { dark: true });

export function JsonEditor({ value, onChange, label, readOnly = false }: { value: string; onChange?: (value: string) => void; label: string; readOnly?: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const editor = useRef<EditorView | null>(null);
  const change = useRef(onChange);
  change.current = onChange;
  useEffect(() => {
    if (!host.current) return;
    const view = new EditorView({ parent: host.current, state: EditorState.create({ doc: value, extensions: [
      json(), theme, syntaxHighlighting(colors), lineNumbers(), highlightActiveLineGutter(),
      bracketMatching(), foldGutter(), indentOnInput(), history(),
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
