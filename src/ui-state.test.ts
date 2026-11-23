import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { sanitizeTabOrder, moveTab, apiKeyAccessSummary, apiKeyExpiry, apiKeyExpiryOptions, apiKeyModelOptions, ALL_GPUS, CHOOSE_GPUS, chosenGpus, deviceChoice, ratioError, splitLabel, toggleGpu, enablesKeyboardFocus, instancesOf, latestOnly, matchingProbe, settingsGroupMatches, panelReplace, playModelAfterLoad, quitText, reloadSlot } from './ui-state.ts';

const file = '/models/acme/qwen.gguf';
const first = { id: 'a1', slot: 1, path: file };
const second = { id: 'b2', slot: 2, path: file };
const other = { id: 'c1', slot: 1, path: '/models/acme/other.gguf' };
const loaded = [second, other, first];

test('a reload passes the replaced instance slot; fresh loads and Load another instance pass null', () => {
  assert.equal(reloadSlot(loaded, 'b2'), 2, 'Reload of instance 2 keeps slot 2');
  assert.equal(reloadSlot(loaded, 'a1'), 1);
  assert.equal(reloadSlot(loaded, 'c1'), 1, 'slots are per file');
  assert.equal(reloadSlot(loaded, null), null, 'Load model and Load another instance take the lowest free slot');
  assert.equal(reloadSlot(loaded, 'gone'), null, 'an instance that stopped on its own meanwhile has no slot to keep');
  assert.equal(reloadSlot([], 'b2'), null);
});

test('the Instance picker lists the selected file by slot and defaults to the lowest', () => {
  const { list, selected } = instancesOf(loaded, file, null);
  assert.deepEqual(list.map(i => i.id), ['a1', 'b2']);
  assert.equal(selected?.id, 'a1');
  assert.equal(instancesOf(loaded, file, 'b2').selected?.id, 'b2', 'the chosen instance, also the newest load');
  assert.equal(instancesOf(loaded, file, 'c1').selected?.id, 'a1', 'a choice from another file falls back to the lowest slot');
  assert.equal(instancesOf(loaded, file, 'gone').selected?.id, 'a1', 'an unloaded choice falls back too');
  assert.deepEqual(instancesOf(loaded, '/models/none.gguf', 'a1'), { list: [], selected: null });
});

test('the Playground keeps its model through a load unless that model was replaced or ejected', () => {
  assert.equal(playModelAfterLoad('c1', 'new', loaded, null), 'c1');
  assert.equal(playModelAfterLoad(null, 'new', loaded, null), 'new');
  assert.equal(playModelAfterLoad('b2', 'new', loaded, 'b2'), 'new', 'reloading the Playground model moves to its replacement');
  assert.equal(playModelAfterLoad('c1', 'new', loaded, null, 'c1'), 'new', 'ejected by the crowded-load dialog');
  assert.equal(playModelAfterLoad('gone', 'new', loaded, null), 'new');
});

// As runtime_devices returns a `--list-devices` line.
const rtx = { name: 'CUDA0', description: 'Example GPU', total_mib: 8192, free_mib: 7168 };
const secondGpu = { name: 'CUDA1', description: 'Second example GPU', total_mib: 11910, free_mib: 11000 };

test('the Device select lists every GPU plus CPU and defaults to the first GPU', () => {
  const choice = deviceChoice('', [rtx, secondGpu]);
  assert.deepEqual(choice.options, [
    { value: 'CUDA0', label: 'Example GPU', detail: 'CUDA0 · 8 GB' },
    { value: 'CUDA1', label: 'Second example GPU', detail: 'CUDA1 · 12 GB' },
    { value: ALL_GPUS, label: 'All GPUs', detail: 'All GPUs (2) · 20 GB' },
    { value: CHOOSE_GPUS, label: 'Choose GPUs…', detail: '2 or more' },
    { value: 'CPU', label: 'CPU', detail: 'CPU runtime' },
  ]);
  assert.equal(choice.effective, 'CUDA0');
  assert.equal(deviceChoice('CUDA1', [rtx, secondGpu]).effective, 'CUDA1');
  assert.equal(deviceChoice('', []).effective, 'CPU', 'no GPU, or a runtime that cannot list devices, defaults to CPU');
  assert.deepEqual(deviceChoice('', []).options, [{ value: 'CPU', label: 'CPU', detail: 'CPU runtime' }]);
});

test('a saved device the runtime does not report stays chosen and is marked not found', () => {
  const choice = deviceChoice('CUDA1', [rtx]);
  assert.equal(choice.missing, true);
  assert.equal(choice.value, 'CUDA1', 'never silently replaced by another device');
  assert.deepEqual(choice.options[0], { value: 'CUDA1', label: 'CUDA1 (not found)' });
  assert.equal(deviceChoice('CPU', [rtx]).missing, false);
  // Before the list is read nothing is marked missing, and an empty value waits for it.
  assert.equal(deviceChoice('CUDA1', null).missing, false);
  assert.equal(deviceChoice('', null).effective, null);
  assert.deepEqual(deviceChoice('', null).options[0], { value: '', label: 'Checking devices…' });
});

test('the fit badge leaves out the picked instance on a reload and none on a fresh load', () => {
  // The picker chose instance 2; Reload replaces it, so the fit and the slot both follow it.
  const picked = instancesOf(loaded, file, 'b2').selected;
  assert.equal(panelReplace(picked, true), 'b2');
  assert.equal(reloadSlot(loaded, panelReplace(picked, true)), 2);
  assert.equal(panelReplace(picked, false), null, 'loaded with these settings: the only load is Load another instance');
  assert.equal(panelReplace(null, false), null, 'Load model and Load beside replace nothing');
});

const pause = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

test('fit requests wait for a pause in typing, run one at a time and apply only the newest', async () => {
  const sent: number[] = [];
  const applied: Array<number | string> = [];
  const show = (result: { ok: true; value: number } | { ok: false; error: unknown }) => { applied.push(result.ok ? result.value : String(result.error)); };
  let release: (() => void) | null = null;
  const requests = latestOnly<number, number>(20, async n => { sent.push(n); if (n === 2) await new Promise<void>(r => { release = r; }); return n * 10; }, show);
  requests.request(1); requests.request(2);
  await pause(40);
  assert.deepEqual(sent, [2], 'typing 1 then 2 quickly sends only 2');
  // 3 and 4 arrive while 2 is running: they wait, and only 4 is sent after it.
  requests.request(3); requests.request(4);
  await pause(40);
  assert.deepEqual(sent, [2], 'one request at a time');
  release!();
  await pause(10);
  assert.deepEqual(sent, [2, 4]);
  assert.deepEqual(applied, [40], 'the superseded result for 2 is never shown');
  const failing = latestOnly<number, number>(0, async () => { throw 'Device CUDA3 was not found'; }, show);
  failing.request(1);
  await pause(10);
  assert.equal(applied.at(-1), 'Device CUDA3 was not found', 'the reason reaches the badge');
  requests.request(5); requests.cancel();
  await pause(40);
  assert.deepEqual(sent, [2, 4], 'cancelled before it was sent');
});

test('a newer debounce discards arguments already waiting behind a running fit', async () => {
  const sent: string[] = [];
  const applied: string[] = [];
  let finishFirst!: () => void;
  const first = new Promise<void>(resolve => { finishFirst = resolve; });
  const fit = latestOnly(20, async (arg: string) => {
    sent.push(arg);
    if (arg === 'A') await first;
    return arg;
  }, result => { if (result.ok) applied.push(result.value); });
  fit.request('A');
  await pause(40);
  fit.request('B');
  await pause(40);
  fit.request('C');
  finishFirst();
  await pause(40);
  assert.deepEqual(sent, ['A', 'C']);
  assert.deepEqual(applied, ['C']);
});

test('settings saved during editing only clear matching draft groups', () => {
  const before = { runtime_path: '/example/old', gpu_layers: 20 };
  const submitted = { ...before, runtime_path: '/example/new' };
  const fields = ['runtime_path'] as const;
  assert.equal(settingsGroupMatches(before, before, fields, true), false);
  assert.equal(settingsGroupMatches(before, submitted, fields, false), false, 'the edit back to the old path stays dirty');
  assert.equal(settingsGroupMatches({ ...submitted, gpu_layers: 24 }, submitted, fields, false), true);
});

test('a runtime probe is shown only for both inputs it checked', () => {
  const result = { path: '/example/build', libraryPath: '/example/lib', output: 'Example output' };
  assert.equal(matchingProbe(result, result.path, result.libraryPath), result.output);
  assert.equal(matchingProbe(result, '/example/other', result.libraryPath), null);
  assert.equal(matchingProbe(result, result.path, ''), null);
});

test('choosing a main runtime edits the Runtime path only', () => {
  const app = readFileSync(new URL('./main.tsx', import.meta.url), 'utf8');
  assert.match(app, /onSelect=\{runtime => edit\('runtime_path', runtime\.path\)\}/);
});

test('selected runtime copy controls reveal on keyboard focus', () => {
  const css = readFileSync(new URL('./styles.css', import.meta.url), 'utf8');
  assert.match(css, /\.keyboard-nav \.runtime-current \.copy-button:focus[^{}]*\{ opacity: 1; \}/);
});

test('the quit prompt counts the loaded models', () => {
  assert.equal(quitText(1), 'The local API and the loaded model will stop. Active downloads will also stop.');
  assert.equal(quitText(3), 'The local API and all 3 loaded models will stop. Active downloads will also stop.');
  assert.equal(quitText(0), 'The local API will stop. Active downloads will also stop.');
  assert.equal(quitText(null), 'The local API and any loaded models will stop. Active downloads will also stop.');
  const dialog = readFileSync(new URL('./QuitDialog.tsx', import.meta.url), 'utf8');
  assert.match(dialog, /<p id="quit-description">\{quitText\(loaded\)\}<\/p>/, 'the dialog shows the counted text');
});

test('API key expiry supports each preset and the end of a custom local day', () => {
  const now = new Date(2028, 1, 28, 12).getTime();
  assert.equal(apiKeyExpiry('never', '', now), null);
  assert.deepEqual(apiKeyExpiryOptions.map(option => option.value), ['never', '1', '7', '30', '90', '365', 'custom']);
  for (const days of [1, 7, 30, 90, 365]) {
    assert.equal(apiKeyExpiry(String(days), '', now), now / 1000 + days * 86400);
  }
  assert.equal(apiKeyExpiry('custom', '2028-02-29', now), new Date(2028, 2, 1).getTime() / 1000);
  assert.throws(() => apiKeyExpiry('custom', '2028-02-30', now), /Choose an expiry date/);
  assert.throws(() => apiKeyExpiry('custom', '2028-02-27', now), /Choose today or a future date/);
});

test('API key access summaries show model identifiers and aliases at creation', () => {
  assert.equal(apiKeyAccessSummary(null), 'Any');
  assert.equal(apiKeyAccessSummary([
    { name: 'sorter', aliases: ['triage', 'tickets'] },
    { name: 'review', aliases: [] },
  ]), 'sorter (aliases at creation: triage, tickets) · review');
});

test('API key options show each Library alias with its model and retain its selected name', () => {
  const options = apiKeyModelOptions([
    { path: '/models/sorter.gguf', name: 'sorter', aliases: ['fast', 'triage'] },
    { path: '/models/other.gguf', name: 'other', aliases: [] },
  ]);
  assert.deepEqual(options.map(option => option.label), ['sorter', 'fast → sorter', 'triage → sorter', 'other']);
  assert.deepEqual(options[1].selection, { path: '/models/sorter.gguf', name: 'fast' });
  assert.equal(new Set(options.map(option => option.id)).size, options.length);
  const [other, moved] = apiKeyModelOptions([{ path: '/models/other.gguf', name: 'other', aliases: ['fast'] }]);
  assert.equal(other.id, options[3].id);
  assert.notEqual(moved.id, options[1].id);
  assert.deepEqual(apiKeyModelOptions([]), []);
});

test('focus rings start only with navigation or control activation', () => {
  const key = (key: string, modifiers = {}) => ({ key, altKey: false, ctrlKey: false, metaKey: false, ...modifiers });
  for (const name of ['Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End', 'PageUp', 'PageDown']) {
    assert.equal(enablesKeyboardFocus(key(name), false), true, name);
  }
  assert.equal(enablesKeyboardFocus(key('Tab', { shiftKey: true }), false), true);
  for (const name of ['Meta', 'Super', 'OS', 'Alt', 'Control', 'Shift', 'Escape', 'a']) {
    assert.equal(enablesKeyboardFocus(key(name), true), false, name);
  }
  for (const name of ['Enter', ' ']) {
    assert.equal(enablesKeyboardFocus(key(name), true), true, name);
    assert.equal(enablesKeyboardFocus(key(name), false), false, name);
  }
  for (const modifier of ['altKey', 'ctrlKey', 'metaKey']) {
    assert.equal(enablesKeyboardFocus(key('ArrowLeft', { [modifier]: true }), true), false, modifier);
  }
});

// Two made-up GPUs as runtime_devices lists them.
const gpuA = { name: 'CUDA0', description: 'Example GPU A', total_mib: 10240, free_mib: 9000 };
const gpuB = { name: 'CUDA1', description: 'Example GPU B', total_mib: 24576, free_mib: 23000 };
const twoGpus = [gpuA, gpuB];

test('the split options appear only with 2 or more GPUs listed', () => {
  const values = (devices: typeof twoGpus) => deviceChoice('', devices).options.map(o => o.value);
  assert.deepEqual(values(twoGpus), ['CUDA0', 'CUDA1', ALL_GPUS, CHOOSE_GPUS, 'CPU']);
  assert.deepEqual(values([gpuA]), ['CUDA0', 'CPU']);
  assert.equal(deviceChoice('', [gpuA]).canSplit, false);
});

test('All GPUs and a chosen set resolve to the listed GPUs in listed order', () => {
  const all = deviceChoice(ALL_GPUS, twoGpus);
  assert.equal(all.value, ALL_GPUS);
  assert.equal(all.effective, 'CUDA0,CUDA1', 'compared with the instance device for Reload');
  assert.deepEqual(all.split?.gpus, ['CUDA0', 'CUDA1']);
  assert.equal(all.problem, null);
  const chosen = deviceChoice('CUDA1,CUDA0', twoGpus);
  assert.equal(chosen.value, CHOOSE_GPUS);
  assert.equal(chosen.effective, 'CUDA0,CUDA1');
  // Single devices are unchanged.
  assert.equal(deviceChoice('CUDA1', twoGpus).split, null);
  assert.equal(deviceChoice('CUDA1', twoGpus).effective, 'CUDA1');
});

test('the GPU checks keep a set in listed order, and fewer than 2 block the load', () => {
  assert.equal(chosenGpus(['CUDA1', 'CUDA0'], twoGpus), 'CUDA0,CUDA1');
  assert.equal(chosenGpus(['CUDA1'], twoGpus), 'CUDA1,', 'one GPU stays a set');
  assert.equal(chosenGpus([], twoGpus), ',');
  assert.equal(toggleGpu('CUDA1,', 'CUDA0', twoGpus), 'CUDA0,CUDA1');
  assert.equal(toggleGpu('CUDA0,CUDA1', 'CUDA0', twoGpus), 'CUDA1,');
  const few = deviceChoice('CUDA1,', twoGpus);
  assert.equal(few.value, CHOOSE_GPUS);
  assert.equal(few.problem, 'Choose at least 2 GPUs to split a model.');
});

test('a split is refused, not changed, when its GPUs are not listed', () => {
  const gone = deviceChoice('CUDA0,CUDA2', twoGpus);
  assert.equal(gone.missing, true);
  assert.match(gone.problem ?? '', /^CUDA2 was not found/);
  assert.deepEqual(gone.split?.gpus, ['CUDA0'], 'CUDA2 is not swapped for CUDA1');
  const lone = deviceChoice(ALL_GPUS, [gpuA]);
  assert.match(lone.problem ?? '', /needs at least 2 GPUs, but the llama.cpp build in Runtime reports 1/);
  assert.deepEqual(lone.options[0], { value: ALL_GPUS, label: 'All GPUs (not available)' });
  assert.equal(deviceChoice(ALL_GPUS, null).problem, null, 'nothing is refused before the list is read');
  assert.equal(deviceChoice(ALL_GPUS, null).effective, null);
});

test('a split is shown by its GPUs or as All GPUs, with the summed memory', () => {
  assert.equal(splitLabel(['CUDA0', 'CUDA1'], false, twoGpus), 'CUDA0 + CUDA1 · 34 GB');
  assert.equal(splitLabel(['CUDA0', 'CUDA1'], true, twoGpus), 'All GPUs (2) · 34 GB');
  assert.equal(splitLabel(['CUDA0', 'CUDA1'], false, [gpuA, { ...gpuB, total_mib: null }]), 'CUDA0 + CUDA1', 'no total without every reading');
  assert.equal(deviceChoice('CUDA0,CUDA1', twoGpus).split?.label, 'CUDA0 + CUDA1 · 34 GB');
});

test('the split ratio takes one number of 0 or more per GPU', () => {
  assert.equal(ratioError('', 2), null, 'empty shares by free memory');
  assert.equal(ratioError(' 3, 1 ', 2), null);
  assert.equal(ratioError('0.5,1.5', 2), null);
  assert.equal(ratioError('3,1,1', 2), 'Split ratio needs one number per GPU: 2 GPUs are in use, but it has 3.');
  for (const bad of ['3,-1', '3,x', '3,,1', 'Infinity,1', '0x10,1', '1e39,1', '3e38,3e38', '1e-40,1']) assert.equal(ratioError(bad, 2), 'Split ratio takes numbers of 0 or more separated by commas, such as 3,1.', bad);
  assert.equal(ratioError('0,0', 2), 'Split ratio needs at least one number above 0.');
});

test('stored tab order drops unknown and duplicate entries and appends missing tabs in default order', () => {
  const defaults = ['playground', 'models', 'hub', 'settings'] as const;
  assert.deepEqual(sanitizeTabOrder(['hub', 'gone', null, 'hub', 'models'], defaults), ['hub', 'models', 'playground', 'settings']);
  assert.deepEqual(sanitizeTabOrder({ order: ['hub'] }, defaults), defaults);
  assert.deepEqual(sanitizeTabOrder(null, defaults), defaults);
});

test('moving a tab inserts it at the destination without mutating the original order', () => {
  const order = Object.freeze(['a', 'b', 'c', 'd']);
  assert.deepEqual(moveTab(order, 0, 2), ['b', 'c', 'a', 'd']);
  assert.deepEqual(moveTab(order, 3, 1), ['a', 'd', 'b', 'c']);
  assert.equal(moveTab(order, 0, -1), order);
  assert.equal(moveTab(order, 3, 4), order);
  assert.equal(moveTab(order, 1, 1), order);
});
