import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { deviceChoice, instancesOf, latestOnly, panelReplace, playModelAfterLoad, quitText, reloadSlot } from './ui-state.ts';

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

// As runtime_devices returns the real `--list-devices` line of this machine's build.
const rtx = { name: 'CUDA0', description: 'NVIDIA GeForce RTX 4060 Laptop GPU', total_mib: 7857, free_mib: 7742 };
const second3060 = { name: 'CUDA1', description: 'NVIDIA GeForce RTX 3060', total_mib: 11910, free_mib: 11000 };

test('the Device select lists every GPU plus CPU and defaults to the first GPU', () => {
  const choice = deviceChoice('', [rtx, second3060]);
  assert.deepEqual(choice.options, [
    { value: 'CUDA0', label: 'CUDA0 · NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)' },
    { value: 'CUDA1', label: 'CUDA1 · NVIDIA GeForce RTX 3060 (12 GB)' },
    { value: 'CPU', label: 'CPU' },
  ]);
  assert.equal(choice.effective, 'CUDA0');
  assert.equal(deviceChoice('CUDA1', [rtx, second3060]).effective, 'CUDA1');
  assert.equal(deviceChoice('', []).effective, 'CPU', 'no GPU, or a runtime that cannot list devices, defaults to CPU');
  assert.deepEqual(deviceChoice('', []).options, [{ value: 'CPU', label: 'CPU' }]);
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

test('the quit prompt counts the loaded models', () => {
  assert.equal(quitText(1), 'The local API and the loaded model will stop. Active downloads will also stop.');
  assert.equal(quitText(3), 'The local API and all 3 loaded models will stop. Active downloads will also stop.');
  assert.equal(quitText(0), 'The local API will stop. Active downloads will also stop.');
  assert.equal(quitText(null), 'The local API and any loaded models will stop. Active downloads will also stop.');
  const dialog = readFileSync(new URL('./QuitDialog.tsx', import.meta.url), 'utf8');
  assert.match(dialog, /<p id="quit-description">\{quitText\(loaded\)\}<\/p>/, 'the dialog shows the counted text');
});
