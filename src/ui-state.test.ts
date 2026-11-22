import test from 'node:test';
import assert from 'node:assert/strict';
import { instancesOf, playModelAfterLoad, reloadSlot } from './ui-state.ts';

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
