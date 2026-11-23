import test from 'node:test';
import assert from 'node:assert/strict';
import * as stateHelpers from './presets.ts';
import { modelStatus, verifyingText } from './model-state.ts';
import { visibleBanner, waitText } from './ui-state.ts';
import { runtimeStatus } from './runtime-state.ts';
import { authorImageSources } from './author-image.ts';
const { buildRequest, presets } = stateHelpers;

test('author logos use the original upload with the existing avatar as fallback', () => {
  const avatar = 'https://cdn-avatars.huggingface.co/v1/production/uploads/owner/logo.png';
  assert.deepEqual(authorImageSources(avatar), { primary: 'https://cdn-uploads.huggingface.co/production/uploads/owner/logo.png', fallback: avatar });
  const other = 'https://example.com/logo.png';
  assert.deepEqual(authorImageSources(other), { primary: other, fallback: other });
});

test('runtime selection keeps the running custom build distinct from a pending download', () => {
  const selected = '/data/runtimes/b11968-cpu/package-0/build/llama-server';
  const active = '/custom/llama-b11954/llama-server';
  const running = runtimeStatus(selected, active, true, '/data');
  assert.equal(running.path, active);
  assert.equal(running.source, 'Custom');
  assert.equal(running.name, 'llama-b11954');
  assert.equal(running.pending, true);
  const stopped = runtimeStatus(selected, active, false, '/data');
  assert.equal(stopped.source, 'Downloaded');
  assert.equal(stopped.name, 'b11968-cpu');
  assert.equal(stopped.inUse, false);
  assert.equal(runtimeStatus('/custom/llama-b11954', active, true, '/data').pending, false);
  assert.equal(runtimeStatus('/data/runtimes-other/llama-server', null, false, '/data').source, 'Custom');
});

test('persistent API banners can be dismissed and new errors are still shown', () => {
  assert.equal(visibleBanner(null, 'port occupied', null, true, false)?.error, true);
  assert.equal(visibleBanner(null, 'port occupied', 'port occupied', true, false), null);
  assert.equal(visibleBanner(null, 'new failure', 'port occupied', true, false)?.text, 'Local API unavailable: new failure. Change its port in Runtime and restart the app.');
  assert.equal(visibleBanner(null, null, null, false, true), null);
});

test('the selected model reports only its own load state', () => {
  // Another model loaded beside it does not make the selection look loaded.
  assert.equal(modelStatus('/models/new.gguf', null).label, 'Selected · not loaded');
  const loaded = modelStatus('/models/new.gguf', { path:'/models/new.gguf', device:'CUDA0', ready:true });
  assert.equal(loaded.label, 'Loaded');
  assert.equal(loaded.device, 'CUDA0');
  const gone = modelStatus('/models/gone.gguf', null, true);
  assert.equal(gone.label, 'File not found');
  assert.equal(gone.missing, true);
  assert.equal(modelStatus('', null, true).missing, false);
  assert.equal(modelStatus('/models/new.gguf', { path:'/models/new.gguf', device:'CPU', ready:false }).label, 'Loading');
});

test('every preset produces native decision questions', () => {
  for (const preset of presets) {
    const request = buildRequest(JSON.stringify(preset.questions), preset.state, 'text');
    assert.equal(request.state, preset.state);
    for (const question of Object.values(preset.questions)) {
      assert.ok(['choice', 'score', 'noul'].includes(question.type));
      assert.ok(question.instructions);
      if (question.type === 'choice') assert.ok(Object.keys(question.criteria!).length >= 2);
    }
  }
});

test('switching text to JSON preserves its exact inference state', () => {
  const convert = (stateHelpers as any).convertState;
  assert.equal(typeof convert, 'function');
  for (const text of ['create a diagram', 'line one\n"quoted" \\ path', '', '{"keep":"as text"}', 'null']) {
    const json = convert(text, 'text', 'json');
    assert.equal(buildRequest('{"a":{}}', json, 'json').state, text);
    assert.equal(convert(json, 'json', 'text'), text);
  }
});

test('structured JSON survives a round trip through text without changing its type', () => {
  const convert = (stateHelpers as any).convertState;
  assert.equal(typeof convert, 'function');
  const source = '{"items":[1,true,null],"nested":{"text":"hi"}}';
  const text = convert(source, 'json', 'text');
  const restored = convert(text, 'text', 'json', source, text);
  assert.deepEqual(JSON.parse(restored), JSON.parse(source));
  assert.throws(() => convert('{invalid', 'json', 'text'), /JSON/);
});
test('text is kept verbatim while JSON state is parsed explicitly', () => {
  assert.equal(buildRequest('{"a":{}}', '{"x":1}', 'text').state, '{"x":1}');
  assert.deepEqual(buildRequest('{"a":{}}', '{"x":1}', 'json').state, { x: 1 });
  assert.throws(() => buildRequest('{}', '', 'text'), /at least one named question/);
  assert.throws(() => buildRequest('[]', '', 'text'), /at least one named question/);
  assert.throws(() => buildRequest('{"a":{}}', 'broken', 'json'), SyntaxError);
});

test('pending actions isolate unrelated controls and reject duplicate submissions', async () => {
  const { PendingActions } = await import('./refresh-state.ts');
  const pending = new PendingActions();
  assert.equal(pending.begin('memory-pref'), true);
  assert.equal(pending.has('start'), false);
  assert.equal(pending.begin('memory-pref'), false);
  assert.equal(pending.begin('search'), true);
  pending.end('memory-pref');
  assert.equal(pending.has('search'), true);
  assert.equal(pending.has('memory-pref'), false);
});

test('unchanged polling retains displayed state and stale replies cannot overwrite newer state', async () => {
  const { retainEqual, LatestRequest } = await import('./refresh-state.ts');
  const old = { settings: { allow_force_fit: false }, instances: [] };
  assert.equal(retainEqual(old, structuredClone(old)), old);
  const next = { ...old, settings: { allow_force_fit: true } };
  assert.equal(retainEqual(old, next), next);
  const requests = new LatestRequest();
  const slow = requests.begin();
  const fresh = requests.begin();
  assert.equal(requests.isLatest(slow), false);
  assert.equal(requests.isLatest(fresh), true);
});

test('the any-available wait reads as seconds when whole, as the 503 message does', () => {
  assert.deepEqual([1, 999, 1000, 1500, 60000, 120000, 600000].map(waitText), ['1 ms', '999 ms', '1 second', '1500 ms', '60 seconds', '120 seconds', '600 seconds']);
});

test('a load that is checking sha256 says so, with progress', () => {
  assert.equal(verifyingText(null), null);
  assert.equal(verifyingText({ file:'/models/m.gguf', slot:'model', done_bytes:42, total_bytes:100 }), 'Checking sha256… 42%');
  assert.equal(verifyingText({ file:'/models/h.gguf', slot:'mtp', done_bytes:0, total_bytes:100 }), 'Checking sha256 of head… 0%');
  assert.equal(verifyingText({ file:'/models/p.gguf', slot:'mmproj', done_bytes:100, total_bytes:100 }), 'Checking sha256 of projector… 100%');
  assert.equal(verifyingText({ file:'/models/e.gguf', slot:'model', done_bytes:0, total_bytes:0 }), 'Checking sha256…');
  const loading = { path:'/models/m.gguf', device:'CPU', ready:false, verifying:{ file:'/models/m.gguf', slot:'model' as const, done_bytes:1, total_bytes:4 } };
  assert.equal(modelStatus('/models/m.gguf', loading).label, 'Checking sha256… 25%');
  assert.equal(modelStatus('/models/m.gguf', { ...loading, verifying:null }).label, 'Loading');
});
