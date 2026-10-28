import test from 'node:test';
import assert from 'node:assert/strict';
import * as stateHelpers from './presets.ts';
import { modelStatus } from './model-state.ts';
import { visibleBanner } from './ui-state.ts';
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
  assert.ok(visibleBanner(null, 'new failure', 'port occupied', true, false));
  assert.equal(visibleBanner(null, null, null, false, true), null);
});

test('selecting a new model never mislabels the running model', () => {
  const status = modelStatus('/models/new.gguf', { path:'/models/old.gguf', device:'CUDA0' }, true, true);
  assert.equal(status.loaded, '/models/old.gguf');
  assert.equal(status.selected, '/models/new.gguf');
  assert.equal(status.switching, true);
  assert.equal(modelStatus('/models/new.gguf', null, false, false).label, 'Selected · not loaded');
  assert.equal(modelStatus('/models/new.gguf', { path:'/models/new.gguf', device:'CPU' }, true, false).label, 'Loading');
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
  assert.throws(() => buildRequest('{}', '', 'text'));
  assert.throws(() => buildRequest('[]', '', 'text'));
  assert.throws(() => buildRequest('{"a":{}}', 'broken', 'json'));
});
