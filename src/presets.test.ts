import test from 'node:test';
import assert from 'node:assert/strict';
import * as stateHelpers from './presets.ts';
import { modelStatus } from './model-state.ts';
import { visibleBanner } from './ui-state.ts';
const { buildRequest, presets } = stateHelpers;

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
