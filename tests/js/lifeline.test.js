// Unit tests for static/js/viz/lifeline.js (#183). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const L = require('../../static/js/viz/lifeline.js');

const data = {
    entity_type: ['AOPs', 'KEs', 'KEs', 'KEs', 'KERs'],
    status: ['added', 'removed', 'returned', 'removed', 'added'],
    presence: ['-x', 'x-', 'xx', 'x-', '-x'],
};

test('rows of one type, optionally one status, in data order', () => {
    assert.deepEqual(L.rowsFor(data, 'KEs', null), [1, 2, 3]);
    assert.deepEqual(L.rowsFor(data, 'KEs', 'removed'), [1, 3]);
    assert.deepEqual(L.rowsFor(data, 'Stressors', null), []);
});

test('a pixel span maps to whole rows, at least one', () => {
    assert.deepEqual(L.rowSpan(0, 420, 420, 2907), [0, 2907]);
    assert.deepEqual(L.rowSpan(42, 84, 420, 100), [10, 20]);
    assert.deepEqual(L.rowSpan(84, 42, 420, 100), [10, 20]);  // either drag direction
    assert.deepEqual(L.rowSpan(50, 50.1, 420, 10), [1, 2]);
});

test('keys move, resize and clear the selection, within bounds', () => {
    assert.deepEqual(L.moveWindow(null, 'ArrowDown', false, 100), [0, 20]);
    assert.deepEqual(L.moveWindow(null, 'End', false, 100), [80, 100]);
    assert.deepEqual(L.moveWindow(null, 'ArrowDown', false, 5), [0, 5]);
    assert.deepEqual(L.moveWindow([0, 20], 'ArrowDown', false, 100), [1, 21]);
    assert.deepEqual(L.moveWindow([0, 20], 'ArrowUp', false, 100), [0, 20]);
    assert.deepEqual(L.moveWindow([0, 20], 'ArrowDown', true, 100), [0, 21]);
    assert.deepEqual(L.moveWindow([0, 1], 'ArrowUp', true, 100), [0, 1]);
    assert.deepEqual(L.moveWindow([10, 30], 'PageDown', false, 100), [30, 50]);
    assert.deepEqual(L.moveWindow([90, 100], 'PageDown', false, 100), [90, 100]);
    assert.deepEqual(L.moveWindow([40, 60], 'Home', false, 100), [0, 20]);
    assert.equal(L.moveWindow([40, 60], 'Escape', false, 100), null);
    assert.equal(L.moveWindow([40, 60], 'a', false, 100), undefined);
});

test('presence strings paint one pixel per cell', () => {
    const px = L.fillPixels(new Uint8ClampedArray(2 * 2 * 4), ['x-', '-x'], [1, 2, 3], [9, 9, 9]);
    assert.deepEqual(Array.from(px), [1, 2, 3, 255, 9, 9, 9, 255, 9, 9, 9, 255, 1, 2, 3, 255]);
    assert.deepEqual(L.rgb('#29235C'), [41, 35, 92]);
});

test('the summary counts every status', () => {
    assert.equal(L.summary(data, 'KEs'), '3 KEs ever: 0 always present, 0 added, 1 returned, 2 removed.');
});

test('100K cells paint well under the render budget', () => {
    const rows = Array.from({ length: 2907 }, (_, i) => (i % 3 ? 'x' : '-').repeat(35));
    const t0 = performance.now();
    L.fillPixels(new Uint8ClampedArray(2907 * 35 * 4), rows, [1, 2, 3], [4, 5, 6]);
    assert.ok(performance.now() - t0 < 100);
});

test('only identifiers.org IRIs become links', () => {
    assert.equal(L.linkable('https://identifiers.org/aop.events/54'), true);
    assert.equal(L.linkable('javascript:alert(1)'), false);
    assert.equal(L.linkable('http://example.org/x'), false);
    assert.equal(L.linkable(undefined), false);
});
