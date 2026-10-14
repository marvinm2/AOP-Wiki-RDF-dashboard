// Unit tests for static/js/viz/scrubber.js (#182). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const S = require('../../static/js/viz/scrubber.js');

const versions = ['2023-10-01', '2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01', '2025-01-01'];
const times = versions.map((v) => Date.parse(`${v}T00:00:00Z`));

test('a time snaps to the nearest snapshot', () => {
    assert.equal(S.nearestIndex(times, Date.parse('2024-02-10T00:00:00Z')), 1);
    assert.equal(S.nearestIndex(times, Date.parse('2024-03-01T00:00:00Z')), 2);
    assert.equal(S.nearestIndex(times, Date.parse('2010-01-01T00:00:00Z')), 0);
    assert.equal(S.nearestIndex(times, Date.parse('2030-01-01T00:00:00Z')), 5);
});

test('keys step one snapshot, a year, or to the ends, and stay in bounds', () => {
    assert.equal(S.stepIndex(2, 'ArrowRight', 6), 3);
    assert.equal(S.stepIndex(2, 'ArrowDown', 6), 1);
    assert.equal(S.stepIndex(1, 'PageUp', 6), 5);
    assert.equal(S.stepIndex(1, 'PageDown', 6), 0);
    assert.equal(S.stepIndex(5, 'ArrowRight', 6), 5);
    assert.equal(S.stepIndex(3, 'Home', 6), 0);
    assert.equal(S.stepIndex(3, 'End', 6), 5);
    assert.equal(S.stepIndex(3, 'a', 6), null);
});

test('the whole range is the default; other ranges are ordered', () => {
    assert.deepEqual(S.normaliseRange(0, 5, versions), { start: null, end: null });
    assert.deepEqual(S.normaliseRange(4, 1, versions), { start: '2024-01-01', end: '2024-10-01' });
    assert.deepEqual(S.normaliseRange(2, 2, versions), { start: '2024-04-01', end: '2024-04-01' });
});

test('play in single mode walks to the latest, starting over if already there', () => {
    assert.deepEqual(S.playNext('single', { i: 2 }, 6, true), { i: 3 });
    assert.deepEqual(S.playNext('single', { i: 5 }, 6, true), { i: 0 });
    assert.deepEqual(S.playNext('single', { i: 4 }, 6, false), { i: 5 });
    assert.equal(S.playNext('single', { i: 5 }, 6, false), null);
});

test('play in range mode grows the end from the start', () => {
    // Whole range: start over from the first snapshot alone.
    assert.deepEqual(S.playNext('range', { i0: 0, i1: 5 }, 6, true), { i0: 0, i1: 0 });
    // A window that ends at the latest: restart at its start.
    assert.deepEqual(S.playNext('range', { i0: 2, i1: 5 }, 6, true), { i0: 2, i1: 2 });
    // A window short of the latest: extend it.
    assert.deepEqual(S.playNext('range', { i0: 1, i1: 3 }, 6, true), { i0: 1, i1: 4 });
    assert.deepEqual(S.playNext('range', { i0: 1, i1: 4 }, 6, false), { i0: 1, i1: 5 });
    assert.equal(S.playNext('range', { i0: 1, i1: 5 }, 6, false), null);
});
