// Unit tests for static/js/viz/bump.js (#187). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const B = require('../../static/js/viz/bump.js');

// Three systems over four snapshots; C is not ranked in column 1.
const TRACES = [
    { name: 'A', x: ['s0', 's1', 's2', 's3'], y: [2, 1, 1, 1], customdata: [[5, 'first snapshot', 3], [9, 'up 1 since the snapshot before', 2], [9, 'same rank as the snapshot before', 3], [12, 'same rank as the snapshot before', 3]] },
    { name: 'B', x: ['s0', 's1', 's2', 's3'], y: [1, 2, 3, 2], customdata: [[6, 'first snapshot', 3], [7, 'down 1 since the snapshot before', 2], [2, 'down 1 since the snapshot before', 3], [8, 'up 1 since the snapshot before', 3]] },
    { name: 'C', x: ['s0', 's1', 's2', 's3'], y: [3, null, 2, 3], customdata: [[1, 'first snapshot', 3], [0, 'not ranked: no AOPs', 0], [3, 'newly ranked', 3], [4, 'down 1 since the snapshot before', 3]] },
];
const LINES = B.linesOf(TRACES);

test('ranked at a snapshot: best rank first, unranked left out', () => {
    assert.deepEqual(B.rankedAt(LINES, 0).map((l) => l.name), ['B', 'A', 'C']);
    assert.deepEqual(B.rankedAt(LINES, 1).map((l) => l.name), ['A', 'B']);
});

test('the first key lands on rank 1 at the newest visible snapshot', () => {
    assert.deepEqual(B.bumpStep('ArrowRight', null, LINES, 0, 3), { curve: 0, col: 3 });
    assert.deepEqual(B.bumpStep('ArrowDown', null, LINES, 0, 1), { curve: 0, col: 1 });
    assert.equal(B.bumpStep('a', null, LINES, 0, 3), undefined);
});

test('up and down move a rank; left and right follow the line, skipping gaps', () => {
    assert.deepEqual(B.bumpStep('ArrowDown', { curve: 0, col: 3 }, LINES, 0, 3), { curve: 1, col: 3 });
    assert.deepEqual(B.bumpStep('ArrowUp', { curve: 0, col: 3 }, LINES, 0, 3), { curve: 0, col: 3 });
    assert.deepEqual(B.bumpStep('ArrowDown', { curve: 2, col: 3 }, LINES, 0, 3), { curve: 2, col: 3 });
    assert.deepEqual(B.bumpStep('ArrowLeft', { curve: 2, col: 2 }, LINES, 0, 3), { curve: 2, col: 0 });
    assert.deepEqual(B.bumpStep('Home', { curve: 2, col: 3 }, LINES, 1, 3), { curve: 2, col: 2 });
    assert.deepEqual(B.bumpStep('End', { curve: 1, col: 0 }, LINES, 0, 2), { curve: 1, col: 2 });
    // Outside the range the focus is pulled back in.
    assert.deepEqual(B.bumpStep('ArrowRight', { curve: 0, col: 3 }, LINES, 0, 1), { curve: 0, col: 1 });
});

test('read-out and highlight', () => {
    assert.equal(B.describePoint(TRACES[1], 1), 'B, s1: rank 2 of 2, 7 AOPs, down 1 since the snapshot before.');
    assert.equal(B.describePoint(TRACES[2], 1), 'C, s1: not ranked, no AOPs.');
    assert.deepEqual(B.opacities(['A', 'B'], null), [1, 1]);
    assert.deepEqual(B.opacities(['A', 'B'], 'Z'), [1, 1]);
    assert.deepEqual(B.opacities(['A', 'B'], 'B'), [0.15, 1]);
});
