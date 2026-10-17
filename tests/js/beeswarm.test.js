// Unit tests for static/js/viz/beeswarm.js (#186). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const B = require('../../static/js/viz/beeswarm.js');

// The rows tests/test_aop_completeness_beeswarm.py builds (frame()), as /api/v2/data sends them.
const A = 'https://identifiers.org/aop/';
const data = {
    version: ['2025-01-01', '2025-01-01', '2025-04-01', '2025-04-01'],
    aop: [`${A}7`, `${A}10`, `${A}2`, `${A}10`],
    aop_id: ['AOP 7', 'AOP 10', 'AOP 2', 'AOP 10'],
    aop_title: ['', 'Ten', 'Two', 'Ten'],
    completeness_optional: [50, 20, 40, 61.23],
    completeness_all: [70, 55, 65, 80],
    entity_count: [3, 5, 2, 7],
};

test('swarm offsets match the server (swarm_offsets in plots/trends_plots.py)', () => {
    const ys = B.swarmOffsets([10, 12, 12, 13, 30, 30.5, 31, 50, 10, 11], 4.5).map((y) => Math.round(y * 1e6) / 1e6);
    assert.deepEqual(ys, [0, 17.774964, -17.888544, 26.719236, 0, 8.9861, -8.944272, 0, 9, -8.944272]);
    assert.deepEqual(B.swarmOffsets([], 4), []);
});

test('the snapshot shown is the range end, or the newest', () => {
    const versions = B.versionsOf(data);
    assert.deepEqual(versions, ['2025-01-01', '2025-04-01']);
    assert.equal(B.shownVersion(versions, null), '2025-04-01');
    assert.equal(B.shownVersion(versions, '2025-01-01'), '2025-01-01');
    assert.equal(B.shownVersion(versions, '2025-03-01'), '2025-01-01');   // not in the data: the one before
    assert.equal(B.shownVersion(versions, '2024-01-01'), '2025-01-01');
    assert.equal(B.shownVersion([], null), null);
});

test('rows of a snapshot, lowest score first, and ranks from the top', () => {
    const rows = B.rowsOf(data, '2025-04-01', 'completeness_optional');
    assert.deepEqual(rows, [2, 3]);
    assert.equal(B.rankOf(rows, 3), 1);
    assert.equal(B.rankOf(rows, 2), 2);
    assert.equal(B.rankOf(rows, 0), null);
    assert.deepEqual(B.rowsOf(data, '2025-01-01', 'completeness_all'), [1, 0]);
});

test('keys step through the AOPs by score; Escape clears; others do nothing', () => {
    const rows = [5, 1, 9, 4];
    assert.equal(B.stepPick(rows, -1, 'ArrowRight'), 1);   // no pick yet: start at the median
    assert.equal(B.stepPick(rows, 1, 'ArrowRight'), 9);
    assert.equal(B.stepPick(rows, 4, 'ArrowRight'), 4);    // stays at the top
    assert.equal(B.stepPick(rows, 5, 'ArrowLeft'), 5);
    assert.equal(B.stepPick(rows, 9, 'Home'), 5);
    assert.equal(B.stepPick(rows, 9, 'End'), 4);
    assert.equal(B.stepPick(rows, 9, 'Escape'), null);
    assert.equal(B.stepPick(rows, 9, 'a'), undefined);
    assert.equal(B.stepPick([], -1, 'ArrowRight'), undefined);
});

test('an AOP by number, and its history oldest first', () => {
    assert.equal(B.aopByNumber(data, '10'), `${A}10`);
    assert.equal(B.aopByNumber(data, 99), null);
    assert.deepEqual(B.historyOf(data, `${A}10`), [1, 3]);
    assert.equal(B.median([3, 1, 2]), 2);
    assert.equal(B.median([1, 2, 3, 4]), 2.5);
    assert.equal(B.median([]), null);
});

test('the history lists the first snapshot, the changes and the one shown', () => {
    const d = {
        completeness_optional: [10, 10, 10, 20, 20, 20],
        completeness_all: [40, 40, 40, 50, 50, 55],
        entity_count: [3, 3, 3, 3, 3, 3],
    };
    const rows = [0, 1, 2, 3, 4, 5];
    assert.deepEqual(B.changesOf(d, rows, -1), [0, 3, 5]);
    assert.deepEqual(B.changesOf(d, rows, 1), [0, 1, 3, 5]);
    assert.deepEqual(B.changesOf(d, [], -1), []);
});
