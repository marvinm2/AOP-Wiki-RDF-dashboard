// Unit tests for static/js/viz/upset.js (#184). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const U = require('../../static/js/viz/upset.js');

// Same fixture as tests/test_organ_upset.py, so both sides order alike.
const data = {
    aop: ['https://identifiers.org/aop/2', 'https://identifiers.org/aop/3', 'https://identifiers.org/aop/4', 'https://identifiers.org/aop/10'],
    systems_all_kes: ['Hepatobiliary; Immune/Haematopoietic', 'Hepatobiliary; Immune/Haematopoietic', 'Nervous', 'Hepatobiliary'],
    systems_apical_kes: ['Hepatobiliary', '', 'Nervous', ''],
    systems_adverse_outcome: ['', '', 'Nervous', ''],
};

test('exclusive intersections, largest first, ties by fewer buckets then name', () => {
    const inter = U.intersections(data, 'systems_all_kes');
    assert.deepEqual(inter.map((c) => [c.key, c.members]), [
        [['Hepatobiliary', 'Immune/Haematopoietic'], [0, 1]],
        [['Hepatobiliary'], [3]],
        [['Nervous'], [2]],
    ]);
    assert.deepEqual(U.intersections(data, 'systems_adverse_outcome').map((c) => c.key), [['Nervous']]);
});

test('bucket totals count every combination a bucket is in', () => {
    assert.deepEqual(U.setSizes(data, 'systems_all_kes').map((s) => [s.name, s.members.length]),
        [['Hepatobiliary', 3], ['Immune/Haematopoietic', 2], ['Nervous', 1]]);
});

test('keys: left/right through columns, up/down through buckets, Escape clears', () => {
    assert.deepEqual(U.step(null, 'ArrowRight', 3, 2), { kind: 'combo', i: 0 });
    assert.deepEqual(U.step(null, 'ArrowLeft', 3, 2), { kind: 'combo', i: 2 });
    assert.deepEqual(U.step({ kind: 'combo', i: 2 }, 'ArrowRight', 3, 2), { kind: 'combo', i: 2 });
    assert.deepEqual(U.step({ kind: 'combo', i: 1 }, 'Home', 3, 2), { kind: 'combo', i: 0 });
    assert.deepEqual(U.step({ kind: 'combo', i: 1 }, 'End', 3, 2), { kind: 'combo', i: 2 });
    assert.deepEqual(U.step({ kind: 'combo', i: 1 }, 'ArrowDown', 3, 2), { kind: 'set', i: 0 });
    assert.deepEqual(U.step({ kind: 'set', i: 0 }, 'ArrowDown', 3, 2), { kind: 'set', i: 1 });
    assert.deepEqual(U.step({ kind: 'set', i: 0 }, 'ArrowUp', 3, 2), { kind: 'set', i: 0 });
    assert.equal(U.step({ kind: 'set', i: 0 }, 'Escape', 3, 2), null);
    assert.equal(U.step(null, 'a', 3, 2), undefined);
    assert.equal(U.step(null, 'ArrowRight', 0, 0), undefined);
});

test('summary and scope fallback', () => {
    assert.equal(U.summary(data, 'all'),
        '4 of 4 AOPs have an organ-system signal (All KEs), in 3 combinations; 0 have none.');
    assert.equal(U.summary(data, 'ao'),
        '1 of 4 AOPs have an organ-system signal (Adverse Outcome KE only), in 1 combination; 3 have none.');
    assert.equal(U.scopeOf('bogus'), 'all');
    assert.equal(U.scopeOf('apical'), 'apical');
});

test('only identifiers.org IRIs become links', () => {
    assert.equal(U.linkable('https://identifiers.org/aop/2'), true);
    assert.equal(U.linkable('javascript:alert(1)'), false);
});
