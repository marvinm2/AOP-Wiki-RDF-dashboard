// Unit tests for static/js/viz/overlap.js (#185). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const O = require('../../static/js/viz/overlap.js');

// The rows tests/test_aop_overlap_matrix.py builds (frame()), as /api/v2/data sends them.
const data = {
    cluster_id: [1, 1, 1, 2],
    cluster_size: [3, 3, 3, 2],
    aop1: ['https://identifiers.org/aop/10', 'https://identifiers.org/aop/10', 'https://identifiers.org/aop/7', 'https://identifiers.org/aop/30'],
    aop1_id: ['AOP 10', 'AOP 10', 'AOP 7', 'AOP 30'],
    aop2: ['https://identifiers.org/aop/7', 'https://identifiers.org/aop/2', 'https://identifiers.org/aop/2', 'https://identifiers.org/aop/31'],
    aop2_id: ['AOP 7', 'AOP 2', 'AOP 2', 'AOP 31'],
    jaccard: [0.8, 0.8, 1.0, 0.5],
    identical_ke_set: [false, false, true, false],
    matrix_row: [0, 0, 1, 3],
    matrix_col: [1, 2, 2, 4],
};

test('clusters: blocks on the diagonal, AOPs in matrix order', () => {
    assert.deepEqual(O.clusters(data), [
        { id: 1, size: 3, pairs: 3, identical: 1, lo: 0, hi: 2, aops: ['AOP 10', 'AOP 7', 'AOP 2'] },
        { id: 2, size: 2, pairs: 1, identical: 0, lo: 3, hi: 4, aops: ['AOP 30', 'AOP 31'] },
    ]);
    assert.deepEqual(O.clusters({}), []);
});

test('colours follow the server rule: palette without magenta, pairs grey', () => {
    const list = [{ id: 1, size: 5 }, { id: 2, size: 2 }, { id: 3, size: 3 }];
    assert.deepEqual(O.colours(list, ['#29235C', '#E6007E', '#307BBF']), { 1: '#29235C', 2: '#B8B8B8', 3: '#307BBF' });
});

test('a cell or an AOP finds its cluster, either way round', () => {
    assert.equal(O.clusterOfPair(data, 'AOP 2', 'AOP 7'), 1);
    assert.equal(O.clusterOfPair(data, 'AOP 31', 'AOP 30'), 2);
    assert.equal(O.clusterOfPair(data, 'AOP 2', 'AOP 30'), null);
    assert.equal(O.clusterOfAop(data, 'AOP 31'), 2);
    assert.deepEqual(O.pairsOf(data, 1), [0, 1, 2]);
});

test('labels and the zoom ranges of a block', () => {
    const [big, pair] = O.clusters(data);
    assert.equal(O.optionLabel(big), 'Cluster 1 · 3 AOPs');
    assert.equal(O.optionLabel(pair), 'Cluster 2 · AOP 30 + AOP 31');
    assert.deepEqual(O.blockRanges(big), { x: [-0.5, 2.5], y: [2.5, -0.5] });
});

test('graph cells: shelves left to right, wrapping at the width', () => {
    const list = [{ id: 1, size: 25 }, { id: 2, size: 9 }, { id: 3, size: 2 }];
    const c = O.cells(list, 500);
    assert.equal(c[1].x, 0);
    assert.ok(c[2].x >= c[1].w || c[2].y > 0);
    // Nothing overlaps.
    const ids = Object.keys(c);
    ids.forEach((a) => ids.forEach((b) => {
        if (a >= b) return;
        const A = c[a], B = c[b];
        assert.ok(A.x + A.w <= B.x || B.x + B.w <= A.x || A.y + A.h <= B.y || B.y + B.h <= A.y, `${a} and ${b} overlap`);
    }));
});

test('tabs: arrows wrap, Home/End jump, other keys do nothing', () => {
    assert.equal(O.tabStep('ArrowRight', 1, 2), 0);
    assert.equal(O.tabStep('ArrowLeft', 0, 2), 1);
    assert.equal(O.tabStep('End', 0, 2), 1);
    assert.equal(O.tabStep('Home', 1, 2), 0);
    assert.equal(O.tabStep('Enter', 0, 2), undefined);
});
