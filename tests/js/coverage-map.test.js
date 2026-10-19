// Unit tests for static/js/viz/coverage-map.js (#187). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../../static/js/viz/coverage-map.js');
const { coverageMapTraces } = require('../e2e/coverage-map-fixture.js');

const nodes = () => M.buildIndex(coverageMapTraces());
const named = (ns, label) => ns.find((n) => n.label === label);

test('the index rebuilds the drawn tree from the traces', () => {
    const ns = nodes();
    assert.equal(ns.length, 8);
    const top = ns[0];
    assert.equal(top.level, 0);
    assert.deepEqual(top.children.map((c) => c.label), ['heart', 'blood vessel']);
    assert.deepEqual(named(ns, 'heart').children.map((c) => c.label), ['cardiac chamber', 'cardiac valve', 'lymph vessel']);
    assert.equal(named(ns, 'mitral valve').parent.label, 'cardiac valve');
    assert.equal(named(ns, 'blood vessel').parent, top);
});

test('keys walk the tree: down, up, along the level', () => {
    const ns = nodes();
    let at = M.step('ArrowDown', ns, null);
    assert.equal(at.level, 0);  // the first key lands on the branch
    at = M.step('ArrowDown', ns, at);
    assert.equal(at.label, 'heart');
    at = M.step('ArrowRight', ns, at);
    assert.equal(at.label, 'blood vessel');
    assert.equal(M.step('ArrowRight', ns, at).label, 'blood vessel');  // stays at the end
    assert.equal(M.step('ArrowDown', ns, at).label, 'blood vessel');   // no children: stays
    at = M.step('ArrowDown', ns, named(ns, 'cardiac valve'));
    assert.equal(at.label, 'mitral valve');
    assert.equal(M.step('ArrowUp', ns, at).label, 'cardiac valve');
    assert.equal(M.step('End', ns, named(ns, 'cardiac chamber')).label, 'lymph vessel');
    assert.equal(M.step('Home', ns, named(ns, 'lymph vessel')).label, 'cardiac chamber');
    assert.equal(M.step('ArrowUp', ns, ns[0]), ns[0]);
    assert.equal(M.step('Enter', ns, ns[0]), undefined);
});

test('a zoom limits the level to the blocks it shows', () => {
    const ns = nodes();
    const heart = named(ns, 'heart');
    const range = M.zoomRange(heart);
    assert.deepEqual(range, [0, 50]);
    assert.deepEqual(M.onLevel(ns, 1, range).map((n) => n.label), ['heart']);
    assert.equal(M.step('ArrowRight', ns, heart, range).label, 'heart');
    assert.deepEqual(M.zoomRange(ns[0]), [0, 100]);
});

test('a block is read out with its status and size', () => {
    const ns = nodes();
    assert.equal(M.describe(named(ns, 'cardiac valve')),
        'cardiac valve (UBERON:0000946), level 2, a coverage hole: nothing in it is annotated. 30 terms in its subtree. 1 sub-branch below.');
    assert.equal(M.describe(ns[0]),
        'cardiovascular system (UBERON:0004535), the whole branch, not annotated itself, but terms below it are. 200 terms in its subtree. 2 sub-branches below.');
    assert.match(M.describe(named(ns, 'cardiac ventricle')), /, level 3, annotated\. 10 terms in its subtree\.$/);
});
