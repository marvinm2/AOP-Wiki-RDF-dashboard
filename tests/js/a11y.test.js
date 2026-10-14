// Unit tests for static/js/viz/a11y.js (#181). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const A = require('../../static/js/viz/a11y.js');

const traces = [
    { type: 'scatter', name: 'AOPs', x: ['2025-01-01', '2025-04-01', '2025-07-01'], y: [500, 505.5, 512] },
    { type: 'scatter', name: 'KEs', x: ['2025-04-01', '2025-07-01'], y: new Float64Array([1480, 1490]) },
    { type: 'scatter', name: 'hidden', visible: 'legendonly', x: ['2030-01-01'], y: [1] },
];

test('categories come from the shown traces, in order, without duplicates', () => {
    assert.deepEqual(A.categoriesOf(traces), ['2025-01-01', '2025-04-01', '2025-07-01']);
});

test('only the categories inside a clamped or zoomed axis are stepped through', () => {
    const cats = ['2018-04-01', '2020-01-01', '2026-01-01', '2026-10-01'];
    // A date axis clamped to 2020-01-01 .. 2026-01-01 (with Plotly's half-day padding).
    const date = { type: 'date', autorange: false, range: ['2019-12-31T12:00:00.000Z', '2026-01-01T12:00:00.000Z'] };
    assert.deepEqual(A.visibleCategories(cats, date), ['2020-01-01', '2026-01-01']);
    assert.deepEqual(A.visibleCategories(cats, { type: 'date', autorange: false, range: ['2019-12-31 12:00', '2026-01-01 12:00'] }),
        ['2020-01-01', '2026-01-01']);
    assert.deepEqual(A.visibleCategories(['a', 'b', 'c', 'd'], { type: 'category', autorange: false, range: [0.5, 2.5] }), ['b', 'c']);
    assert.deepEqual(A.visibleCategories(cats, { type: 'date', autorange: true, range: date.range }), cats);
    assert.deepEqual(A.visibleCategories(cats, undefined), cats);
});

test('values at a category: one entry per trace that has it, typed arrays included', () => {
    assert.deepEqual(A.valuesAt(traces, '2025-04-01'), [
        { curveNumber: 0, pointNumber: 1, name: 'AOPs', value: 505.5 },
        { curveNumber: 1, pointNumber: 0, name: 'KEs', value: 1480 },
    ]);
    assert.deepEqual(A.valuesAt(traces, '2025-01-01').map((p) => p.name), ['AOPs']);
});

test('horizontal bars and pies step through their categories, not their values', () => {
    const h = [{ type: 'bar', orientation: 'h', name: 'n', x: [3, 9], y: ['GO', 'CHEBI'] }];
    assert.deepEqual(A.categoriesOf(h), ['GO', 'CHEBI']);
    assert.equal(A.valuesAt(h, 'CHEBI')[0].value, 9);
    const pie = [{ type: 'pie', labels: ['Endorsed', 'Under review'], values: [40, 60] }];
    assert.equal(A.describeStep('Under review', A.valuesAt(pie, 'Under review')), 'Under review: 60.');
});

test('the spoken step names each series and caps long lists', () => {
    assert.equal(A.describeStep('2025-04-01', A.valuesAt(traces, '2025-04-01')), '2025-04-01: AOPs 505.5, KEs 1480.');
    const many = Array.from({ length: 10 }, (_, i) => ({ name: `s${i}`, value: i }));
    assert.match(A.describeStep('x', many), /s7 7, and 2 more\.$/);
    assert.equal(A.describeStep('x', []), 'x: no values.');
});

test('the table shows the first rows, escapes cells and says what it left out', () => {
    const payload = {
        meta: { rows: 3, columns: [{ name: 'term' }, { name: 'count' }] },
        data: { term: ['<b>GO</b>', 'PR', 'MP'], count: [3, null, 1] },
    };
    const html = A.tableHtml(payload, 'Data behind: Terms', 2);
    assert.match(html, /<caption class="sr-only">Data behind: Terms<\/caption>/);
    assert.match(html, /<th scope="col">term<\/th>/);
    assert.match(html, /&lt;b&gt;GO&lt;\/b&gt;/);
    assert.equal((html.match(/<tr>/g) || []).length, 3);  // header + 2 rows
    assert.match(html, /Showing first 2 of 3 rows/);
    assert.doesNotMatch(A.tableHtml(payload, '', 100), /Showing first/);
});

test('motionMs is the given duration when no reduced-motion preference is visible', () => {
    assert.equal(A.motionMs(500), 500);
});
