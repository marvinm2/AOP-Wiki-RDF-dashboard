// Unit tests for static/js/viz/property-heatmap.js (#187). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const H = require('../../static/js/viz/property-heatmap.js');

test('the props state: AOPs are the default and stay out of the URL', () => {
    assert.equal(H.entityOf(null), 'aop');
    assert.equal(H.entityOf('nonsense'), 'aop');
    assert.equal(H.entityOf('ker'), 'ker');
    assert.equal(H.stateValue('aop'), null);
    assert.equal(H.stateValue('stressor'), 'stressor');
});

test('cells: the first key lands in the corner, then moves within bounds', () => {
    assert.deepEqual(H.cellStep('ArrowRight', null, 4, 2, 9), [0, 2]);
    assert.deepEqual(H.cellStep('End', null, 4, 2, 9), [0, 9]);
    assert.deepEqual(H.cellStep('ArrowDown', [3, 5], 4, 2, 9), [3, 5]);
    assert.deepEqual(H.cellStep('ArrowUp', [0, 5], 4, 2, 9), [0, 5]);
    assert.deepEqual(H.cellStep('ArrowLeft', [1, 2], 4, 2, 9), [1, 2]);
    assert.deepEqual(H.cellStep('Home', [1, 7], 4, 2, 9), [1, 2]);
    // A range change can leave the cell outside: it is pulled back in.
    assert.deepEqual(H.cellStep('ArrowDown', [1, 12], 4, 2, 9), [2, 9]);
    assert.equal(H.cellStep('Enter', [1, 1], 4, 2, 9), undefined);
    assert.equal(H.cellStep('ArrowDown', null, 0, 0, 3), undefined);
});

test('visible columns follow a date range, all of them when autoranged', () => {
    const xs = ['2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01'];
    assert.deepEqual(H.visibleColumns(xs, { autorange: true }), [0, 3]);
    assert.deepEqual(H.visibleColumns(xs, { autorange: false, range: ['2019-12-31 12:00', '2021-01-01 12:00'] }), [1, 2]);
    assert.deepEqual(H.visibleColumns(xs, { autorange: false, range: [Date.parse('2021-06-01'), Date.parse('2023-01-01')] }), [3, 3]);
    assert.deepEqual(H.visibleColumns(xs, { autorange: false, range: ['1990-01-01', '1991-01-01'] }), [0, 3]);
});

test('a cell is read out with its count and first snapshot', () => {
    const trace = {
        x: ['2019-01-01', '2020-01-01'], y: ['Title', 'Creator'],
        z: [[100, 100], [0, 12.345]],
        text: [['10 of 10', '20 of 20'], ['0 of 10', '2 of 20']],
        customdata: [[['Essential', '2019-01-01'], ['Essential', '2019-01-01']], [['Metadata', '2020-01-01'], ['Metadata', '2020-01-01']]],
    };
    assert.equal(H.describeCell(trace, 1, 1, 'aop'), 'Creator (Metadata), 2020-01-01: 12.3% of AOPs (2 of 20), first filled in 2020-01-01.');
    trace.customdata[1][0] = ['Metadata', 'never'];
    assert.match(H.describeCell(trace, 1, 0, 'ke'), /0% of KEs \(0 of 10\), never filled in\.$/);
});

test('download links switch entity and keep the range', () => {
    assert.equal(H.downloadHref('/download/trend/aop_property_heatmap?format=csv&start=2020-01-01', 'ker'),
        '/download/trend/ker_property_heatmap?format=csv&start=2020-01-01');
    assert.equal(H.tabStep('ArrowLeft', 0, 4), 3);
    assert.equal(H.tabStep('Enter', 0, 4), undefined);
});
