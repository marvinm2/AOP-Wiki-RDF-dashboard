// Unit tests for static/js/viz/figure.js (#181). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const F = require('../../static/js/viz/figure.js');

function bdata(dtype, values) {
    const T = { f8: Float64Array, i4: Int32Array, u1: Uint8Array }[dtype];
    return { dtype, bdata: Buffer.from(new T(values).buffer).toString('base64') };
}

test('decodeArray reads Plotly typed-array specs and passes lists through', () => {
    assert.deepEqual(F.decodeArray(bdata('i4', [219, 227, 230])), [219, 227, 230]);
    assert.deepEqual(F.decodeArray(bdata('f8', [0.5, 1.25])), [0.5, 1.25]);
    assert.deepEqual(F.decodeArray([1, 2]), [1, 2]);
    assert.equal(F.decodeArray(null), null);
});

test('differences start at 0 and skip gaps', () => {
    assert.deepEqual(F.differences([219, 227, 230, 236]), [0, 8, 3, 6]);
    assert.deepEqual(F.differences([1, null, 4, 6]), [0, null, null, 2]);
});

test('relabelHover swaps only the y label', () => {
    assert.equal(F.relabelHover('Component=Process<br>version=%{x}<br>Count=%{y}<extra></extra>', 'Change'),
        'Component=Process<br>version=%{x}<br>Change=%{y}<extra></extra>');
    assert.equal(F.relabelHover('Count=%{y}', 'Change'), 'Change=%{y}');
});

const absolute = {
    data: [
        { type: 'scatter', mode: 'lines+markers', name: 'Average KEs per AOP', legendgroup: 'Average KEs per AOP',
          x: ['2024-01-01', '2024-04-01', '2024-07-01'], y: bdata('f8', [4.5, 4.75, 5.0]),
          line: { color: '#29235C' }, marker: { symbol: 'circle' },
          hovertemplate: 'Metric=Average KEs per AOP<br>version=%{x}<br>Average=%{y}<extra></extra>' },
    ],
    layout: { yaxis: { title: { text: 'Average' } }, title: { text: 'T' } },
};

test('deriveDelta relabels and differences without touching the input', () => {
    const before = JSON.stringify(absolute);
    const d = F.deriveDelta(absolute, { y_title: 'Δ Average', name_replace: ['Average ', 'Δ '] });
    assert.equal(JSON.stringify(absolute), before);
    assert.deepEqual(d.data[0].y, [0, 0.25, 0.25]);
    assert.equal(d.data[0].name, 'Δ KEs per AOP');
    assert.equal(d.data[0].legendgroup, 'Δ KEs per AOP');
    assert.equal(d.data[0].hovertemplate, 'Metric=Δ KEs per AOP<br>version=%{x}<br>Δ Average=%{y}<extra></extra>');
    assert.equal(d.layout.yaxis.title.text, 'Δ Average');
    assert.equal(d.layout.title.text, 'T');
    assert.equal(d.data[0].type, 'scatter');
});

test('deriveDelta draws bars in the line colour when asked', () => {
    const d = F.deriveDelta(absolute, { y_title: 'Change', bar: true });
    assert.equal(d.data[0].type, 'bar');
    assert.equal(d.data[0].mode, undefined);
    assert.equal(d.data[0].line, undefined);
    assert.equal(d.data[0].marker.color, '#29235C');
    assert.equal(d.data[0].marker.symbol, undefined);
    assert.equal(d.layout.barmode, 'relative');
});
