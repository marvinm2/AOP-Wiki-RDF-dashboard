// Unit tests for static/js/viz/network-evolution.js (#189). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const N = require('../../static/js/viz/network-evolution.js');

const COLUMNS = ['Element', 'ID', 'Label', 'Source', 'Target', 'X', 'Y', 'First Seen', 'Last Seen', 'Presence'];
const ROWS = [
    ['KE', 'KE 1', 'Binding', '', '', 0, 0, '2025-01-01', '2025-07-01', 'xxx'],
    ['KE', 'KE 2', 'Inflammation', '', '', 10, 5, '2025-01-01', '2025-04-01', 'xx-'],
    ['KE', 'KE 3', 'Fibrosis', '', '', -5, 3, '2025-04-01', '2025-07-01', '-xx'],
    ['KER', 'KER 5', '', 'KE 1', 'KE 2', null, null, '2025-01-01', '2025-07-01', 'xxx'],
    ['KER', 'KER 6', '', 'KE 1', 'KE 3', null, null, '2025-04-01', '2025-07-01', '-xx'],
    ['KER', 'KER 9', '', 'KE 1', 'KE 99', null, null, '2025-07-01', '2025-07-01', '--x'],
];
const payload = {
    meta: { columns: COLUMNS.map((name) => ({ name })), rows: ROWS.length },
    data: Object.fromEntries(COLUMNS.map((c, i) => [c, ROWS.map((r) => r[i])])),
};
const VERSIONS = ['2025-01-01', '2025-04-01', '2025-07-01'];

test('an element is new, shown, gone or hidden at a snapshot', () => {
    assert.equal(N.stateAt('xx-', 0), 'shown');   // the first snapshot has nothing new
    assert.equal(N.stateAt('-xx', 1), 'new');
    assert.equal(N.stateAt('-xx', 2), 'shown');
    assert.equal(N.stateAt('xx-', 2), 'gone');
    assert.equal(N.stateAt('x--', 2), 'hidden');
});

test('an edge needs both its Key Events in the snapshot', () => {
    assert.equal(N.edgePresence('xxx', 'xxx', 'xx-'), 'xx-');
    assert.equal(N.edgePresence('-xx', 'xxx', '-xx'), '-xx');
});

test('elements: nodes at their positions, edges to unknown Key Events dropped', () => {
    const el = N.elementsOf(payload);
    assert.deepEqual(el.nodes.map((n) => n.id), ['KE 1', 'KE 2', 'KE 3']);
    assert.deepEqual(el.edges.map((e) => e.id), ['KER 5', 'KER 6']);
    assert.equal(el.edges[0].presence, 'xx-');   // KE 2 is gone in the last snapshot
    assert.deepEqual([el.nodes[1].x, el.nodes[1].y], [10, 5]);
});

test('the read-out counts what is shown, new and gone', () => {
    const el = N.elementsOf(payload);
    assert.equal(N.describe(el, VERSIONS, 0), '2025-01-01: 2 Key Events and 1 relationship.');
    assert.equal(N.describe(el, VERSIONS, 2),
        '2025-07-01: 2 Key Events and 1 relationship. Since 2025-04-01: 0 Key Events and 0 relationships new, '
        + '1 Key Event and 1 relationship gone.');
    assert.deepEqual(N.countsAt(el, 1).kes, { shown: 2, new: 1, gone: 0, hidden: 0 });
});
