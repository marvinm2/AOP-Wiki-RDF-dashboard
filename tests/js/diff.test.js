// Unit tests for static/js/viz/diff.js (#188). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const D = require('../../static/js/viz/diff.js');

const COLUMNS = ['Type', 'ID', 'Name', 'Change', 'Fields Changed', 'OECD Stage A', 'OECD Stage B', 'Link'];
const ROWS = [
    ['AOPs', '4', 'New liver AOP', 'Added', '', '', 'No Status', 'https://identifiers.org/aop/4'],
    ['AOPs', '1', 'Liver <b>fibrosis</b>', 'Edited', 'Description', 'Under Development', 'Under Review', 'https://identifiers.org/aop/1'],
    ['KEs', '10', 'Steatosis', 'Links only', 'Part of AOP', '', '', 'https://identifiers.org/aop.events/10'],
];
const payload = {
    meta: { columns: COLUMNS.map((name) => ({ name, type: 'string' })), rows: ROWS.length },
    data: Object.fromEntries(COLUMNS.map((c, i) => [c, ROWS.map((r) => r[i])])),
};

test('rows come back as objects in server order', () => {
    const rows = D.rowsOf(payload);
    assert.equal(rows.length, 3);
    assert.deepEqual(rows.map((r) => r.ID), ['4', '1', '10']);
    assert.equal(rows[2].Change, 'Links only');
});

test('filters combine type, change and text (name, or the exact ID)', () => {
    const rows = D.rowsOf(payload);
    assert.deepEqual(D.filterRows(rows, { type: 'AOPs' }).map((r) => r.ID), ['4', '1']);
    assert.deepEqual(D.filterRows(rows, { change: 'Edited' }).map((r) => r.ID), ['1']);
    assert.deepEqual(D.filterRows(rows, { text: 'LIVER' }).map((r) => r.ID), ['4', '1']);
    assert.deepEqual(D.filterRows(rows, { text: '10' }).map((r) => r.ID), ['10']);
    assert.deepEqual(D.filterRows(rows, { type: 'KEs', text: 'liver' }), []);
});

test('the filter choices follow the rows', () => {
    const rows = D.rowsOf(payload);
    assert.deepEqual(D.distinct(rows, 'Type'), ['AOPs', 'KEs']);
    assert.deepEqual(D.distinct(rows, 'Change'), ['Added', 'Edited', 'Links only']);
});

test('version A is any version older than B, newest first', () => {
    const versions = ['2026-10-01', '2026-07-01', '2026-04-01', '2025-01-01'];
    assert.deepEqual(D.againstOptions(versions, '2026-07-01'), ['2026-04-01', '2025-01-01']);
    assert.deepEqual(D.againstOptions(versions, '2025-01-01'), []);
});

test('the table escapes names, links each ID and shows stage moves', () => {
    const html = D.tableHtml(D.rowsOf(payload), 10);
    assert.match(html, /Liver &lt;b&gt;fibrosis&lt;\/b&gt;/);
    assert.match(html, /<a href="https:\/\/identifiers.org\/aop\/1" target="_blank" rel="noopener">1<\/a>/);
    assert.match(html, /Under Development → Under Review/);
    assert.match(html, /— → No Status/);
    assert.equal((D.tableHtml(D.rowsOf(payload), 2).match(/<tr>/g) || []).length, 3);   // header + 2
});
