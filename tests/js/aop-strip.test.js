// AOP composition strip (#190): cell states, the read-out, the summary, the picker.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const S = require('../../static/js/viz/aop-strip.js');

const VERSIONS = ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'];
const COLUMNS = ['Element', 'ID', 'Title', 'Role', 'First Seen', 'Last Seen', 'Roles', 'Stages'];
const ROWS = [
    ['AOP', 'AOP 7', 'Liver fibrosis', 'AOP', VERSIONS[1], VERSIONS[3], '-xxx', '|Under Development|Under Development|No Status'],
    ['KE', 'KE 10', 'Binding', 'MIE', VERSIONS[1], VERSIONS[3], '-MMM', ''],
    ['KE', 'KE 20', '', 'KE', VERSIONS[1], VERSIONS[3], '-K-K', ''],
    ['KE', 'KE 30', 'Fibrosis', 'AO', VERSIONS[1], VERSIONS[3], '-AAA', ''],
];
const payload = {
    meta: { rows: ROWS.length },
    data: Object.fromEntries(COLUMNS.map((c, i) => [c, ROWS.map((r) => r[i])])),
};

test('rows come back as objects, the AOP first', () => {
    const rows = S.rowsOf(payload);
    assert.equal(rows[0].ID, 'AOP 7');
    assert.equal(rows[2].Roles, '-K-K');
});

test('the strip starts at the AOP\'s first version', () => {
    assert.equal(S.startOf('-xxx'), 1);
    assert.equal(S.startOf('xxxx'), 0);
});

test('cells know their role and whether the KE joined or left', () => {
    assert.deepEqual(S.cellState('-K-K', 1, 1), { role: 'KE', added: false, removed: false });   // the AOP's first version
    assert.deepEqual(S.cellState('-K-K', 2, 1), { role: null, added: false, removed: true });
    assert.deepEqual(S.cellState('-K-K', 3, 1), { role: 'KE', added: true, removed: false });
    assert.deepEqual(S.cellState('-MMM', 2, 1), { role: 'MIE', added: false, removed: false });
});

test('a cell is read out in words', () => {
    const [, mie, ke] = S.rowsOf(payload);
    assert.equal(S.describeCell(mie, '2024-04-01', S.cellState(mie.Roles, 1, 1)),
        'KE 10 Binding, 2024-04-01: molecular initiating event.');
    assert.equal(S.describeCell(ke, '2024-07-01', S.cellState(ke.Roles, 2, 1)),
        'KE 20, 2024-07-01: not in the AOP, left it in this version.');
    assert.equal(S.describeCell(ke, '2024-10-01', S.cellState(ke.Roles, 3, 1)),
        'KE 20, 2024-10-01: key event, joined the AOP in this version.');
});

test('the summary counts the Key Events, versions and changes', () => {
    assert.equal(S.summary(S.rowsOf(payload), VERSIONS),
        'AOP 7: Liver fibrosis. 3 Key Events in 3 versions from 2024-04-01 to 2024-10-01; 3 in 2024-10-01. '
        + 'After 2024-04-01: 1 joined, 1 left.');
});

test('the picker takes a number, "AOP n", a list entry or a unique title', () => {
    const index = [{ number: 3, title: 'Mitochondrial complex I inhibition' }, { number: 7, title: 'Liver fibrosis' },
        { number: 8, title: 'Liver steatosis' }];
    assert.equal(S.parseAop('3', index), 3);
    assert.equal(S.parseAop('AOP 12', index), 12);
    assert.equal(S.parseAop(S.optionText(index[1]), index), 7);
    assert.equal(S.parseAop('complex i', index), 3);
    assert.equal(S.parseAop('liver', index), null);    // two match
    assert.equal(S.parseAop('', index), null);
});
