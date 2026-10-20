// Unit tests for static/js/viz/race.js (#187). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const R = require('../../static/js/viz/race.js');

const KE = (n) => `https://identifiers.org/aop.events/${n}`;
// Columnar /api/v2/data: KE 2 overtakes KE 1; KE 5 enters in 2021; KE 1 drops out of the top.
const rows = [
    ['2020-01-01', 1, 9, 1], ['2020-01-01', 2, 8, 2], ['2020-01-01', 5, 0, null],
    ['2021-01-01', 1, 9, 16], ['2021-01-01', 2, 12, 1], ['2021-01-01', 5, 10, 2],
];
const data = {
    Version: rows.map((r) => r[0]), KE: rows.map((r) => KE(r[1])), 'KE ID': rows.map((r) => `KE ${r[1]}`),
    Title: rows.map((r) => (r[1] === 2 ? 'Increase,   Oxidative   Stress' : '')), AOPs: rows.map((r) => r[2]), Rank: rows.map((r) => r[3]),
};

test('the shown snapshot follows the range end', () => {
    const versions = R.versionsOf(data);
    assert.deepEqual(versions, ['2020-01-01', '2021-01-01']);
    assert.equal(R.shownVersion(versions, null), '2021-01-01');
    assert.equal(R.shownVersion(versions, '2020-07-01'), '2020-01-01');
    assert.equal(R.shownVersion(versions, '2019-01-01'), '2020-01-01');
});

test('bars keep one slot per KE; outside the top they wait below the chart', () => {
    const bars = R.barsAt(data, '2021-01-01');
    assert.deepEqual(bars.map((b) => [b.id, b.y, b.rank]), [['KE 1', R.OUT, null], ['KE 2', 1, 1], ['KE 5', 2, 2]]);
    assert.equal(R.barsAt(data, '2020-01-01')[2].y, R.OUT);
    assert.equal(R.label('KE 2', bars[1].title), 'KE 2 · Increase, Oxidative Stress');
    assert.equal(R.label('KE 9', 'x'.repeat(60)).length, 'KE 9 · '.length + 42);
});

test('up and down walk the ranks; Escape clears', () => {
    const bars = R.barsAt(data, '2021-01-01');
    assert.equal(R.stepPick(bars, null, 'ArrowDown'), KE(2));
    assert.equal(R.stepPick(bars, KE(2), 'ArrowDown'), KE(5));
    assert.equal(R.stepPick(bars, KE(5), 'ArrowDown'), KE(5));
    assert.equal(R.stepPick(bars, KE(5), 'ArrowUp'), KE(2));
    assert.equal(R.stepPick(bars, null, 'ArrowUp'), KE(5));
    assert.equal(R.stepPick(bars, KE(2), 'Escape'), null);
    assert.equal(R.stepPick(bars, KE(2), 'ArrowLeft'), undefined);
});

test('the pick is described with its change from the snapshot before', () => {
    const versions = R.versionsOf(data);
    assert.equal(R.describePick(data, versions, '2021-01-01', KE(2)),
        'KE 2, Increase, Oxidative Stress: rank 1 of 15 in 2021-01-01, in 12 AOPs, was 2 in 2020-01-01.');
    assert.equal(R.describePick(data, versions, '2021-01-01', KE(5)), 'KE 5: rank 2 of 15 in 2021-01-01, in 10 AOPs, new in the top 15.');
    assert.equal(R.describePick(data, versions, '2021-01-01', KE(1)), 'KE 1: not in the top 15 in 2021-01-01, in 9 AOPs, out of the top 15 (was 1).');
    assert.equal(R.describePick(data, versions, '2020-01-01', KE(1)), 'KE 1: rank 1 of 15 in 2020-01-01, in 9 AOPs.');
});
