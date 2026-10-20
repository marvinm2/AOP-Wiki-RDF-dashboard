// Unit tests for static/js/viz/calendar.js (#187). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const C = require('../../static/js/viz/calendar.js');

// Two strips as the server draws them: customdata[weekday][week], '' = no day.
function strip(year, cells, axis) {
    const cd = Array.from({ length: 7 }, () => new Array(54).fill(''));
    cells.forEach(([r, c, text]) => { cd[r][c] = text; });
    return { type: 'heatmap', name: String(year), customdata: cd, xaxis: axis[0], yaxis: axis[1] };
}
const traces = [
    strip(2024, [[0, 52, 'Mon 2024-12-23: 0'], [1, 52, 'Tue 2024-12-24: 2'], [1, 53, 'Tue 2024-12-31: 1']], ['x', 'y']),
    // 2025-01-01 is a Wednesday; nothing drawn on 2025-01-02 (after the newest snapshot, say).
    strip(2025, [[2, 0, 'Wed 2025-01-01: 4'], [2, 1, 'Wed 2025-01-08: 0'], [4, 1, 'Fri 2025-01-10: 1']], ['x2', 'y2']),
];

test('days come from the cells of every strip, oldest first, with their subplot', () => {
    const days = C.daysOf(traces);
    assert.deepEqual(days.map((d) => d.date), ['2024-12-23', '2024-12-24', '2024-12-31', '2025-01-01', '2025-01-08', '2025-01-10']);
    assert.deepEqual(days[3], { date: '2025-01-01', count: 4, curve: 1, row: 2, col: 0, subplot: 'x2y2' });
});

test('arrows move a day or a week, across strips and over gaps', () => {
    const days = C.daysOf(traces);
    const at = (date) => days.findIndex((d) => d.date === date);
    assert.equal(C.calendarStep('ArrowDown', null, days), days.length - 1);
    assert.equal(C.calendarStep('ArrowDown', at('2024-12-31'), days), at('2025-01-01'));
    assert.equal(C.calendarStep('ArrowRight', at('2024-12-24'), days), at('2024-12-31'));
    assert.equal(C.calendarStep('ArrowRight', at('2025-01-01'), days), at('2025-01-08'));
    // 2025-01-09 is not drawn: down from the 8th goes on to the 10th.
    assert.equal(C.calendarStep('ArrowDown', at('2025-01-08'), days), at('2025-01-10'));
    assert.equal(C.calendarStep('ArrowLeft', at('2025-01-01'), days), at('2024-12-24'));
    assert.equal(C.calendarStep('ArrowDown', at('2025-01-10'), days), at('2025-01-10'));
    // A year back from 2025-01-01 is before the first drawn day: stay.
    assert.equal(C.calendarStep('PageUp', at('2025-01-01'), days), at('2025-01-01'));
    // A year on from 2023-12-25 is 2024-12-25, not drawn: the next drawn day.
    const later = C.daysOf([strip(2023, [[0, 51, 'Mon 2023-12-25: 0']], ['x3', 'y3']), ...traces]);
    assert.equal(later[C.calendarStep('PageDown', 0, later)].date, '2024-12-31');
    assert.equal(C.calendarStep('Home', 3, days), 0);
    assert.equal(C.calendarStep('Enter', 3, days), undefined);
});

test('a day is read out with its weekday and count', () => {
    const days = C.daysOf(traces);
    assert.equal(C.describeDay(days[1]), 'Tuesday 2024-12-24: 2 KERs created.');
    assert.equal(C.describeDay(days[0]), 'Monday 2024-12-23: no KERs created.');
    assert.equal(C.describeDay(days[5]), 'Friday 2025-01-10: 1 KER created.');
});
