// Unit tests for static/js/viz/status-journey.js (#187). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const J = require('../../static/js/viz/status-journey.js');
const { journeyFigure } = require('../e2e/status-journey-fixture.js');

const BANDS = J.bandsOf(journeyFigure().data[0]);

test('bands are ordered by step, then by the stages they join', () => {
    assert.deepEqual(BANDS.map((b) => [b.step, b.i]), [[0, 0], [0, 1], [1, 2], [1, 3], [1, 4]]);
    assert.equal(BANDS[2].stepLabel, '2024-01-01 to 2025-01-01');
});

test('keys: first lands on the first band, then up/down in a step and left/right across', () => {
    assert.equal(J.journeyStep('ArrowRight', null, BANDS), 0);
    assert.equal(J.journeyStep('End', null, BANDS), 2);
    assert.equal(J.journeyStep('ArrowDown', 0, BANDS), 1);
    assert.equal(J.journeyStep('ArrowDown', 1, BANDS), 1);
    assert.equal(J.journeyStep('ArrowUp', 2, BANDS), 2);
    assert.equal(J.journeyStep('ArrowRight', 1, BANDS), 2);
    assert.equal(J.journeyStep('ArrowRight', 4, BANDS), 4);
    assert.equal(J.journeyStep('ArrowLeft', 4, BANDS), 0);
    assert.equal(J.journeyStep('Home', 3, BANDS), 0);
    assert.equal(J.journeyStep('x', 3, BANDS), undefined);
});

test('read-out and focus opacity', () => {
    assert.equal(J.describeBand(BANDS, 2),
        '2024-01-01 to 2025-01-01, band 1 of 3: 2 AOPs moved: Under Development (2024-01-01) to No Status (2025-01-01).');
    assert.deepEqual(J.focusOpacities([{ pointNumber: 0, alpha: 0.12 }, { pointNumber: 1, alpha: 0.4 }], 1), ['0.12', '0.85']);
    assert.deepEqual(J.focusOpacities([{ pointNumber: 0, alpha: 0.12 }], -1), ['0.12']);
});

test('a UMD bundle is captured without a global, and define is put back', async () => {
    const win = { define: 'theirs' };
    const lib = await J.captureUmd(win, (onload) => {
        // What a UMD bundle does when it finds an AMD loader.
        if (typeof win.define === 'function' && win.define.amd) win.define([], () => ({ version: 'full' }));
        onload();
    });
    assert.deepEqual(lib, { version: 'full' });
    assert.equal(win.define, 'theirs');

    const bare = {};
    await assert.rejects(J.captureUmd(bare, (onload, onerror) => onerror()));
    assert.equal('define' in bare, false);
});
