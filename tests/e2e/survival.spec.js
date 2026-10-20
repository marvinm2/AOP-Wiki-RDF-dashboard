// Browser tests for cohort survival (#187): its x axis is years since first
// seen, so neither the /trends range nor the shared zoom may touch it, and its
// own zoom must not reach the snapshot plots.
const { test, expect } = require('@playwright/test');
const { installStubs, VERSIONS } = require('./stubs');

const ASC = VERSIONS.slice().reverse();

function survivalFigure() {
    const x = [0, 0.25, 0.5, 0.75, 1];
    return {
        data: ['Already in 2018-04-01', 'First seen 2018–2019'].map((name, i) => ({
            type: 'scatter', mode: 'lines', name, x, y: x.map((t) => 100 - (i + 1) * 10 * t),
            line: { shape: 'hv' },
        })),
        layout: { height: 300, xaxis: { title: { text: 'Years since first seen' } } },
    };
}

test.beforeEach(async ({ page }) => {
    page.errors = [];
    page.on('pageerror', (e) => page.errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') page.errors.push(m.text()); });
});

test('the range and the shared zoom leave survival alone, and its zoom stays put', async ({ page }) => {
    await installStubs(page);
    await page.route(/\/api\/v2\/figure\/entity_survival/, (route) => route.fulfill({
        json: { meta: { id: 'entity_survival' }, figure: survivalFigure() },
    }));
    await page.goto(`/trends?start=${ASC[1]}&end=${ASC[4]}`);
    const counts = page.locator('.lazy-plot[data-plot-name="aop_entity_counts_absolute"] .js-plotly-plot');
    await expect.poll(() => counts.evaluate((gd) => gd.layout.xaxis.autorange)).toBe(false);
    const clamped = await counts.evaluate((gd) => gd._fullLayout.xaxis.range.map(String));

    const card = page.locator('#entity-survival');
    await card.scrollIntoViewIfNeeded();
    const survival = card.locator('.js-plotly-plot');
    await expect(survival).toBeVisible();
    await page.waitForTimeout(300);
    expect(await survival.evaluate((gd) => gd._fullLayout.xaxis.range.map(Number))).toEqual([0, 1]);

    // Zooming survival does not move the snapshot plots.
    await survival.evaluate((gd) => window.Plotly.relayout(gd, { 'xaxis.range[0]': 0.25, 'xaxis.range[1]': 0.75 }));
    await page.waitForTimeout(300);
    expect(await counts.evaluate((gd) => gd._fullLayout.xaxis.range.map(String))).toEqual(clamped);

    // Zooming a snapshot plot does not move survival.
    await counts.evaluate((gd, r) => window.Plotly.relayout(gd, { 'xaxis.range[0]': r[0], 'xaxis.range[1]': r[1] }),
        [ASC[2], ASC[3]]);
    await page.waitForTimeout(300);
    expect(await survival.evaluate((gd) => gd._fullLayout.xaxis.range.map(Number))).toEqual([0.25, 0.75]);
    expect(page.errors).toEqual([]);
});
