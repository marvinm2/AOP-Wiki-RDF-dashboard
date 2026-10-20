// Browser tests for the KER curation calendar (#187): the keyboard walks the
// days across the year strips, and the /trends range leaves the chart alone.
const { test, expect } = require('@playwright/test');
const { installStubs, VERSIONS } = require('./stubs');

const ASC = VERSIONS.slice().reverse();

/** Two strips like the server's: Mon..Sun x 54 weeks, a few days drawn. */
function calendarFigure() {
    const strip = (year, cells, axes) => {
        const z = Array.from({ length: 7 }, () => new Array(54).fill(null));
        const cd = Array.from({ length: 7 }, () => new Array(54).fill(''));
        cells.forEach(([r, c, date, n]) => { z[r][c] = n; cd[r][c] = `${date}: ${n}`; });
        return { type: 'heatmap', name: String(year), z, customdata: cd, x: [...Array(54).keys()],
            y: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            xaxis: axes[0], yaxis: axes[1], hovertemplate: '%{customdata} KERs created<extra></extra>' };
    };
    return {
        data: [
            strip(2024, [[1, 52, '2024-12-24', 2], [1, 53, '2024-12-31', 1]], ['x', 'y']),
            strip(2025, [[2, 0, '2025-01-01', 4], [3, 0, '2025-01-02', 0]], ['x2', 'y2']),
        ],
        layout: {
            height: 320, grid: { rows: 2, columns: 1 },
            xaxis: { range: [-0.5, 53.5] }, xaxis2: { range: [-0.5, 53.5] },
            yaxis: { autorange: 'reversed', domain: [0.55, 1] }, yaxis2: { autorange: 'reversed', domain: [0, 0.45] },
        },
    };
}

test('arrow keys walk the days across the strips; the range does not touch it', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    await installStubs(page);
    await page.route(/\/api\/v2\/figure\/ker_curation_calendar/, (route) => route.fulfill({
        json: { meta: { id: 'ker_curation_calendar' }, figure: calendarFigure() },
    }));
    await page.goto(`/trends?start=${ASC[1]}&end=${ASC[4]}`);
    const card = page.locator('#ker-curation-calendar');
    await card.scrollIntoViewIfNeeded();
    const chart = card.locator('.lazy-plot');
    await expect(chart.locator('.js-plotly-plot')).toBeVisible();
    await expect(chart).toHaveAttribute('role', 'figure');
    await page.waitForTimeout(300);
    expect(await chart.locator('.js-plotly-plot').evaluate((gd) => gd._fullLayout.xaxis.range.map(Number))).toEqual([-0.5, 53.5]);

    const live = card.locator('.calendar-live');
    await chart.focus();
    await chart.press('End');
    await expect(live).toHaveText('Thursday 2025-01-02: no KERs created.');
    await chart.press('ArrowUp');
    await expect(live).toHaveText('Wednesday 2025-01-01: 4 KERs created.');
    await expect(chart.locator('.hoverlayer .hovertext')).toContainText('2025-01-01: 4 KERs created');
    await chart.press('ArrowLeft');
    await expect(live).toHaveText('Tuesday 2024-12-24: 2 KERs created.');
    await chart.press('Escape');
    await expect(live).toHaveText('Day cleared.');
    expect(errors).toEqual([]);
});
