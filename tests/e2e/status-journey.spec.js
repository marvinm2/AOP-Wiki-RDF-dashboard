// Browser tests for the OECD status journey (#187): the full Plotly bundle
// draws the sankey without replacing the page's cartesian Plotly, and the
// keyboard walks the bands.
const { test, expect } = require('@playwright/test');
const { installStubs } = require('./stubs');
const { journeyFigure } = require('./status-journey-fixture');

test.beforeEach(async ({ page }) => {
    page.errors = [];
    page.on('pageerror', (e) => page.errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') page.errors.push(m.text()); });
});

test.afterEach(async ({ page }) => {
    expect(page.errors).toEqual([]);
});

async function open(page) {
    await installStubs(page);
    await page.route(/\/api\/v2\/figure\/oecd_status_journey/, (route) => {
        route.fulfill({ json: { meta: { id: 'oecd_status_journey' }, figure: journeyFigure() } });
    });
    await page.goto('/trends');
    const card = page.locator('#oecd-status-journey');
    await card.scrollIntoViewIfNeeded();
    await expect(card.locator('.sankey-link')).toHaveCount(5);
    return card;
}

test('the sankey is drawn with the full bundle and the page keeps the cartesian one', async ({ page }) => {
    const card = await open(page);
    const libs = await page.evaluate(() => {
        const gd = document.querySelector('#oecd-status-journey .plotly-full-graph');
        return {
            define: typeof window.define,
            same: gd._dashPlotly === window.Plotly,
            pageHasSankey: !!(window.Plotly.PlotSchema.get().traces.sankey),
            fullHasSankey: !!(gd._dashPlotly.PlotSchema.get().traces.sankey),
        };
    });
    expect(libs).toEqual({ define: 'undefined', same: false, pageHasSankey: false, fullHasSankey: true });
    // The /trends range selector only drives .plotly-graph-div figures.
    await expect(card.locator('.plotly-graph-div')).toHaveCount(0);
    await expect(card.locator('.gtitle')).toHaveText('journey');
});

test('the keyboard walks the bands and darkens the focused one', async ({ page }) => {
    const card = await open(page);
    const view = card.locator('.lazy-plot');
    await expect(view).toHaveAttribute('role', 'figure');
    const live = card.locator('.journey-live');
    const opacity = (n) => card.locator('.sankey-link').evaluateAll(
        (paths, i) => paths.find((p) => p.__data__.pointNumber === i).style.fillOpacity, n);
    const before = await opacity(2);

    await view.focus();
    await view.press('ArrowRight');
    await expect(live).toHaveText('2023-01-01 to 2024-01-01, band 1 of 2: 3 AOPs stayed: Under Development (2023-01-01) to Under Development (2024-01-01).');
    await view.press('ArrowRight');
    await expect(live).toContainText('2024-01-01 to 2025-01-01, band 1 of 3: 2 AOPs moved');
    expect(await opacity(2)).toBe('0.85');
    await view.press('ArrowDown');
    await view.press('ArrowDown');
    await expect(live).toContainText('band 3 of 3: 1 AOP moved: Under Development (2024-01-01) to Under Review (2025-01-01)');
    expect(await opacity(2)).toBe(before);

    await view.press('Escape');
    await expect(live).toHaveText('Band cleared.');
    expect(await opacity(4)).toBe('0.4');
});
