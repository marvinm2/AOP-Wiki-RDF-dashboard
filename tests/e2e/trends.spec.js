// Browser tests for the trends page (#181): figures drawn from /api/v2/figure,
// delta views derived in the browser, and linked plots.
const { test, expect } = require('@playwright/test');
const { installStubs, VERSIONS } = require('./stubs');

async function drawn(page) {
    return page.$$eval('.lazy-plot[data-rendered-url]', (els) => els.map((el) => ({
        name: el.dataset.plotName, url: el.dataset.renderedUrl,
        ys: (el.querySelector('.js-plotly-plot') || {}).data?.map((t) => Array.from(t.y)) || null,
    })));
}

test.beforeEach(async ({ page }) => {
    // Tall enough that several trend plots are in view (they load lazily).
    await page.setViewportSize({ width: 1400, height: 2600 });
    page.errors = [];
    page.on('pageerror', (e) => page.errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') page.errors.push(m.text()); });
});

test('trend plots draw from figure JSON; deltas are derived without a request', async ({ page }) => {
    const stats = await installStubs(page);
    await page.goto('/trends');
    await expect(page.locator('.lazy-plot[data-plot-name="aop_entity_counts_absolute"] .js-plotly-plot')).toBeVisible();
    const abs = (await drawn(page)).find((d) => d.name === 'aop_entity_counts_absolute');
    expect(abs.url).toBe('/api/v2/figure/aop_entity_counts_absolute');

    await page.click('#toggle-delta');
    const delta = page.locator('.lazy-plot[data-plot-name="aop_entity_counts_delta"]');
    await expect(delta.locator('.js-plotly-plot')).toBeVisible();
    const d = (await drawn(page)).find((x) => x.name === 'aop_entity_counts_delta');
    // Derived from the absolute figure: y[i] - y[i-1], 0 first.
    expect(d.ys[0]).toEqual(abs.ys[0].map((v, i) => (i === 0 ? 0 : v - abs.ys[0][i - 1])));
    expect(await delta.locator('.js-plotly-plot').evaluate((gd) => gd.layout.yaxis.title.text)).toBe('Count');
    expect(stats.requests.filter((u) => /_delta(\?|$)/.test(u))).toEqual([]);
    expect(stats.requests.filter((u) => u.endsWith('/api/v2/figure/aop_entity_counts_absolute'))).toHaveLength(1);
    expect(page.errors).toEqual([]);
});

test('linked plots: crosshair, shared zoom without an echo loop, click to snapshot', async ({ page }) => {
    await installStubs(page);
    await page.goto('/trends');
    const plots = page.locator('.lazy-plot[data-rendered-url] .js-plotly-plot');
    await expect(plots.nth(1)).toBeVisible();

    // Crosshair: hovering a snapshot on one plot shows it on the others.
    const target = VERSIONS[2];
    await plots.nth(0).evaluate((gd, x) => gd.emit('plotly_hover', { points: [{ x }] }), target);
    await expect(plots.nth(1).locator('.hoverlayer .hovertext').first()).toBeVisible();

    // Zoom one; the others follow, and the page stays responsive (no relayout ping-pong).
    const range = [VERSIONS[4], VERSIONS[1]];
    await plots.nth(0).evaluate((gd, r) => window.Plotly.relayout(gd, { 'xaxis.range[0]': r[0], 'xaxis.range[1]': r[1] }), range);
    await expect.poll(() => plots.nth(1).evaluate((gd) => gd._fullLayout.xaxis.range.map((v) => String(v).slice(0, 10))))
        .toEqual(range);
    const t0 = Date.now();
    await page.evaluate(() => new Promise((r) => setTimeout(r, 50)));
    expect(Date.now() - t0).toBeLessThan(2000);

    // Click a snapshot: opens it on the Snapshot page.
    await plots.nth(0).evaluate((gd, x) => gd.emit('plotly_click', { points: [{ x }] }), target);
    await expect(page).toHaveURL(new RegExp(`/snapshot\\?version=${target}`));
    expect(page.errors.filter((e) => !/Failed to load resource/.test(e))).toEqual([]);
});
