// Browser tests for the ontology coverage map (#187): click to zoom, the
// keyboard walk through the tree, and the organ system shared with the holes
// chart (both selectors, the URL and the download links).
const { test, expect } = require('@playwright/test');
const { installStubs } = require('./stubs');
const { coverageMapHtml } = require('./coverage-map-fixture');

test.beforeEach(async ({ page }) => {
    page.errors = [];
    page.on('pageerror', (e) => page.errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') page.errors.push(m.text()); });
});

test.afterEach(async ({ page }) => {
    expect(page.errors).toEqual([]);
});

async function open(page, query = '') {
    const stats = await installStubs(page);
    await page.route((url) => url.pathname === '/api/plot/latest_ontology_coverage_map', (route) => {
        stats.requests.push(route.request().url());
        return route.fulfill({ json: { success: true, html: coverageMapHtml() } });
    });
    await page.goto(`/snapshot${query}`);
    const card = page.locator('#latest-ontology-coverage-map');
    await card.scrollIntoViewIfNeeded();
    const view = card.locator('.lazy-plot');
    await expect(view).toHaveAttribute('role', 'figure');
    return { card, view, stats };
}

const xRange = (view) => view.locator('.js-plotly-plot').evaluate((gd) => gd._fullLayout.xaxis.range.map((v) => Math.round(v * 100) / 100));

test('clicking a block zooms to it, double-click zooms out', async ({ page }) => {
    const { card, view } = await open(page);
    await expect(view.locator('.point').first()).toBeVisible();  // the bars are drawn
    // Click the middle of "heart" (level 1, 0-50% of the width).
    const box = await view.locator('.js-plotly-plot .nsewdrag').first().boundingBox();
    const level1 = await view.locator('.js-plotly-plot').evaluate((gd) => {
        const y = gd._fullLayout.yaxis;
        return y._offset + y.l2p(1);
    });
    await page.mouse.click(box.x + box.width * 0.25, (await view.locator('.js-plotly-plot').boundingBox()).y + level1);
    await expect.poll(() => xRange(view)).toEqual([0, 50]);
    await expect(card.locator('.coverage-map-live')).toContainText('Zoomed to heart.');
    await page.mouse.dblclick(box.x + box.width * 0.25, (await view.locator('.js-plotly-plot').boundingBox()).y + level1);
    await expect.poll(() => xRange(view)).toEqual([0, 100]);
});

test('the keyboard walks the tree and zooms', async ({ page }) => {
    const { card, view } = await open(page);
    const live = card.locator('.coverage-map-live');
    await view.focus();
    await view.press('ArrowDown');
    await expect(live).toContainText('cardiovascular system (UBERON:0004535), the whole branch');
    await view.press('ArrowDown');
    await view.press('ArrowDown');
    await expect(live).toHaveText(/^cardiac chamber \(UBERON:0004151\), level 2/);
    await view.press('ArrowRight');
    await expect(live).toHaveText('cardiac valve (UBERON:0000946), level 2, a coverage hole: nothing in it is annotated. '
        + '30 terms in its subtree. 1 sub-branch below.');
    await expect(view.locator('.hoverlayer .hovertext')).toHaveCount(1);
    await view.press('Enter');
    await expect.poll(() => xRange(view)).toEqual([20, 35]);
    // Inside the zoom, Left/Right stay on the blocks it shows.
    await view.press('ArrowLeft');
    await expect(live).toHaveText(/^cardiac valve/);
    await view.press('ArrowUp');
    await expect(live).toHaveText(/^heart \(UBERON:0000948\), level 1/);
    await view.press('Escape');
    await expect(live).toHaveText('Zoomed out to the whole branch.');
    await expect.poll(() => xRange(view)).toEqual([0, 100]);
});

test('the organ system is one setting for both charts', async ({ page }) => {
    const { card, stats } = await open(page);
    await card.locator('#ontology-map-branch-select').selectOption('Nervous');
    await expect(page).toHaveURL(/[?&]branch=Nervous(&|$)/);
    await expect(page.locator('#ontology-branch-select')).toHaveValue('Nervous');
    await expect.poll(() => stats.requests.some((u) => u.includes('/api/plot/latest_ontology_coverage_map') && u.includes('branch=Nervous'))).toBe(true);
    await expect(card.locator('.download-dropdown a').first()).toHaveAttribute('href', /branch=Nervous/);
});

test('a shared URL opens on its organ system', async ({ page }) => {
    const { card } = await open(page, '?branch=Sensory');
    await expect(card.locator('#ontology-map-branch-select')).toHaveValue('Sensory');
    await expect(page.locator('#ontology-branch-select')).toHaveValue('Sensory');
});
