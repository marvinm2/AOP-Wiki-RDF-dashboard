// Browser tests for the AOP-AOP overlap (#185): the cluster picker over the
// matrix, a matrix cell click, the graph tab (Cytoscape loaded on demand),
// the threshold driving data and downloads, keyboard tabs, and the graph's
// "SVG (as shown)".
const { test, expect } = require('@playwright/test');
const { installStubs, overlapData, overlapMatrixHtml, pickVersion, VERSIONS } = require('./stubs');

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
    // The matrix as the server draws it, from the same rows as the data stub.
    await page.route((url) => url.pathname === '/api/plot/latest_aop_aop_overlap', (route) => {
        const u = new URL(route.request().url());
        return route.fulfill({ json: { success: true,
            html: overlapMatrixHtml(overlapData(u.searchParams.get('version'), u.searchParams.get('min_jaccard'))) } });
    });
    await page.goto(`/snapshot${query}`);
    const card = page.locator('#latest-aop-aop-overlap');
    await card.scrollIntoViewIfNeeded();
    await expect(card.locator('.overlap-cluster option')).toHaveCount(8);  // "All" + 7 clusters
    return { card, stats };
}

const matrixRange = (page) => page.$eval('.overlap-panel--matrix .js-plotly-plot',
    (gd) => ({ x: gd.layout.xaxis.range, y: gd.layout.yaxis.range, auto: gd.layout.xaxis.autorange }));

test('picking a cluster zooms the matrix to its block and lists its pairs', async ({ page }) => {
    const { card } = await open(page);
    const select = card.locator('.overlap-cluster');
    await expect(select.locator('option').nth(1)).toHaveText('Cluster 1 · 8 AOPs');
    await expect(select.locator('option').nth(7)).toHaveText('Cluster 7 · AOP 35 + AOP 36');
    await expect(card.locator('.overlap-hint')).toBeVisible();

    await select.selectOption('2');  // 6 AOPs at positions 8-13
    const table = card.locator('.overlap-rows');
    await expect(table.locator('caption')).toHaveText('Cluster 2: 6 AOPs, 6 pairs at or above the threshold');
    await expect(table.locator('tbody tr')).toHaveCount(6);
    await expect(table.locator('tbody a').first()).toHaveAttribute('href', /^https:\/\/identifiers\.org\/aop\/\d+$/);
    await expect(table.locator('tbody tr').first().locator('td').last()).toHaveText('yes');
    await expect.poll(() => matrixRange(page)).toEqual({ x: [7.5, 13.5], y: [13.5, 7.5], auto: false });
    await expect(card.locator('.overlap-live')).toHaveText('Cluster 2 · 6 AOPs: 6 pairs listed below.');

    await select.press('Escape');
    await expect(card.locator('.overlap-hint')).toBeVisible();
    await expect(select).toHaveValue('');
    await expect.poll(async () => (await matrixRange(page)).auto).toBe(true);
});

test('clicking a matrix cell picks its cluster', async ({ page }) => {
    const { card } = await open(page);
    // The cell of the pair at positions (8, 9): cluster 2.
    const point = await page.$eval('.overlap-panel--matrix .js-plotly-plot', (gd) => {
        const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis, box = gd.getBoundingClientRect();
        return { x: box.left + xa._offset + xa.d2p(gd.data[0].x[9]), y: box.top + ya._offset + ya.d2p(gd.data[0].y[8]) };
    });
    await page.mouse.click(point.x, point.y);
    await expect(card.locator('.overlap-cluster')).toHaveValue('2');
    await expect(card.locator('.overlap-rows caption')).toHaveText(/^Cluster 2: /);
});

test('the graph tab loads Cytoscape, draws the pairs and follows the picker', async ({ page }) => {
    const { card, stats } = await open(page);
    expect(await page.evaluate(() => typeof window.cytoscape)).toBe('undefined');  // not loaded until asked for
    await card.getByRole('tab', { name: 'Graph' }).click();
    await expect(page).toHaveURL(/overlap=graph/);
    await expect(card.getByRole('tab', { name: 'Graph' })).toHaveAttribute('aria-selected', 'true');
    await expect(card.locator('#overlap-panel-matrix')).toBeHidden();
    const graph = card.locator('.overlap-graph');
    await expect(graph.locator('canvas').first()).toBeVisible();
    await expect(graph).toHaveAttribute('aria-label', /^AOP-AOP overlap graph: 27 AOPs in 7 clusters, \d+ pairs\./);
    expect(stats.requests.some((u) => u.endsWith('/api/v2/data/latest_aop_aop_overlap'))).toBe(true);

    await card.locator('.overlap-cluster').selectOption('1');
    await expect(card.locator('.overlap-rows caption')).toHaveText(/^Cluster 1: 8 AOPs/);
    await graph.getByRole('button', { name: 'Zoom in' }).click();
    await graph.getByRole('button', { name: 'Fit' }).click();

    // Keyboard: left arrow on the focused tab goes back to the matrix.
    await card.getByRole('tab', { name: 'Graph' }).focus();
    await page.keyboard.press('ArrowLeft');
    await expect(card.getByRole('tab', { name: 'Matrix' })).toBeFocused();
    await expect(page).not.toHaveURL(/overlap=/);
    await expect(card.locator('#overlap-panel-matrix')).toBeVisible();
});

test('a shared ?overlap=graph link opens on the graph', async ({ page }) => {
    const { card } = await open(page, '?overlap=graph');
    await expect(card.locator('#overlap-panel-graph')).toBeVisible();
    await expect(card.locator('.overlap-graph canvas').first()).toBeVisible();
});

test('the threshold and version drive the data, the picker and the downloads', async ({ page }) => {
    const { card, stats } = await open(page);
    await card.locator('.overlap-cluster').selectOption('3');
    await page.locator('#aop-aop-threshold-slider').evaluate((el) => {
        el.value = '0.40';
        el.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await expect(page).toHaveURL(/min_jaccard=0\.4(&|$)/);
    await card.scrollIntoViewIfNeeded();
    await expect(card.locator('.overlap-cluster option')).toHaveCount(7);   // the last cluster dropped out
    await expect(card.locator('.overlap-cluster')).toHaveValue('');         // clusters renumber, so the pick clears
    expect(stats.requests.some((u) => u.includes('latest_aop_aop_overlap') && u.includes('min_jaccard=0.4'))).toBe(true);
    const csv = card.locator('.dropdown-menu a', { hasText: 'CSV' });
    await expect(csv).toHaveAttribute('href', '/download/latest/aop_aop_overlap?format=csv&min_jaccard=0.4');

    await pickVersion(page, VERSIONS[1]);
    await expect(csv).toHaveAttribute('href', `/download/latest/aop_aop_overlap?format=csv&min_jaccard=0.4&version=${VERSIONS[1]}`);
    await card.scrollIntoViewIfNeeded();
    await expect(card.locator('.overlap-cluster option').nth(7)).toHaveCount(0);
    await expect(card.locator('.overlap-cluster option').nth(6)).toHaveText('Cluster 6 · AOP 133 + AOP 134');
});

test('"SVG (as shown)" on the graph tab saves the drawn graph', async ({ page }) => {
    const { card } = await open(page, '?overlap=graph');
    await expect(card.locator('.overlap-graph canvas').first()).toBeVisible();
    await card.locator('.overlap-cluster').selectOption('1');
    await card.locator('.plot-header').getByRole('button', { name: /Download/ }).click();
    const item = card.locator('.plot-header').getByRole('button', { name: 'SVG (as shown)' });
    const [download] = await Promise.all([page.waitForEvent('download'), item.click()]);
    expect(download.suggestedFilename()).toBe('latest_aop_aop_overlap.svg');
    const text = require('fs').readFileSync(await download.path(), 'utf8');
    expect(text).toContain('<svg');
    expect((text.match(/<circle/g) || []).length).toBe(27);
    expect(text).toContain('AOP 10');
});
