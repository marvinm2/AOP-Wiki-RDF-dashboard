// Browser tests for the organ-system bump chart (#187): the Highlight select in
// the URL, and the keyboard path along a line and between ranks.
const { test, expect } = require('@playwright/test');
const { installStubs, VERSIONS } = require('./stubs');

const X = VERSIONS.slice().reverse();  // oldest first, as the server sends it
const N = X.length;
// Nervous leads until Hepatobiliary passes it in the last snapshot; Sensory
// has no AOPs in the second snapshot.
const RANKS = {
    Hepatobiliary: X.map((_, i) => (i === N - 1 ? 1 : 2)),
    Nervous: X.map((_, i) => (i === N - 1 ? 2 : 1)),
    Sensory: X.map((_, i) => (i === 1 ? null : 3)),
};

function bump() {
    return {
        data: Object.entries(RANKS).map(([name, y]) => ({
            type: 'scatter', mode: 'lines+markers', name, x: X, y,
            customdata: y.map((r, i) => (r == null ? [0, 'not ranked: no AOPs', 0]
                : [10 * (4 - r) + i, i === 0 ? 'first snapshot' : 'same rank as the snapshot before', i === 1 ? 2 : 3])),
        })),
        layout: { title: { text: 'bump' }, height: 360, yaxis: { autorange: false, range: [3.5, 0.5] } },
    };
}

test.beforeEach(async ({ page }) => {
    page.errors = [];
    page.on('pageerror', (e) => page.errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') page.errors.push(m.text()); });
});

test.afterEach(async ({ page }) => {
    expect(page.errors).toEqual([]);
});

async function open(page, query = '') {
    await installStubs(page);
    await page.route(/\/api\/v2\/figure\/organ_coverage_bump/, (route) => {
        route.fulfill({ json: { meta: { id: 'organ_coverage_bump' }, figure: bump() } });
    });
    await page.goto(`/trends${query}`);
    const card = page.locator('#organ-bump');
    await card.scrollIntoViewIfNeeded();
    await expect(card.locator('.gtitle')).toHaveText('bump');
    return card;
}

const opacities = (card) => card.locator('.js-plotly-plot').evaluate((gd) => gd.data.map((t) => t.opacity));

test('Highlight dims the other lines and keeps the pick in the URL', async ({ page }) => {
    const card = await open(page);
    const select = card.locator('#organ-bump-highlight');
    await expect(select.locator('option')).toHaveText(['All organ systems', 'Hepatobiliary', 'Nervous', 'Sensory']);
    expect(await opacities(card)).toEqual([1, 1, 1]);

    await select.selectOption('Nervous');
    await expect(page).toHaveURL(/[?&]organ=Nervous(&|$)/);
    await expect.poll(() => opacities(card)).toEqual([0.15, 1, 0.15]);

    await select.selectOption('');
    await expect(page).not.toHaveURL(/organ=/);
    await expect.poll(() => opacities(card)).toEqual([1, 1, 1]);
});

test('a shared URL opens with its system highlighted', async ({ page }) => {
    const card = await open(page, '?organ=Sensory');
    await expect(card.locator('#organ-bump-highlight')).toHaveValue('Sensory');
    await expect.poll(() => opacities(card)).toEqual([0.15, 0.15, 1]);
});

test('the keyboard follows a line and moves between ranks', async ({ page }) => {
    const card = await open(page);
    const view = card.locator('.lazy-plot');
    await expect(view).toHaveAttribute('role', 'figure');
    const live = card.locator('.bump-live');

    await view.focus();
    await view.press('ArrowRight');
    await expect(live).toHaveText(`Hepatobiliary, ${X[N - 1]}: rank 1 of 3, ${30 + N - 1} AOPs, same rank as the snapshot before.`);
    await view.press('ArrowDown');
    await expect(live).toContainText(`Nervous, ${X[N - 1]}: rank 2 of 3`);
    await view.press('ArrowLeft');
    await expect(live).toContainText(`Nervous, ${X[N - 2]}: rank 1 of 3`);
    await expect(view.locator('.hoverlayer .hovertext')).toHaveCount(1);

    // Sensory is not ranked in the second snapshot, so Left skips over it.
    await view.press('ArrowDown');
    await view.press('ArrowDown');
    await expect(live).toContainText(`Sensory, ${X[N - 2]}: rank 3`);
    await view.press('Home');
    await view.press('ArrowRight');
    await expect(live).toContainText(`Sensory, ${X[2]}: rank 3`);

    await view.press('Escape');
    await expect(live).toHaveText('Point cleared.');
    await view.press('ArrowRight');
    await Promise.all([page.waitForURL(/\/snapshot\?version=/), view.press('Enter')]);
    expect(new URL(page.url()).searchParams.get('version')).toBe(X[N - 1]);
});
