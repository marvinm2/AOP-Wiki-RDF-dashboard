// Browser tests for the property x snapshot heatmap (#187): entity tabs in the
// URL, download links that follow the tab, and the cell-by-cell keyboard path
// limited to the /trends range.
const { test, expect } = require('@playwright/test');
const { installStubs, VERSIONS } = require('./stubs');

const X = VERSIONS.slice().reverse();  // oldest first, as the server sends it
const ROWS = ['Title', 'Description', 'Creator'];

/** A heatmap figure like the server's, labelled with the entity it is for. */
function heatmap(entity) {
    return {
        data: [{
            type: 'heatmap', x: X, y: ROWS, zmin: 0, zmax: 100,
            z: ROWS.map((_, r) => X.map((_, c) => (r === 2 && c < 2 ? 0 : 10 * (c + 1)))),
            text: ROWS.map(() => X.map((_, c) => `${c + 1} of 10`)),
            customdata: ROWS.map((_, r) => X.map(() => ['Essential', r === 2 ? X[2] : X[0]])),
        }],
        layout: { title: { text: `${entity} heatmap` }, height: 220, yaxis: { autorange: 'reversed' } },
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
    await page.route(/\/api\/v2\/figure\/[a-z]+_property_heatmap/, (route) => {
        const entity = new URL(route.request().url()).pathname.split('/').pop().split('_')[0];
        route.fulfill({ json: { meta: { id: `${entity}_property_heatmap` }, figure: heatmap(entity) } });
    });
    await page.goto(`/trends${query}`);
    const card = page.locator('#property-heatmap');
    await card.scrollIntoViewIfNeeded();
    return card;
}

const plotIn = (card, entity) => card.locator(`.lazy-plot[data-plot-name="${entity}_property_heatmap"]`);
const titleOf = (container) => container.locator('.gtitle');

test('tabs switch entity, keep it in the URL and move the download links', async ({ page }) => {
    const card = await open(page);
    await expect(titleOf(plotIn(card, 'aop'))).toHaveText('aop heatmap');
    await expect(plotIn(card, 'ke')).toBeHidden();

    await card.getByRole('tab', { name: 'KEs' }).click();
    await expect(page).toHaveURL(/[?&]props=ke(&|$)/);
    await expect(card.getByRole('tab', { name: 'KEs' })).toHaveAttribute('aria-selected', 'true');
    await expect(titleOf(plotIn(card, 'ke'))).toHaveText('ke heatmap');
    await expect(plotIn(card, 'aop')).toBeHidden();
    const hrefs = await card.locator('.props-downloads a').evaluateAll((as) => as.map((a) => a.getAttribute('href')));
    expect(hrefs).toEqual(['csv', 'png', 'svg'].map((f) => `/download/trend/ke_property_heatmap?format=${f}`));

    // Arrow keys on the tab list; AOPs again drops the key from the URL.
    await card.getByRole('tab', { name: 'KEs' }).press('ArrowLeft');
    await expect(card.getByRole('tab', { name: 'AOPs' })).toBeFocused();
    await expect(page).not.toHaveURL(/props=/);
});

test('a shared URL opens on its tab', async ({ page }) => {
    const card = await open(page, '?props=stressor');
    await expect(card.getByRole('tab', { name: 'Stressors' })).toHaveAttribute('aria-selected', 'true');
    await expect(titleOf(plotIn(card, 'stressor'))).toHaveText('stressor heatmap');
    await expect(plotIn(card, 'aop')).toBeHidden();
});

test('the keyboard reads the heatmap cell by cell', async ({ page }) => {
    const card = await open(page);
    const view = plotIn(card, 'aop');
    await expect(titleOf(view)).toHaveText('aop heatmap');
    await expect(view).toHaveAttribute('role', 'figure');
    const live = card.locator('.props-live');

    await view.focus();
    await view.press('ArrowRight');
    await expect(live).toHaveText(`Title (Essential), ${X[0]}: 10% of AOPs (1 of 10), first filled in ${X[0]}.`);
    await view.press('ArrowDown');
    await view.press('ArrowDown');
    await expect(live).toHaveText(`Creator (Essential), ${X[0]}: 0% of AOPs (1 of 10), first filled in ${X[2]}.`);
    await view.press('End');
    await expect(live).toContainText(`Creator (Essential), ${X[X.length - 1]}: 70%`);
    // The hover label follows the keyboard.
    await expect(view.locator('.hoverlayer .hovertext')).toHaveCount(1);
    await view.press('Escape');
    await expect(live).toHaveText('Cell cleared.');

    await view.press('ArrowRight');
    await Promise.all([page.waitForURL(/\/snapshot\?version=/), view.press('Enter')]);
    expect(new URL(page.url()).searchParams.get('version')).toBe(X[0]);
});

test('the keyboard stays inside the /trends range', async ({ page }) => {
    const start = X[2];
    const end = X[4];
    const card = await open(page, `?start=${start}&end=${end}`);
    const view = plotIn(card, 'aop');
    await expect(titleOf(view)).toHaveText('aop heatmap');
    const live = card.locator('.props-live');
    // The range selector clamps the axis once the figure is drawn.
    await expect.poll(() => view.locator('.js-plotly-plot').evaluate((gd) => gd._fullLayout.xaxis.autorange)).toBe(false);

    await view.focus();
    await view.press('ArrowRight');
    await expect(live).toContainText(`Title (Essential), ${start}:`);
    await view.press('End');
    await expect(live).toContainText(`Title (Essential), ${end}:`);
    await view.press('ArrowRight');
    await expect(live).toContainText(`, ${end}:`);
});
