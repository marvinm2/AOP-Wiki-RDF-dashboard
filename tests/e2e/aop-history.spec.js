// Browser tests for the AOP composition history (#190): the picker sets ?aop=,
// the strip marks joins and departures, the keyboard reads the cells, and AOP
// links elsewhere open it.
const { test, expect } = require('@playwright/test');
const { installStubs } = require('./stubs');

test.beforeEach(async ({ page }) => {
    page.errors = [];
    page.on('pageerror', (e) => page.errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') page.errors.push(m.text()); });
});

test.afterEach(async ({ page }) => {
    expect(page.errors).toEqual([]);
});

// The stub versions, oldest first (tests/e2e/stubs.js).
const VERSIONS = ['2024-01-01', '2025-07-01', '2025-10-01', '2026-01-01', '2026-04-01', '2026-07-01', '2026-10-01'];
const COLUMNS = ['Element', 'ID', 'Title', 'Role', 'First Seen', 'Last Seen', 'Roles', 'Stages'];
const ROWS = [
    ['AOP', 'AOP 7', 'Liver fibrosis', 'AOP', VERSIONS[1], VERSIONS[6], '-xxxxxx', '|Under Development|Under Development|Under Review|Under Review|Under Review|Under Review'],
    ['KE', 'KE 10', 'Binding', 'MIE', VERSIONS[1], VERSIONS[6], '-MMMMMM', ''],
    ['KE', 'KE 20', 'Inflammation', 'KE', VERSIONS[1], VERSIONS[6], '-KK-KKK', ''],
    ['KE', 'KE 30', 'Fibrosis', 'AO', VERSIONS[1], VERSIONS[6], '-AAAAAA', ''],
];
const AOPS = [{ number: 3, title: 'Mitochondrial complex I inhibition' }, { number: 7, title: 'Liver fibrosis' }];

async function open(page, query) {
    const stats = await installStubs(page);
    await page.route('**/api/aops', (route) => route.fulfill({ json: { aops: AOPS } }));
    await page.route(/\/api\/v2\/data\/latest_aop_composition/, (route) => {
        const url = new URL(route.request().url());
        stats.requests.push(url.href);
        const rows = url.searchParams.get('aop') === '7' ? ROWS : [];
        route.fulfill({
            json: {
                meta: { id: 'latest_aop_composition', columns: COLUMNS.map((name) => ({ name, type: 'string' })), rows: rows.length },
                data: Object.fromEntries(COLUMNS.map((c, i) => [c, rows.map((r) => r[i])])),
            },
        });
    });
    await page.goto(`/aop${query || ''}`);
    return stats;
}

test('without an AOP the page asks for one', async ({ page }) => {
    await open(page);
    await expect(page.locator('.aop-empty')).toContainText('Pick an AOP');
});

test('the picker sets ?aop= and the strip marks joins and departures', async ({ page }) => {
    const stats = await open(page);
    await page.fill('#aop-input', 'liver fib');
    await page.click('.aop-picker-button');
    await expect(page).toHaveURL(/aop=7/);
    await expect(page.locator('svg.aop-strip')).toBeVisible();
    await expect(page.locator('#aop-input')).toHaveValue('7 · Liver fibrosis');
    expect(stats.requests.some((u) => /aop=7/.test(u))).toBe(true);
    await expect(page.locator('.aop-mark--added')).toHaveCount(1);
    await expect(page.locator('.aop-mark--removed')).toHaveCount(1);
    await expect(page.locator('.aop-cell')).toHaveCount(6 + 5 + 6);
    await expect(page.locator('.aop-summary')).toHaveText(
        'AOP 7: Liver fibrosis. 3 Key Events in 6 versions from 2025-07-01 to 2026-10-01; 3 in 2026-10-01. '
        + 'After 2025-07-01: 1 joined, 1 left.');
    await expect(page.locator('a[href^="/download/latest/aop_composition"]').first()).toHaveAttribute('href', /aop=7/);
});

test('a picker entry that names no single AOP is refused', async ({ page }) => {
    await open(page, '?aop=7');
    await page.fill('#aop-input', 'nothing like it');
    await page.click('.aop-picker-button');
    await expect(page.locator('#aop-picker-help')).toHaveText(/No single AOP matches/);
    await expect(page).toHaveURL(/aop=7/);
});

test('the keyboard reads the cells', async ({ page }) => {
    await open(page, '?aop=7');
    const strip = page.locator('svg.aop-strip');
    await strip.focus();
    const live = page.locator('.aop-live');
    await expect(live).toHaveText('KE 10 Binding, 2026-10-01: molecular initiating event.');
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('ArrowLeft');
    await page.keyboard.press('ArrowLeft');
    await page.keyboard.press('ArrowLeft');
    await expect(live).toHaveText('KE 20 Inflammation, 2026-01-01: not in the AOP, left it in this version.');
    await page.keyboard.press('ArrowRight');
    await expect(live).toHaveText('KE 20 Inflammation, 2026-04-01: key event, joined the AOP in this version.');
    await page.keyboard.press('Home');
    await expect(live).toHaveText('KE 20 Inflammation, 2025-07-01: key event.');
});

test('the Version Diff worklist links AOPs to their history', async ({ page }) => {
    await installStubs(page);
    const COLS = ['Type', 'ID', 'Name', 'Change', 'Fields Changed', 'OECD Stage A', 'OECD Stage B', 'Link'];
    const rows = [['AOPs', '7', 'Liver fibrosis', 'Edited', 'Title', '', '', 'https://identifiers.org/aop/7'],
        ['KEs', '1', 'Steatosis', 'Links only', 'Part of AOP', '', '', 'https://identifiers.org/aop.events/1']];
    await page.route(/\/api\/v2\/data\/latest_version_diff_worklist/, (route) => route.fulfill({
        json: { meta: { columns: COLS.map((name) => ({ name, type: 'string' })), rows: rows.length },
            data: Object.fromEntries(COLS.map((c, i) => [c, rows.map((r) => r[i])])) },
    }));
    await page.goto('/diff');
    await page.locator('[data-plot-name="latest_version_diff_worklist"]').scrollIntoViewIfNeeded();
    const links = page.locator('.diff-worklist-table .history-link');
    await expect(links).toHaveCount(1);
    await expect(links.first()).toHaveAttribute('href', '/aop?aop=7');
});
