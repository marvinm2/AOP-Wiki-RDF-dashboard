// Browser tests for the Version Diff Explorer (#188): the A/B selectors in the
// URL, the worklist's filters and links, and the two-column stage sankey.
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

const COLUMNS = ['Type', 'ID', 'Name', 'Change', 'Fields Changed', 'OECD Stage A', 'OECD Stage B', 'Link'];
const ROWS = [
    ['AOPs', '645', 'Binding of AhR leads to cardiovascular aging', 'Added', '', '', 'No Status', 'https://identifiers.org/aop/645'],
    ['AOPs', '17', 'Liver fibrosis', 'Edited', 'Description; OECD Status', 'No Status', 'Under Development', 'https://identifiers.org/aop/17'],
    ['KEs', '1', 'Steatosis', 'Links only', 'Part of AOP', '', '', 'https://identifiers.org/aop.events/1'],
    ['KERs', '9', '', 'Metadata only', 'Modification Date', '', '', 'https://identifiers.org/aop.relationships/9'],
];

function worklist() {
    return {
        meta: { id: 'latest_version_diff_worklist', columns: COLUMNS.map((name) => ({ name, type: 'string' })), rows: ROWS.length },
        data: Object.fromEntries(COLUMNS.map((c, i) => [c, ROWS.map((r) => r[i])])),
    };
}

function stagesFigure() {
    const nodes = [['2026-07-01', 'No Status', 1], ['2026-07-01', 'Not in 2026-07-01', 1],
        ['2026-10-01', 'Under Development', 1], ['2026-10-01', 'No Status', 1]];
    const links = [[0, 2, '1 AOP moved: No Status (2026-07-01) to Under Development (2026-10-01)'],
        [1, 3, '1 AOP moved: Not in 2026-07-01 (2026-07-01) to No Status (2026-10-01)']];
    return {
        data: [{
            type: 'sankey', arrangement: 'fixed',
            node: { label: nodes.map((n) => n[1]), customdata: nodes.map((n) => [n[0], n[2]]),
                x: [0.001, 0.001, 0.999, 0.999], y: [0.3, 0.7, 0.3, 0.7], pad: 12, thickness: 14 },
            link: { source: links.map((l) => l[0]), target: links.map((l) => l[1]), value: [1, 1],
                color: ['rgba(0,0,0,0.4)', 'rgba(0,0,0,0.4)'], customdata: links.map((l) => l[2]) },
        }],
        layout: { title: { text: 'stages' }, height: 360 },
    };
}

async function open(page, query) {
    const stats = await installStubs(page);
    await page.route(/\/api\/v2\/data\/latest_version_diff_worklist/, (route) => {
        stats.requests.push(route.request().url());
        route.fulfill({ json: worklist() });
    });
    await page.route(/\/api\/v2\/figure\/latest_version_diff_stages/, (route) => {
        stats.requests.push(route.request().url());
        route.fulfill({ json: { meta: { id: 'latest_version_diff_stages' }, figure: stagesFigure() } });
    });
    await page.goto(`/diff${query || ''}`);
    return stats;
}

test('?a= and ?b= become the page state, and A only offers versions older than B', async ({ page }) => {
    const stats = await open(page, '?a=2026-01-01&b=2026-07-01');
    await expect(page).toHaveURL(/\/diff\?version=2026-07-01&against=2026-01-01$/);
    await expect(page.locator('#version-selector')).toHaveValue('2026-07-01');
    await expect(page.locator('#diff-against')).toHaveValue('2026-01-01');
    const options = await page.locator('#diff-against option').evaluateAll((os) => os.map((o) => o.value));
    expect(options).toEqual(['', '2026-04-01', '2026-01-01', '2025-10-01', '2025-07-01', '2024-01-01']);
    await expect.poll(() => stats.requests.some((u) => u.includes('latest_version_diff?')
        && u.includes('version=2026-07-01') && u.includes('against=2026-01-01'))).toBe(true);
    // Downloads hold the same pair.
    await expect(page.locator('a[href^="/download/latest/version_diff?"]').first())
        .toHaveAttribute('href', /version=2026-07-01.*against=2026-01-01|against=2026-01-01.*version=2026-07-01/);
});

test('choosing a B older than A goes back to the version before B', async ({ page }) => {
    await open(page, '?version=2026-07-01&against=2026-01-01');
    await expect(page.locator('#diff-against')).toHaveValue('2026-01-01');
    await page.locator('#version-selector').selectOption('2025-10-01');
    await expect(page).toHaveURL(/\/diff\?version=2025-10-01$/);
    await expect(page.locator('#diff-against')).toHaveValue('');
    await expect(page.locator('#diff-against option').first()).toHaveText('The version before B (2025-07-01)');
});

test('the worklist filters by type, change and text, with a link per entity', async ({ page }) => {
    await open(page);
    const card = page.locator('[data-plot-name="latest_version_diff_worklist"]');
    await card.scrollIntoViewIfNeeded();
    await expect(card.locator('tbody tr')).toHaveCount(4);
    await expect(card.locator('.diff-count')).toHaveText('4 changed entities.');
    await expect(card.locator('tbody a').first()).toHaveAttribute('href', 'https://identifiers.org/aop/645');

    // Keyboard only: the arrow key on the focused type filter picks "AOPs".
    await card.locator('.diff-filters select').first().focus();
    await page.keyboard.press('ArrowDown');
    await expect(card.locator('.diff-filters select').first()).toHaveValue('AOPs');
    await expect(card.locator('tbody tr')).toHaveCount(2);
    await card.locator('.diff-filters select').nth(1).selectOption('Edited');
    await expect(card.locator('.diff-count')).toHaveText('1 of 4 changed entities match.');
    await expect(card.locator('tbody td').nth(5)).toHaveText('No Status → Under Development');

    await card.locator('.diff-filters select').first().selectOption('');
    await card.locator('.diff-filters select').nth(1).selectOption('');
    await card.locator('.diff-filters input').fill('steat');
    await expect(card.locator('tbody tr')).toHaveCount(1);
    await card.locator('.diff-filters input').fill('nothing like this');
    await expect(card.locator('.diff-empty')).toBeVisible();
});

test('the stage sankey has two columns and is walked with the keyboard', async ({ page }) => {
    await open(page);
    const card = page.locator('#diff-stages .plot-box');
    await card.scrollIntoViewIfNeeded();
    await expect(card.locator('.sankey-link')).toHaveCount(2);
    const view = card.locator('.lazy-plot');
    await expect(view).toHaveAttribute('aria-label', /from version A to version B/);
    await view.focus();
    await view.press('ArrowDown');
    await expect(card.locator('.journey-live')).toHaveText(
        '2026-07-01 to 2026-10-01, band 1 of 2: 1 AOP moved: No Status (2026-07-01) to Under Development (2026-10-01).');
    await view.press('ArrowDown');
    await expect(card.locator('.journey-live')).toContainText('band 2 of 2');
});
