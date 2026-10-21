// Browser tests for the network evolution (#189): one layout for every
// snapshot, the scrubber picks the snapshot, new and gone elements marked.
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
const COLUMNS = ['Element', 'ID', 'Label', 'Source', 'Target', 'X', 'Y', 'First Seen', 'Last Seen', 'Presence'];
const ROWS = [
    ['KE', 'KE 1', 'Binding', '', '', 0, 0, VERSIONS[0], VERSIONS[6], 'xxxxxxx'],
    ['KE', 'KE 2', 'Inflammation', '', '', 200, 100, VERSIONS[0], VERSIONS[5], 'xxxxxx-'],
    ['KE', 'KE 3', 'Fibrosis', '', '', -150, 80, VERSIONS[6], VERSIONS[6], '------x'],
    ['KER', 'KER 5', '', 'KE 1', 'KE 2', null, null, VERSIONS[0], VERSIONS[6], 'xxxxxxx'],
    ['KER', 'KER 6', '', 'KE 1', 'KE 3', null, null, VERSIONS[6], VERSIONS[6], '------x'],
];

async function open(page, query) {
    await installStubs(page);
    await page.route('**/api/network/graph', (route) => route.fulfill({ json: { elements: [], version: VERSIONS[6], stats: {} } }));
    await page.route('**/api/network/metrics', (route) => route.fulfill({ json: { metrics: [], version: VERSIONS[6] } }));
    await page.route('**/api/network/communities', (route) => route.fulfill({ json: { communities: [], version: VERSIONS[6] } }));
    await page.route(/\/api\/v2\/data\/network_evolution/, (route) => route.fulfill({
        json: {
            meta: { id: 'network_evolution', columns: COLUMNS.map((name) => ({ name, type: 'string' })), rows: ROWS.length },
            data: Object.fromEntries(COLUMNS.map((c, i) => [c, ROWS.map((r) => r[i])])),
        },
    }));
    await page.goto(`/network${query || ''}`);
    await page.locator('.tab-btn[data-tab="evolution-view"]').click();
    const card = page.locator('#network-evolution');
    await expect(card.locator('.evolution-canvas canvas').first()).toBeAttached();
    return card;
}

const classesOf = (page) => page.evaluate(() => {
    const el = document.querySelector('[data-renderer="cy-evolution"] .evolution-canvas');
    const cy = el && el._cyreg && el._cyreg.cy;
    return Object.fromEntries(cy.elements().map((e) => [e.id(), e.classes().join(' ') || 'shown']));
});

test('the latest snapshot marks what is new and what is gone', async ({ page }) => {
    const card = await open(page);
    await expect(card.locator('.evolution-summary')).toHaveText(
        '2026-10-01: 2 Key Events and 1 relationship. Since 2026-07-01: 1 Key Event and 1 relationship new, '
        + '1 Key Event and 1 relationship gone.');
    expect(await classesOf(page)).toEqual({ 'KE 1': 'shown', 'KE 2': 'gone', 'KE 3': 'new', 'KER 5': 'gone', 'KER 6': 'new' });
});

test('the scrubber picks the snapshot with the keyboard, and positions stay', async ({ page }) => {
    const card = await open(page);
    const position = () => page.evaluate(() => document.querySelector('[data-renderer="cy-evolution"] .evolution-canvas')
        ._cyreg.cy.getElementById('KE 2').position());
    const before = await position();
    const handle = card.locator('.scrubber-handle').first();
    await handle.focus();
    await page.keyboard.press('ArrowLeft');
    await expect(page).toHaveURL(/version=2026-07-01/);
    await expect(card.locator('.evolution-summary')).toHaveText(
        '2026-07-01: 2 Key Events and 1 relationship. Since 2026-04-01: 0 Key Events and 0 relationships new, '
        + '0 Key Events and 0 relationships gone.');
    expect(await classesOf(page)).toEqual({ 'KE 1': 'shown', 'KE 2': 'shown', 'KE 3': 'hidden', 'KER 5': 'shown', 'KER 6': 'hidden' });
    expect(await position()).toEqual(before);
    await page.keyboard.press('Home');
    await expect(card.locator('.evolution-summary')).toHaveText('2024-01-01: 2 Key Events and 1 relationship.');
});

test('?version= opens on that snapshot', async ({ page }) => {
    const card = await open(page, '?version=2026-07-01');
    await expect(card.locator('.evolution-summary')).toContainText('2026-07-01: 2 Key Events and 1 relationship.');
});
