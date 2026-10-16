// Browser tests for the organ-system UpSet (#184): drawn from /api/v2/data,
// follows the scope and the scrubber's version, column/bucket selection by
// pointer and keyboard driving the linked table, "as shown" export, and the
// older panels kept behind a toggle.
const { test, expect } = require('@playwright/test');
const { installStubs, pickVersion, upsetData, VERSIONS } = require('./stubs');
const U = require('../../static/js/viz/upset.js');

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
    await page.goto(`/snapshot${query}`);
    const view = page.locator('.lazy-plot[data-plot-name="latest_organ_upset"]');
    await view.scrollIntoViewIfNeeded();
    await expect(view.locator('svg.upset-svg')).toBeVisible();
    return { view, stats };
}

test('draws the 25 largest combinations from /api/v2/data, and all on request', async ({ page }) => {
    const { view } = await open(page);
    await expect(view).toHaveAttribute('data-rendered-url', '/api/v2/data/latest_organ_upset');
    await expect(view.locator('.upset-summary')).toHaveText(
        '480 of 600 AOPs have an organ-system signal (All KEs), in 55 combinations; 120 have none.');
    await expect(view).toHaveAttribute('aria-label', /: 25 of 55 combinations shown$/);
    // 25 columns (titled "...: n AOPs") and one hit row per bucket ("...: n AOPs in total").
    await expect(view.locator('.upset-hits title').filter({ hasNotText: 'in total' })).toHaveCount(25);
    const buckets = U.setSizes(upsetData(null).data, 'systems_all_kes').length;
    await expect(view.locator('.upset-hits title').filter({ hasText: 'in total' })).toHaveCount(buckets);

    const all = view.getByRole('button', { name: 'Show all 55 combinations' });
    await all.click();
    await expect(view.locator('.upset-hits title').filter({ hasNotText: 'in total' })).toHaveCount(55);
    await expect(view.getByRole('button', { name: 'Show the 25 largest' })).toHaveAttribute('aria-pressed', 'true');
});

test('clicking a column or a bucket lists its AOPs, linked to identifiers.org', async ({ page }) => {
    const { view } = await open(page);
    await expect(view.locator('.upset-hint')).toBeVisible();
    const first = view.locator('.upset-hits rect').first();
    const name = (await first.locator('title').textContent()).replace(/: \d+ AOPs$/, '');
    const size = Number((await first.locator('title').textContent()).match(/(\d+) AOPs$/)[1]);
    await first.click();
    const table = view.locator('.upset-rows');
    await expect(table.locator('caption')).toHaveText(`${size} AOPs: ${name.replace(/ \+ /g, ' + ')} (exactly)`);
    await expect(table.locator('tbody tr')).toHaveCount(size);
    await expect(table.locator('tbody a').first()).toHaveAttribute('href', /^https:\/\/identifiers\.org\/aop\/\d+$/);
    // The picked column turns magenta.
    await expect(view.locator('svg.upset-svg rect[fill="#E6007E"]').first()).toBeAttached();

    const label = /^Nervous: \d+ AOPs in total$/;
    const total = Number((await view.locator('.upset-hits title', { hasText: label }).textContent()).match(/(\d+) AOPs/)[1]);
    await view.locator('.upset-hits rect', { has: page.locator('title', { hasText: label }) }).click();
    await expect(table.locator('caption')).toHaveText(`${total} AOPs: Nervous (alone or with others)`);
});

test('keyboard: arrows step through columns and buckets, Escape clears', async ({ page }) => {
    const { view } = await open(page);
    const chart = view.locator('.upset-chart');
    await chart.focus();
    await page.keyboard.press('ArrowRight');
    await expect(view.locator('.upset-rows caption')).toHaveText(/ \(exactly\)$/);
    await expect(view.locator('.upset-live')).toHaveText(/ \(exactly\): \d+ AOPs\.$/);
    await page.keyboard.press('End');
    const last = await view.locator('.upset-rows caption').textContent();
    await page.keyboard.press('ArrowLeft');
    await expect(view.locator('.upset-rows caption')).not.toHaveText(last);
    await page.keyboard.press('ArrowDown');
    await expect(view.locator('.upset-rows caption')).toHaveText(/ \(alone or with others\)$/);
    await page.keyboard.press('Escape');
    await expect(view.locator('.upset-hint')).toBeVisible();
    await expect(view.locator('.upset-live')).toHaveText('Selection cleared.');
});

test('follows the scope toggle and the scrubber version', async ({ page }) => {
    const { view, stats } = await open(page);
    await page.getByRole('button', { name: 'Adverse Outcome only' }).click();
    await expect(page).toHaveURL(/scope=ao/);
    await expect(view).toHaveAttribute('data-rendered-url', '/api/v2/data/latest_organ_upset?scope=ao');
    await expect(view.locator('.upset-summary')).toHaveText(/\(Adverse Outcome KE only\)/);

    await pickVersion(page, VERSIONS[2]);
    await view.scrollIntoViewIfNeeded();  // plots out of view redraw when they come back
    await expect(view).toHaveAttribute('data-rendered-url', `/api/v2/data/latest_organ_upset?version=${VERSIONS[2]}&scope=ao`);
    await expect(view.locator('.upset-summary')).toHaveText(/ of 520 AOPs /);
    expect(stats.requests.some((u) => u.includes('latest_organ_upset') && u.includes(`version=${VERSIONS[2]}`))).toBe(true);
});

test('"SVG (as shown)" saves the drawn chart', async ({ page }) => {
    const { view } = await open(page, '?scope=apical');
    const box = page.locator('#organ-upset > .plot-header');
    await box.getByRole('button', { name: /Download/ }).click();
    const item = box.getByRole('button', { name: 'SVG (as shown)' });
    const [download] = await Promise.all([page.waitForEvent('download'), item.click()]);
    expect(download.suggestedFilename()).toBe('latest_organ_upset_scope-apical.svg');
    const text = require('fs').readFileSync(await download.path(), 'utf8');
    expect(text).toContain('<svg');
    expect(text).toContain('AOPs per organ system');
    expect(text).not.toContain('upset-hits');
    await expect(view).toBeVisible();
});

test('the older panels wait behind a toggle and load when opened', async ({ page }) => {
    const { stats } = await open(page);
    const details = page.locator('details.legacy-panels');
    await expect(details).not.toHaveAttribute('open', '');
    expect(stats.requests.some((u) => u.includes('latest_organ_coverage_unified'))).toBe(false);
    await details.locator(':scope > summary').click();
    const bar = page.locator('.lazy-plot[data-plot-name="latest_organ_coverage_unified"]');
    await bar.scrollIntoViewIfNeeded();
    await expect(bar.locator('[data-label]')).toHaveAttribute('data-label', /^latest_organ_coverage_unified\|latest\|/);
    // One scope for both: the old bar follows the toggle above the chart.
    await page.getByRole('button', { name: 'Apical KEs only' }).click();
    await bar.scrollIntoViewIfNeeded();
    await expect(bar.locator('[data-label]')).toHaveAttribute('data-label', 'latest_organ_coverage_unified|latest|scope=apical');
    await page.getByRole('button', { name: 'Percentage' }).click();
    await expect(page).toHaveURL(/view=percentage/);
});

test('at phone width the chart scrolls inside its card, not the page', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 800 });
    const { view } = await open(page);
    const scroller = view.locator('.upset-scroll');
    expect(await scroller.evaluate((el) => el.scrollWidth > el.clientWidth)).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});
