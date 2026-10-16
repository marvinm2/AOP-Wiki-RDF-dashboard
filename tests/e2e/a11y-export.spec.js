// Browser tests for client export and the a11y helpers (#181): "as shown"
// downloads named after the state, keyboard reading of a figure, and data
// tables that follow the version.
const { test, expect } = require('@playwright/test');
const { installStubs, pickVersion, VERSIONS } = require('./stubs');

test.beforeEach(async ({ page }) => {
    page.errors = [];
    page.on('pageerror', (e) => page.errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') page.errors.push(m.text()); });
});

async function exportFrom(page, plotName, format) {
    const box = page.locator(`.plot-box:has(.lazy-plot[data-plot-name="${plotName}"])`);
    const item = box.locator(`[data-export-for="${plotName}"][data-export-format="${format}"]`);
    await expect(item).toHaveCount(1);
    await item.evaluate((el) => el.closest('.download-dropdown').querySelector('.dropdown-toggle').click());
    const [download] = await Promise.all([page.waitForEvent('download'), item.click()]);
    return download;
}

test('trend export: the figure as shown, named with the snapshot range', async ({ page }) => {
    await installStubs(page);
    const name = 'aop_entity_counts_absolute';
    await page.goto('/trends');
    await expect(page.locator(`.lazy-plot[data-plot-name="${name}"] .js-plotly-plot`)).toBeVisible();
    expect((await exportFrom(page, name, 'png')).suggestedFilename()).toBe(`${name}.png`);

    await page.goto(`/trends?start=${VERSIONS[5]}&end=${VERSIONS[1]}`);
    await expect(page.locator(`.lazy-plot[data-plot-name="${name}"] .js-plotly-plot`)).toBeVisible();
    const download = await exportFrom(page, name, 'svg');
    expect(download.suggestedFilename()).toBe(`${name}_${VERSIONS[5]}_to_${VERSIONS[1]}.svg`);
    const svg = await (await download.createReadStream()).toArray();
    expect(Buffer.concat(svg).toString()).toContain('<svg');
    expect(page.errors).toEqual([]);
});

test('snapshot export: named with the version and settings', async ({ page }) => {
    await installStubs(page);
    await page.goto(`/snapshot?version=${VERSIONS[2]}&min_jaccard=0.5`);
    const name = 'latest_aop_aop_overlap';
    const plot = page.locator('.overlap-panel--matrix .lazy-plot');  // the matrix tab (#185)
    await plot.scrollIntoViewIfNeeded();
    await expect(plot.locator('.js-plotly-plot')).toBeVisible();
    expect((await exportFrom(page, name, 'png')).suggestedFilename())
        .toBe(`${name}_${VERSIONS[2]}_min_jaccard-0.5.png`);
    expect(page.errors).toEqual([]);
});

test('a drawn figure is labelled and can be read with the keyboard', async ({ page }) => {
    await installStubs(page);
    await page.goto('/trends');
    const name = 'aop_entity_counts_absolute';
    const plot = page.locator(`.lazy-plot[data-plot-name="${name}"]`);
    await expect(plot.locator('.js-plotly-plot')).toBeVisible();
    await expect(plot).toHaveAttribute('role', 'figure');
    await expect(plot).toHaveAccessibleName(/^AOP Entity Counts\. Use the left and right arrow keys/);
    await expect(plot).toHaveAccessibleDescription(/AOPs/);

    const asc = VERSIONS.slice().reverse();
    await plot.focus();
    await page.keyboard.press('ArrowRight');
    await expect(page.locator('#plot-announcer')).toHaveText(`${asc[0]}: A 10, B 5.`);
    await expect(plot.locator('.hoverlayer .hovertext').first()).toBeVisible();
    await page.keyboard.press('End');
    await expect(page.locator('#plot-announcer')).toHaveText(`${asc[asc.length - 1]}: A ${10 + 36}, B ${5 + 12}.`);
    await page.keyboard.press('Escape');
    await expect(plot.locator('.hoverlayer .hovertext')).toHaveCount(0);

    // With a snapshot range set, the keys stay inside it.
    await page.goto(`/trends?start=${VERSIONS[5]}&end=${VERSIONS[1]}`);
    await expect(plot.locator('.js-plotly-plot')).toBeVisible();
    await expect.poll(() => plot.locator('.js-plotly-plot').evaluate((gd) => gd.layout.xaxis.autorange)).toBe(false);
    await plot.focus();
    await page.keyboard.press('Home');
    await expect(page.locator('#plot-announcer')).toHaveText(new RegExp(`^${VERSIONS[5]}:`));
    await page.keyboard.press('End');
    await expect(page.locator('#plot-announcer')).toHaveText(new RegExp(`^${VERSIONS[1]}:`));
    expect(page.errors).toEqual([]);
});

test('the raw-data table follows the version and settings', async ({ page }) => {
    const stats = await installStubs(page);
    await page.goto(`/snapshot?version=${VERSIONS[2]}`);
    await expect(page.locator('#version-scrubber .scrubber-marks circle')).toHaveCount(VERSIONS.length);
    const table = page.locator('[data-table-for="latest_entity_counts"]');
    await table.getByRole('button', { name: 'Show Raw Data' }).click();
    await expect(table.getByRole('button', { name: 'Hide Raw Data' })).toHaveAttribute('aria-expanded', 'true');
    await expect(table.locator('td').first()).toHaveText(`latest_entity_counts|${VERSIONS[2]}|`);
    await expect(table.locator('tbody tr')).toHaveCount(100);
    await expect(table).toContainText('Showing first 100 of 150 rows');
    await expect(table.locator('caption')).toHaveText(/^Data behind: /);

    // An open table reloads when the version changes.
    await pickVersion(page, VERSIONS[3]);
    await expect(table.locator('td').first()).toHaveText(`latest_entity_counts|${VERSIONS[3]}|`);
    expect(stats.requests.filter((u) => u.includes('/api/v2/data/latest_entity_counts'))).toHaveLength(2);
    expect(page.errors).toEqual([]);
});
