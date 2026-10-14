// Browser tests for the version scrubber (#182): one control over the
// snapshots that writes the page state on /snapshot and /trends.
const { test, expect } = require('@playwright/test');
const { installStubs, pickVersion, VERSIONS } = require('./stubs');

const ASC = VERSIONS.slice().reverse();

test.beforeEach(async ({ page }) => {
    page.errors = [];
    page.on('pageerror', (e) => page.errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') page.errors.push(m.text()); });
});

async function markCenter(page, version) {
    const box = await page.locator('#version-scrubber .scrubber-marks circle').nth(ASC.indexOf(version)).boundingBox();
    return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

test('snapshot: click and keyboard pick the version; plots follow', async ({ page }) => {
    await installStubs(page);
    await page.goto('/snapshot');
    const slider = page.getByRole('slider', { name: 'Snapshot version' });
    await expect(slider).toHaveAttribute('aria-valuetext', `${VERSIONS[0]} (current)`);

    await pickVersion(page, VERSIONS[3]);
    await expect(page).toHaveURL(new RegExp(`version=${VERSIONS[3]}`));
    await expect(slider).toHaveAttribute('aria-valuetext', VERSIONS[3]);
    await expect(page.locator('#version-banner')).toBeVisible();
    await expect(page.locator('.lazy-plot[data-plot-name="latest_entity_counts"] [data-label]'))
        .toHaveAttribute('data-label', `latest_entity_counts|${VERSIONS[3]}|`);

    // Keyboard: one step newer, then to the latest (the default: no ?version=).
    await slider.focus();
    await page.keyboard.press('ArrowRight');
    await expect(page).toHaveURL(new RegExp(`version=${VERSIONS[2]}`));
    await page.keyboard.press('End');
    await expect(page).not.toHaveURL(/version=/);
    await expect(slider).toHaveAttribute('aria-valuenow', String(ASC.length - 1));
    expect(page.errors).toEqual([]);
});

test('snapshot: a ?version= link and the banner reset both move the handle', async ({ page }) => {
    await installStubs(page);
    await page.goto(`/snapshot?version=${VERSIONS[4]}`);
    const slider = page.getByRole('slider', { name: 'Snapshot version' });
    await expect(slider).toHaveAttribute('aria-valuetext', VERSIONS[4]);
    await page.evaluate(() => window.resetToCurrentVersion());
    await expect(slider).toHaveAttribute('aria-valuetext', `${VERSIONS[0]} (current)`);
    expect(page.errors).toEqual([]);
});

test('snapshot: Play steps through the versions and Pause stops it', async ({ page }) => {
    await installStubs(page);
    await page.goto(`/snapshot?version=${ASC[1]}`);
    const play = page.getByRole('button', { name: 'Play through the snapshots' });
    await play.click();
    await expect(play).toHaveAttribute('aria-pressed', 'true');
    await expect(page).toHaveURL(new RegExp(`version=${ASC[2]}`));
    await expect(page).toHaveURL(new RegExp(`version=${ASC[3]}`), { timeout: 4000 });
    await play.click();
    await expect(play).toHaveAttribute('aria-pressed', 'false');
    const url = page.url();
    await page.waitForTimeout(2500);
    expect(page.url()).toBe(url);
    expect(page.errors).toEqual([]);
});

test('trends: brushing picks a range that clamps the plots and the downloads', async ({ page }) => {
    await installStubs(page);
    await page.goto('/trends');
    await expect(page.locator('#trends-range-start')).toBeHidden();
    await expect(page.locator('.lazy-plot[data-plot-name="aop_entity_counts_absolute"] .js-plotly-plot')).toBeVisible();

    const a = await markCenter(page, ASC[1]);
    const b = await markCenter(page, ASC[4]);
    await page.mouse.move(a.x - 2, a.y);
    await page.mouse.down();
    await page.mouse.move(b.x + 2, b.y, { steps: 5 });
    await page.mouse.up();
    await expect(page).toHaveURL(new RegExp(`start=${ASC[1]}&end=${ASC[4]}`));
    await expect(page.locator('#trends-range-status')).toHaveText(`Constrained to ${ASC[1]} → ${ASC[4]}.`);
    await expect(page.locator('.scrubber-readout')).toHaveText(`${ASC[1]} → ${ASC[4]}`);
    const hrefs = await page.$$eval('a[href^="/download/trend/"]', (as) => as.map((x) => x.getAttribute('href')));
    for (const h of hrefs) expect(h).toContain(`start=${ASC[1]}`);
    await expect.poll(() => page.locator('.lazy-plot[data-plot-name="aop_entity_counts_absolute"] .js-plotly-plot')
        .evaluate((gd) => gd.layout.xaxis.autorange)).toBe(false);

    // Reset (outside the scrubber) clears the range, and the scrubber follows.
    await page.click('#trends-range-reset');
    await expect(page).not.toHaveURL(/start=/);
    await expect(page.locator('.scrubber-readout')).toHaveText('All snapshots');
    expect(page.errors).toEqual([]);
});

test('trends: the range handles work from the keyboard and never cross', async ({ page }) => {
    await installStubs(page);
    await page.goto(`/trends?start=${ASC[2]}&end=${ASC[4]}`);
    const start = page.getByRole('slider', { name: 'Range start' });
    const end = page.getByRole('slider', { name: 'Range end' });
    await expect(start).toHaveAttribute('aria-valuetext', ASC[2]);
    await expect(end).toHaveAttribute('aria-valuetext', ASC[4]);

    await start.focus();
    await page.keyboard.press('ArrowLeft');
    await expect(page).toHaveURL(new RegExp(`start=${ASC[1]}&end=${ASC[4]}`));
    await end.focus();
    await page.keyboard.press('Home');  // past the start: both meet there
    await expect(page).toHaveURL(new RegExp(`start=${ASC[0]}&end=${ASC[0]}`));
    await page.keyboard.press('End');
    // Start at the first, end at the last: all snapshots, the default.
    await expect(page).not.toHaveURL(/start=|end=/);
    expect(page.errors).toEqual([]);
});

test('trends: Play grows the range one snapshot at a time', async ({ page }) => {
    await installStubs(page);
    await page.goto('/trends');
    const play = page.getByRole('button', { name: 'Play the range forward' });
    await play.click();
    await expect(page).toHaveURL(new RegExp(`start=${ASC[0]}&end=${ASC[0]}`));
    await expect(page).toHaveURL(new RegExp(`start=${ASC[0]}&end=${ASC[1]}`), { timeout: 4000 });
    await page.getByRole('slider', { name: 'Range end' }).focus();
    await page.keyboard.press('ArrowRight');  // any other input pauses
    await expect(play).toHaveAttribute('aria-pressed', 'false');
    expect(page.errors).toEqual([]);
});

test('without D3 the version <select> stays and still works', async ({ page }) => {
    await installStubs(page);
    await page.route('**/vendor/d3@*/**', (route) => route.abort());
    await page.goto('/snapshot');
    await expect(page.locator('#version-selector option')).toHaveCount(VERSIONS.length);
    await expect(page.locator('#version-selector')).toBeVisible();
    await expect(page.locator('#version-scrubber .scrubber-handle')).toHaveCount(0);
    await page.selectOption('#version-selector', VERSIONS[2]);
    await expect(page).toHaveURL(new RegExp(`version=${VERSIONS[2]}`));
});
