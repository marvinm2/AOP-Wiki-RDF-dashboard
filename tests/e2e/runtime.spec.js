// Browser tests for the frontend runtime (#181): smoke, ARIA snapshot, rapid
// version switching, and downloads/settings following the URL-backed state.
const { test, expect } = require('@playwright/test');
const { installStubs, VERSIONS } = require('./stubs');

/** Plot containers on screen that have drawn something. */
async function drawnLabels(page) {
    return page.$$eval('.lazy-plot[data-rendered-url]', (els) => els
        .filter((el) => { const r = el.getBoundingClientRect(); return r.top < innerHeight && r.bottom > 0; })
        .map((el) => ({ name: el.dataset.plotName, label: el.querySelector('[data-label]')?.dataset.label })));
}

function collectErrors(page) {
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    return errors;
}

test('smoke: every page loads and draws its visible plots without script errors', async ({ page }) => {
    const errors = collectErrors(page);
    await installStubs(page);
    // The landing page has no lazy plots; it only has to load cleanly.
    await page.goto('/');
    await expect(page.locator('main, body').first()).toBeVisible();
    for (const path of ['/snapshot', '/trends']) {
        await page.goto(path);
        await expect(page.locator('.lazy-plot[data-rendered-url]').first()).toBeVisible();
        const drawn = await drawnLabels(page);
        expect(drawn.length, path).toBeGreaterThan(0);
        for (const d of drawn) expect(d.label?.split('|')[0], path).toBe(d.name);
    }
    await page.goto('/network');
    await expect(page.locator('h1, h2').first()).toBeVisible();
    // Network data is not stubbed and the test server has no endpoint, so its
    // own fetch errors are expected; script errors are not.
    expect(errors.filter((e) => !/Failed to load resource|network/i.test(e))).toEqual([]);
});

test('ARIA snapshot: the snapshot controls are labelled', async ({ page }) => {
    await installStubs(page);
    await page.goto('/snapshot');
    await expect(page.locator('#version-selector option')).toHaveCount(VERSIONS.length);
    await expect(page.locator('.snapshot-topbar-version')).toMatchAriaSnapshot(`
      - text: "View version:"
      - combobox "View version:":
        - option "Current (${VERSIONS[0]})" [selected]
    `);
    await expect(page.locator('.coverage-toggle')).toMatchAriaSnapshot(`
      - group "Coverage scope":
        - button "All KEs" [pressed]
        - button "Apical KEs only"
        - button "Adverse Outcome only"
      - group "Coverage view":
        - button "Absolute" [pressed]
        - button "Percentage"
    `);
});

test('five version switches in two seconds leave no stale plot', async ({ page }) => {
    const stats = await installStubs(page, { minDelay: 50, maxDelay: 900 });
    await page.goto('/snapshot');
    await expect(page.locator('#version-selector option')).toHaveCount(VERSIONS.length);
    await expect(page.locator('.lazy-plot[data-rendered-url]').first()).toBeVisible();

    // Record every draw, so a stale response drawn on the way (even if a later
    // request then corrects it) fails the test.
    await page.evaluate(() => {
        window.__draws = [];
        document.addEventListener('plot:rendered', (e) => window.__draws.push({
            name: e.detail.plotName,
            label: e.target.querySelector('[data-label]')?.dataset.label,
            after: !!window.__lastSwitch,
        }));
    });
    const picks = [VERSIONS[3], VERSIONS[1], VERSIONS[6], VERSIONS[2], VERSIONS[5]];
    const started = Date.now();
    for (const [i, v] of picks.entries()) {
        if (i === picks.length - 1) await page.evaluate(() => { window.__lastSwitch = true; });
        await page.selectOption('#version-selector', v);
        await page.waitForTimeout(250);
    }
    expect(Date.now() - started).toBeLessThan(2000);
    const final = picks[picks.length - 1];

    await expect(page).toHaveURL(new RegExp(`version=${final}`));
    await expect.poll(async () => {
        const drawn = await drawnLabels(page);
        return drawn.length > 0 && drawn.every((d) => d.label === `${d.name}|${final}|`);
    }, { timeout: 15000 }).toBe(true);
    // Settle, then check nothing older was drawn over the final selection.
    await page.waitForTimeout(1200);
    for (const d of await drawnLabels(page)) expect(d.label).toBe(`${d.name}|${final}|`);
    const late = await page.evaluate(() => window.__draws.filter((d) => d.after));
    expect(late.length).toBeGreaterThan(0);
    const versioned = await page.evaluate(() => window.VERSIONED_PLOTS);
    for (const d of late.filter((d) => versioned.includes(d.name))) {
        expect(d.label, 'a stale plot was drawn after the last switch').toBe(`${d.name}|${final}|`);
    }
    expect(stats.maxInflight).toBeLessThanOrEqual(4);
});

test('downloads and settings follow the URL-backed state', async ({ page }) => {
    await installStubs(page);
    await page.goto('/snapshot');
    await expect(page.locator('#version-selector option')).toHaveCount(VERSIONS.length);

    // Version: snapshot download links carry it; trend ones do not exist here.
    await page.selectOption('#version-selector', VERSIONS[2]);
    const hrefs = await page.$$eval('a[href^="/download/latest"]', (as) => as.map((a) => a.getAttribute('href')));
    expect(hrefs.length).toBeGreaterThan(10);
    for (const h of hrefs) expect(h).toContain(`version=${VERSIONS[2]}`);

    // A setting goes into the URL and survives a reload.
    const overlap = page.locator('.lazy-plot[data-plot-name="latest_aop_aop_overlap"]');
    await overlap.scrollIntoViewIfNeeded();
    await page.locator('#aop-aop-threshold-slider').fill('0.5');
    await page.locator('#aop-aop-threshold-slider').dispatchEvent('change');
    await expect(page).toHaveURL(/min_jaccard=0\.5/);
    await expect(overlap.locator('[data-label]')).toHaveAttribute('data-label',
        `latest_aop_aop_overlap|${VERSIONS[2]}|min_jaccard=0.5`);

    await page.reload();
    await expect(page.locator('#version-selector')).toHaveValue(VERSIONS[2]);
    await expect(page.locator('#aop-aop-threshold-value')).toHaveText('0.5');
    await overlap.scrollIntoViewIfNeeded();
    await expect(overlap.locator('[data-label]')).toHaveAttribute('data-label',
        `latest_aop_aop_overlap|${VERSIONS[2]}|min_jaccard=0.5`);

    // Coverage scope: pressed state and request both follow.
    const bar = page.locator('.lazy-plot[data-plot-name="latest_organ_coverage_unified"]');
    await bar.scrollIntoViewIfNeeded();
    await page.getByRole('button', { name: 'Adverse Outcome only' }).click();
    await expect(page).toHaveURL(/scope=ao/);
    await expect(page.getByRole('button', { name: 'Adverse Outcome only' })).toHaveAttribute('aria-pressed', 'true');
    await expect(bar.locator('[data-label]')).toHaveAttribute('data-label',
        `latest_organ_coverage_unified|${VERSIONS[2]}|scope=ao`);

    // Back to the latest: the version leaves the URL and the links.
    await page.selectOption('#version-selector', VERSIONS[0]);
    await expect(page).not.toHaveURL(/version=/);
    const after = await page.$$eval('a[href^="/download/latest"]', (as) => as.map((a) => a.getAttribute('href')));
    for (const h of after) expect(h).not.toContain('version=');

    // Trends: the range goes into the URL and the download links.
    await page.goto(`/trends?start=${VERSIONS[5]}&end=${VERSIONS[1]}`);
    await expect.poll(() => page.$$eval('a[href^="/download/trend/"]', (as) => as.map((a) => a.getAttribute('href'))))
        .toEqual(expect.arrayContaining([expect.stringContaining(`start=${VERSIONS[5]}`)]));
    const trendLinks = await page.$$eval('a[href^="/download/trend/"]', (as) => as.map((a) => a.getAttribute('href')));
    for (const h of trendLinks) expect(h).toContain(`end=${VERSIONS[1]}`);
});
