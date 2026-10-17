// Browser tests for the AOP completeness beeswarm (#186): dots keyed on the
// AOP IRI that move with the end of the range, reduced motion, picking an AOP
// by click, keyboard and number, and the cohort in the URL.
const { test, expect } = require('@playwright/test');
const { beeswarmData, installStubs, VERSIONS } = require('./stubs');

const NEWEST = VERSIONS[0];
const IRI = (k) => `https://identifiers.org/aop/${k}`;

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
    // Count the animated updates.
    await page.addInitScript(() => {
        window.animations = 0;
        let real;
        Object.defineProperty(window, 'Plotly', {
            configurable: true,
            get() { return real; },
            set(p) { real = p; const animate = p.animate; p.animate = (...a) => { window.animations++; return animate.apply(p, a); }; },
        });
    });
    await page.goto(`/trends${query}`);
    const view = page.locator('.lazy-plot[data-plot-name="aop_completeness_beeswarm"]');
    await view.scrollIntoViewIfNeeded();
    await expect(view.locator('.beeswarm-plot .scatterlayer .point').first()).toBeVisible();
    return view;
}

const dots = (page) => page.$eval('.beeswarm-plot', (gd) => ({ ids: gd.data[0].ids, x: gd.data[0].x, mark: gd.data[1].ids }));
const xOf = async (page, k) => { const d = await dots(page); return d.x[d.ids.indexOf(IRI(k))]; };

test('draws the newest snapshot, one dot per AOP keyed on its IRI', async ({ page }) => {
    const view = await open(page);
    await expect(view).toHaveAttribute('data-rendered-url', '/api/v2/data/aop_completeness_beeswarm');
    const d = await dots(page);
    expect(d.ids).toHaveLength(39);                   // 40 AOPs, AOP 7 left the newest snapshot
    expect(d.ids).not.toContain(IRI(7));
    expect(new Set(d.ids).size).toBe(39);
    await expect(view.locator('.beeswarm-caption')).toHaveText(new RegExp(`^39 AOPs in the ${NEWEST} snapshot, median `));
    await expect(view).toHaveAttribute('aria-label', new RegExp(`39 AOPs in the ${NEWEST} snapshot, optional properties`));
    await expect(view.getByRole('button', { name: 'Optional properties' })).toHaveAttribute('aria-pressed', 'true');
    // A Plotly view, so the menu gets the client "as shown" exports.
    await expect(page.locator('[data-export-for="aop_completeness_beeswarm"]').first()).toBeAttached();
});

test('the end of the range picks the snapshot, and the same dots move', async ({ page }) => {
    await open(page, `?end=${VERSIONS[3]}`);
    const before = await xOf(page, 10);
    expect(await xOf(page, 7)).not.toBeUndefined();   // still there in the older snapshot
    await page.evaluate((v) => window.DashState.set({ end: v }), VERSIONS[2]);
    await expect.poll(() => xOf(page, 10)).toBe(before + 5);
    expect(await page.evaluate(() => window.animations)).toBe(1);
    await page.evaluate(() => window.DashState.set({ end: null }));
    await expect.poll(() => xOf(page, 7)).toBeUndefined();
});

test('with reduced motion the dots jump instead of sliding', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await open(page, `?end=${VERSIONS[3]}`);
    const before = await xOf(page, 10);
    await page.evaluate((v) => window.DashState.set({ end: v }), VERSIONS[2]);
    await expect.poll(() => xOf(page, 10)).toBe(before + 5);
    expect(await page.evaluate(() => window.animations)).toBe(0);
});

test('clicking a dot picks the AOP, marks it and lists it across the snapshots', async ({ page }) => {
    const view = await open(page);
    const point = await page.$eval('.beeswarm-plot', (gd, iri) => {
        const i = gd.data[0].ids.indexOf(iri);
        const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis, box = gd.getBoundingClientRect();
        return { x: box.left + xa._offset + xa.d2p(gd.data[0].x[i]), y: box.top + ya._offset + ya.d2p(gd.data[0].y[i]) };
    }, IRI(10));
    await page.mouse.click(point.x, point.y);
    await expect(page).toHaveURL(/aop=10(&|$)/);
    await expect.poll(async () => (await dots(page)).mark).toEqual([IRI(10)]);
    const link = view.locator('.beeswarm-readout a');
    await expect(link).toHaveAttribute('href', IRI(10));
    await expect(link).toHaveText('AOP 10 · AOP number 10');
    await expect(view.locator('.beeswarm-readout')).toContainText(`in the ${NEWEST} snapshot, rank `);
    const table = view.locator('.beeswarm-rows');
    await expect(table.locator('caption')).toHaveText('AOP 10 across the snapshots: in 6, with 5 changes in its scores (unchanged snapshots left out)');
    await expect(table.locator('tbody tr')).toHaveCount(VERSIONS.length - 1);  // AOP 10 joins at the second
    await expect(table.locator('tbody tr[aria-current="true"] th')).toHaveText(NEWEST);

    // The pick follows the AOP to an older snapshot, and says when it is not in one.
    await page.evaluate((v) => window.DashState.set({ end: v, aop: '1' }), VERSIONS[VERSIONS.length - 1]);
    await expect(view.locator('.beeswarm-readout')).toHaveText(`AOP 1 · AOP number 1 is not in the ${VERSIONS[VERSIONS.length - 1]} snapshot.`);
    await expect.poll(async () => (await dots(page)).mark).toEqual([]);
});

test('keyboard: arrows step by score, Enter opens AOP-Wiki, Escape clears', async ({ page, context }) => {
    const view = await open(page);
    await context.route('https://identifiers.org/**', (route) => route.fulfill({ body: 'stub' }));
    const chart = view.locator('.beeswarm-chart');
    await chart.focus();
    await page.keyboard.press('End');                   // the most complete AOP
    await expect(view.locator('.beeswarm-live')).toHaveText(/^AOP 39: 74\.0%, rank 1 of 39\.$/);
    await page.keyboard.press('ArrowLeft');
    await expect(view.locator('.beeswarm-live')).toHaveText(/, rank 2 of 39\.$/);
    await page.keyboard.press('End');
    const [popup] = await Promise.all([context.waitForEvent('page'), page.keyboard.press('Enter')]);
    expect(popup.url()).toBe(IRI(39));
    await chart.focus();
    await page.keyboard.press('Escape');
    await expect(page).not.toHaveURL(/aop=/);
    await expect(view.locator('.beeswarm-live')).toHaveText('Pick cleared.');
    await expect(view.locator('.beeswarm-rows')).toHaveCount(0);
});

test('an AOP by number, the cohort in the URL, and a shared link', async ({ page }) => {
    const view = await open(page);
    await view.locator('.beeswarm-find-input').fill('12');
    await view.locator('.beeswarm-find-input').press('Enter');
    await expect(page).toHaveURL(/aop=12(&|$)/);
    await view.locator('.beeswarm-find-input').fill('999');
    await view.getByRole('button', { name: 'Show' }).click();
    await expect(view.locator('.beeswarm-live')).toHaveText('AOP 999 is in none of the snapshots.');

    const optional = await xOf(page, 12);
    await view.getByRole('button', { name: 'All properties' }).click();
    await expect(page).toHaveURL(/cohort=all/);
    await expect(view.getByRole('button', { name: 'All properties' })).toHaveAttribute('aria-pressed', 'true');
    await expect.poll(() => xOf(page, 12)).not.toBe(optional);
    const rows = beeswarmData().data;
    const i = rows.aop.findIndex((a, j) => a === IRI(12) && rows.version[j] === NEWEST);
    expect(await xOf(page, 12)).toBe(rows.completeness_all[i]);

    await page.goto('/trends?cohort=all&aop=12');
    const again = page.locator('.lazy-plot[data-plot-name="aop_completeness_beeswarm"]');
    await again.scrollIntoViewIfNeeded();
    await expect(again.locator('.beeswarm-rows caption')).toHaveText(/^AOP 12 across the snapshots: /);
    await expect.poll(async () => (await dots(page)).mark).toEqual([IRI(12)]);
});
