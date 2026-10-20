// Browser tests for the KE reuse rank race (#187): the bars follow the end of
// the /trends range, slide (or jump under reduced motion), and the keyboard
// picks a Key Event.
const { test, expect } = require('@playwright/test');
const { installStubs, VERSIONS } = require('./stubs');

const ASC = VERSIONS.slice().reverse();
const KE = (n) => `https://identifiers.org/aop.events/${n}`;

/** Three KEs over the stub snapshots: KE 2 overtakes KE 1 halfway. */
function raceData() {
    const rows = [];
    ASC.forEach((v, i) => {
        const late = i >= ASC.length / 2;
        rows.push([v, 1, 10, late ? 2 : 1], [v, 2, late ? 12 + i : 5, late ? 1 : 2], [v, 3, 3, 3]);
    });
    const col = (k) => rows.map((r) => r[k]);
    return {
        meta: { id: 'ke_reuse_race', rows: rows.length, columns: ['Version', 'KE', 'KE ID', 'Title', 'AOPs', 'Rank'].map((name) => ({ name })) },
        data: {
            Version: col(0), KE: rows.map((r) => KE(r[1])), 'KE ID': rows.map((r) => `KE ${r[1]}`),
            Title: rows.map((r) => `Title ${r[1]}`), AOPs: col(2), Rank: col(3),
        },
    };
}

test.beforeEach(async ({ page }) => {
    page.errors = [];
    page.on('pageerror', (e) => page.errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') page.errors.push(m.text()); });
});

async function open(page) {
    await installStubs(page);
    await page.route(/\/api\/v2\/data\/ke_reuse_race/, (route) => route.fulfill({ json: raceData() }));
    await page.goto('/trends');
    const card = page.locator('#ke-reuse-race');
    await card.scrollIntoViewIfNeeded();
    const gd = card.locator('.race-plot.js-plotly-plot');
    await expect(gd).toBeVisible();
    return { card, gd };
}

const ys = (gd) => gd.evaluate((g) => Array.from(g.data[0].y));

test('the bars follow the range end and slide to their new ranks', async ({ page }) => {
    const { card, gd } = await open(page);
    expect(await ys(gd)).toEqual([2, 1, 3]);   // newest snapshot: KE 2 leads
    await expect(card.locator('.race-caption')).toContainText(ASC[ASC.length - 1]);

    await page.evaluate(() => {
        window.__animated = 0;
        const animate = window.Plotly.animate;
        window.Plotly.animate = (...a) => { window.__animated++; return animate(...a); };
    });
    await page.evaluate((end) => window.DashState.set({ start: null, end }), ASC[1]);
    await expect.poll(() => ys(gd)).toEqual([1, 2, 3]);
    expect(await page.evaluate(() => window.__animated)).toBe(1);
    await expect(card.locator('.race-caption')).toContainText(ASC[1]);
    // The names move with the bars.
    expect(await gd.evaluate((g) => Array.from(g.data[1].text))).toEqual(['KE 1 · Title 1', 'KE 2 · Title 2', 'KE 3 · Title 3']);
    expect(page.errors).toEqual([]);
});

test('keyboard picks a Key Event and follows it through the snapshots', async ({ page }) => {
    const { card, gd } = await open(page);
    const chart = card.locator('.race-chart');
    await chart.focus();
    await chart.press('ArrowDown');
    await expect(card.locator('.race-live')).toHaveText(`KE 2, Title 2: rank 1 of 15 in ${ASC[ASC.length - 1]}, in ${12 + ASC.length - 1} AOPs.`);
    const colours = await gd.evaluate((g) => Array.from(g.data[0].marker.color));
    expect(new Set(colours).size).toBe(2);

    await page.evaluate((end) => window.DashState.set({ end }), ASC[1]);
    await expect(card.locator('.race-live')).toHaveText(`KE 2, Title 2: rank 2 of 15 in ${ASC[1]}, in 5 AOPs.`);
    await expect(card.locator('.race-readout a')).toHaveAttribute('href', KE(2));
    await chart.press('Escape');
    await expect(card.locator('.race-live')).toHaveText('Pick cleared.');
    expect(page.errors).toEqual([]);
});

test('with reduced motion the bars jump without Plotly.animate', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    const { gd } = await open(page);
    await page.evaluate(() => {
        window.__animated = 0;
        const animate = window.Plotly.animate;
        window.Plotly.animate = (...a) => { window.__animated++; return animate(...a); };
    });
    await page.evaluate((end) => window.DashState.set({ end }), ASC[1]);
    await expect.poll(() => ys(gd)).toEqual([1, 2, 3]);
    expect(await page.evaluate(() => window.__animated)).toBe(0);
    expect(page.errors).toEqual([]);
});
