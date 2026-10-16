// Browser tests for the entity lifeline (#183): canvas drawing within the
// render budget, type/status in the URL, brush and keyboard selection
// driving the linked table.
const { test, expect } = require('@playwright/test');
const { installStubs } = require('./stubs');

test.beforeEach(async ({ page }) => {
    page.errors = [];
    page.on('pageerror', (e) => page.errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') page.errors.push(m.text()); });
});

async function open(page, query = '') {
    await installStubs(page);
    await page.goto(`/trends${query}`);
    const view = page.locator('.lazy-plot[data-plot-name="entity_lifeline"]');
    await view.scrollIntoViewIfNeeded();
    await expect(view.locator('canvas.lifeline-canvas')).toBeVisible();
    await expect(view).toHaveAttribute('data-draw-ms', /\d/);
    return view;
}

test('draws the KEs by default, within the render budget, from /api/v2/data', async ({ page }) => {
    const view = await open(page);
    await expect(view).toHaveAttribute('data-rendered-url', '/api/v2/data/entity_lifeline');
    await expect(view.getByRole('button', { name: /^KEs/ })).toHaveAttribute('aria-pressed', 'true');
    await expect(view.locator('.lifeline-summary')).toHaveText(/^300 KEs ever: /);
    await expect(view).toHaveAttribute('aria-label', 'Entity lifeline of 300 KEs');
    // Something is painted, in the KE colour (brand palette[1], magenta).
    const painted = await view.locator('canvas').evaluate((c) => {
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let magenta = 0;
        for (let i = 0; i < d.length; i += 4) if (d[i] > 200 && d[i + 1] < 40 && d[i + 2] > 100) magenta++;
        return magenta;
    });
    expect(painted).toBeGreaterThan(1000);

    // The largest type, 2,907 KERs, still draws well under 100 ms.
    await view.getByRole('button', { name: /^KERs/ }).click();
    await expect(view.locator('.lifeline-summary')).toHaveText(/^2,907 KERs ever: /);
    expect(Number(await view.getAttribute('data-draw-ms'))).toBeLessThan(100);
    await expect(page).toHaveURL(/entity=KERs/);
    // Not a Plotly figure: no client "as shown" export, and a11y.js leaves its label alone.
    await expect(page.locator('[data-export-for="entity_lifeline"]')).toHaveCount(0);
    expect(page.errors).toEqual([]);
});

test('type and status come from the URL and write back to it', async ({ page }) => {
    const view = await open(page, '?entity=Stressors&status=removed');
    await expect(view.getByRole('button', { name: /^Stressors/ })).toHaveAttribute('aria-pressed', 'true');
    await expect(view.locator('.lifeline-status')).toHaveValue('removed');
    await expect(view.locator('.lifeline-summary')).toHaveText(/Showing \d+ removed\.$/);
    await view.locator('.lifeline-status').selectOption('');
    await expect(page).not.toHaveURL(/status=/);
    await view.getByRole('button', { name: /^KEs/ }).click();
    await expect(page).not.toHaveURL(/entity=/);  // KEs is the default
    expect(page.errors).toEqual([]);
});

test('brushing rows lists them with links to AOP-Wiki; hover names a cell', async ({ page }) => {
    const view = await open(page);
    const box = await view.locator('.lifeline-overlay').boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + 0.5);  // the first row
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height * 0.1, { steps: 4 });
    await page.mouse.up();
    const caption = view.locator('.lifeline-rows caption');
    await expect(caption).toHaveText(/^\d+ KEs selected \(rows 1 to \d+ of 300\)$/);
    const first = view.locator('.lifeline-rows tbody tr').first();
    await expect(first.getByRole('link')).toHaveAttribute('href', /^https:\/\/identifiers\.org\/aop\.events\/\d+$/);

    await page.mouse.move(box.x + box.width * 0.95, box.y + box.height * 0.5);
    await expect(view.locator('.lifeline-tip')).toBeVisible();
    await expect(view.locator('.lifeline-tip')).toHaveText(/^KE \d+ · KEs number \d+ · \d{4}-\d{2}-\d{2}: (present|absent)$/);
    expect(page.errors).toEqual([]);
});

test('the keyboard selects, resizes and clears rows', async ({ page }) => {
    const view = await open(page);
    const chart = view.locator('.lifeline-chart');
    await expect(chart).toHaveAccessibleName(/^Entity lifeline of 300 KEs, one row each/);
    await chart.focus();
    await page.keyboard.press('ArrowDown');
    await expect(view.locator('.lifeline-rows tbody tr')).toHaveCount(20);
    await expect(view.locator('.lifeline-live')).toHaveText('Rows 1 to 20 of 300 KEs selected.');
    await page.keyboard.press('Shift+ArrowDown');
    await expect(view.locator('.lifeline-rows tbody tr')).toHaveCount(21);
    await page.keyboard.press('End');
    await expect(view.locator('.lifeline-rows caption')).toHaveText('21 KEs selected (rows 280 to 300 of 300)');
    // The brush shows the keyboard selection too.
    await expect(view.locator('.lifeline-overlay .selection')).toHaveAttribute('height', /\d/);
    await page.keyboard.press('Escape');
    await expect(view.locator('.lifeline-hint')).toBeVisible();
    expect(page.errors).toEqual([]);
});
