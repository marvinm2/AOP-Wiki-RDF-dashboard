// Browser tests for the Bar | Pie switches on the Snapshot page (#255): each
// card has its own setting, kept in the URL and followed by the plot and its
// downloads.
const { test, expect } = require('@playwright/test');
const { installStubs } = require('./stubs');

test('a card switches to its pie without touching the others', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await installStubs(page);
    await page.goto('/snapshot');
    const depth = page.locator('.plot-box', { has: page.locator('[data-plot-name="latest_ke_annotation_depth"]') });
    await depth.scrollIntoViewIfNeeded();
    const plot = depth.locator('[data-plot-name="latest_ke_annotation_depth"] [data-label]');
    await expect(plot).toHaveAttribute('data-label', 'latest_ke_annotation_depth|latest|');
    await expect(depth.locator('[data-chart="bar"]')).toHaveAttribute('aria-pressed', 'true');

    await depth.locator('[data-chart="pie"]').click();
    await expect(page).toHaveURL(/depth_chart=pie/);
    await expect(plot).toHaveAttribute('data-label', 'latest_ke_annotation_depth|latest|depth_chart=pie');
    await expect(depth.locator('[data-chart="pie"]')).toHaveAttribute('aria-pressed', 'true');
    for (const href of await depth.locator('a[href^="/download/"]').evaluateAll((as) => as.map((a) => a.getAttribute('href')))) {
        expect(href).toMatch(/^\/download\/latest\/ke_annotation_depth\?format=\w+&depth_chart=pie$/);
    }
    // The other switches stay on bars.
    const ontology = page.locator('.chart-switch[data-chart-key="ontology_chart"] [data-chart="bar"]');
    await expect(ontology).toHaveAttribute('aria-pressed', 'true');
    expect(page.url()).not.toMatch(/ontology_chart=/);

    // Back to bars: the default leaves the URL.
    await depth.locator('[data-chart="bar"]').click();
    await expect(page).not.toHaveURL(/depth_chart=/);
    await expect(plot).toHaveAttribute('data-label', 'latest_ke_annotation_depth|latest|');
    expect(errors).toEqual([]);
});

test('a shared link opens with the pie', async ({ page }) => {
    await installStubs(page);
    await page.goto('/snapshot?process_chart=pie');
    const card = page.locator('.plot-box', { has: page.locator('[data-plot-name="latest_process_usage"]') });
    await card.scrollIntoViewIfNeeded();
    await expect(card.locator('[data-chart="pie"]')).toHaveAttribute('aria-pressed', 'true');
    await expect(card.locator('[data-plot-name="latest_process_usage"] [data-label]'))
        .toHaveAttribute('data-label', 'latest_process_usage|latest|process_chart=pie');
});
