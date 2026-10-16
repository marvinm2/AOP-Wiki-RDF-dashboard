// Clicking a KE in the KE-reuse plot opens its identifiers.org IRI (#240).
const { test, expect } = require('@playwright/test');
const { installStubs } = require('./stubs');

const IRI = 'https://identifiers.org/aop.events/5';

test('clicking a KE reuse bar opens its identifiers.org page', async ({ page, context }) => {
    await installStubs(page);
    // Registered after the stubs, so it wins: one bar carrying the IRI, as the server sends it.
    await page.route((url) => url.pathname === '/api/plot/latest_ke_reuse', (route) => route.fulfill({ json: { success: true, html:
        '<div id="ke" class="plotly-graph-div" style="height:200px;width:100%"></div>'
        + `<script>Plotly.newPlot("ke", [{type:"bar", orientation:"h", x:[12], y:["KE 5"], customdata:[[${JSON.stringify(IRI)}]]}], {height:200}, {displayModeBar:false});</script>` } }));
    await context.route('https://identifiers.org/**', (route) => route.fulfill({ body: 'stub' }));
    await page.goto('/snapshot');
    const view = page.locator('.lazy-plot[data-plot-name="latest_ke_reuse"]');
    await view.scrollIntoViewIfNeeded();
    const bar = view.locator('.bars .point path').first();
    await expect(bar).toBeVisible();
    // Plotly takes clicks on its drag layer above the bars, so click by position.
    const box = await bar.boundingBox();
    const [popup] = await Promise.all([context.waitForEvent('page'),
        page.mouse.click(box.x + box.width / 2, box.y + box.height / 2)]);
    expect(popup.url()).toBe(IRI);
});
