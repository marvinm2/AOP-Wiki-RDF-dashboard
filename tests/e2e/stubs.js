// Data stubs for the browser tests (#181): versions, and plot responses that
// draw a tiny real Plotly figure labelled with what was asked for. Responses
// are delayed by a varying amount so out-of-order arrival is the norm.

const VERSIONS = ['2026-10-01', '2026-07-01', '2026-04-01', '2026-01-01', '2025-10-01', '2025-07-01', '2024-01-01'];

let seq = 0;

/** The label a stub plot carries: plot|version|settings (sorted). */
function labelFor(url) {
    const u = new URL(url);
    const plot = u.pathname.split('/').pop();
    const params = [...u.searchParams.entries()].filter(([k]) => k !== 'version').sort();
    return `${plot}|${u.searchParams.get('version') || 'latest'}|${params.map(([k, v]) => `${k}=${v}`).join('&')}`;
}

async function installStubs(page, { minDelay = 20, maxDelay = 400 } = {}) {
    const stats = { requests: [], inflight: 0, maxInflight: 0 };
    await page.route('**/api/versions', (route) => route.fulfill({
        json: { versions: VERSIONS.map((v) => ({ version: v, graph_uri: `http://aopwiki.org/graph/${v}`, date: v })) },
    }));
    await page.route('**/api/plot/**', async (route) => {
        const url = route.request().url();
        stats.requests.push(url);
        stats.inflight++;
        stats.maxInflight = Math.max(stats.maxInflight, stats.inflight);
        const n = seq++;
        // Deterministic but uneven delays (later requests can finish first).
        const delay = minDelay + ((n * 137) % (maxDelay - minDelay + 1));
        await new Promise((r) => setTimeout(r, delay));
        stats.inflight--;
        const label = labelFor(url);
        const id = `stub-${n}`;
        const html = `<div id="${id}" class="plotly-graph-div" data-label="${label}" style="height:120px;width:100%"></div>`
            + `<script>Plotly.newPlot(${JSON.stringify(id)}, [{x:[1,2],y:[1,2],type:"bar"}], {title:{text:${JSON.stringify(label)}},height:120}, {displayModeBar:false});</script>`;
        try {
            await route.fulfill({ json: { success: true, html } });
        } catch (e) { /* the page aborted it */ }
    });
    // Trend figures (/api/v2/figure) and the Plotly setup (/api/v2/plotly).
    await page.route('**/api/v2/plotly', (route) => route.fulfill({
        json: { template: { layout: { font: { family: 'Arial' } } }, config: { displayModeBar: false, responsive: true } },
    }));
    await page.route('**/api/v2/figure/**', async (route) => {
        const url = route.request().url();
        stats.requests.push(url);
        const n = seq++;
        await new Promise((r) => setTimeout(r, minDelay + ((n * 137) % (maxDelay - minDelay + 1))));
        const plot = new URL(url).pathname.split('/').pop();
        const x = VERSIONS.slice().reverse();
        const figure = {
            data: [
                { type: 'scatter', mode: 'lines+markers', name: 'A', x, y: x.map((_, i) => 10 + i * i),
                  hovertemplate: 'version=%{x}<br>Count=%{y}<extra></extra>' },
                { type: 'scatter', mode: 'lines+markers', name: 'B', x, y: x.map((_, i) => 5 + 2 * i),
                  hovertemplate: 'version=%{x}<br>Count=%{y}<extra></extra>' },
            ],
            layout: { title: { text: plot }, height: 160, yaxis: { title: { text: 'Count' } }, meta: plot },
        };
        try {
            await route.fulfill({ json: { meta: { id: plot }, figure } });
        } catch (e) { /* aborted */ }
    });
    // Plot data (/api/v2/data): one row naming what was asked for, plus filler
    // rows so the table has something to truncate.
    await page.route('**/api/v2/data/**', (route) => {
        const url = route.request().url();
        stats.requests.push(url);
        const label = labelFor(url);
        const rows = 150;
        return route.fulfill({ json: {
            meta: { id: label.split('|')[0], rows, columns: [{ name: 'label', type: 'string' }, { name: 'n', type: 'integer' }] },
            data: { label: Array.from({ length: rows }, (_, i) => (i === 0 ? label : `row ${i}`)), n: Array.from({ length: rows }, (_, i) => i) },
        } });
    });
    return stats;
}

/**
 * Pick a version on the snapshot page's scrubber (#182) by clicking its mark,
 * as a pointer user would. Scrolls to the top first: the scrubber is not sticky.
 */
async function pickVersion(page, version) {
    const asc = VERSIONS.slice().reverse();
    // Instant: html has scroll-behavior: smooth, and the mark must be measured after the scroll.
    await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
    const box = await page.locator('#version-scrubber .scrubber-marks circle').nth(asc.indexOf(version)).boundingBox();
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
}

module.exports = { installStubs, labelFor, pickVersion, VERSIONS };
