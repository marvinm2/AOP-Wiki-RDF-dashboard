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
    return stats;
}

module.exports = { installStubs, labelFor, VERSIONS };
