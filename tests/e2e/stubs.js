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
    // The entity lifeline (#183): realistic sizes (2,907 KERs), presence over the stub versions.
    await page.route('**/api/v2/data/entity_lifeline*', (route) => {
        stats.requests.push(route.request().url());
        return route.fulfill({ json: lifelineData() });
    });
    // The organ-system UpSet (#184): ~600 AOPs over the 13 buckets, more than 25 combinations.
    await page.route('**/api/v2/data/latest_organ_upset*', (route) => {
        stats.requests.push(route.request().url());
        return route.fulfill({ json: upsetData(new URL(route.request().url()).searchParams.get('version')) });
    });
    // The AOP-AOP overlap (#185): pairs by cluster; the threshold drops the weakest.
    await page.route('**/api/v2/data/latest_aop_aop_overlap*', (route) => {
        const u = new URL(route.request().url());
        stats.requests.push(u.href);
        return route.fulfill({ json: overlapData(u.searchParams.get('version'), u.searchParams.get('min_jaccard')) });
    });
    // The completeness beeswarm (#186): every AOP in every snapshot it is in.
    await page.route('**/api/v2/data/aop_completeness_beeswarm*', (route) => {
        stats.requests.push(route.request().url());
        return route.fulfill({ json: beeswarmData() });
    });
    return stats;
}

const BEESWARM_AOPS = 40;

/**
 * Beeswarm rows as the server sends them: AOP k joins at snapshot k % 3 and
 * gains 5 points a snapshot; AOP 7 is not in the newest snapshot.
 */
function beeswarmData() {
    const asc = VERSIONS.slice().reverse();
    const cols = { version: [], aop: [], aop_id: [], aop_title: [], completeness_optional: [],
        completeness_all: [], entity_count: [] };
    asc.forEach((v, j) => {
        for (let k = 1; k <= BEESWARM_AOPS; k++) {
            if (j < k % 3 || (k === 7 && j === asc.length - 1)) continue;
            cols.version.push(v);
            cols.aop.push(`https://identifiers.org/aop/${k}`);
            cols.aop_id.push(`AOP ${k}`);
            cols.aop_title.push(`AOP number ${k}`);
            cols.completeness_optional.push(Math.min(100, 5 + k + 5 * (j - k % 3)));
            cols.completeness_all.push(Math.min(100, 40 + k + 3 * (j - k % 3)));
            cols.entity_count.push(1 + k % 5);
        }
    });
    const columns = Object.keys(cols).map((name) => ({ name, type: name.startsWith('completeness') ? 'number' : name === 'entity_count' ? 'integer' : 'string' }));
    return { meta: { id: 'aop_completeness_beeswarm', rows: cols.aop.length, columns }, data: cols };
}

const OVERLAP_SIZES = [8, 6, 4, 3, 2, 2, 2];

/**
 * Overlap rows as the server sends them: clusters of OVERLAP_SIZES AOPs, each
 * a chain plus a link from its first AOP to every third one, in matrix order.
 * The first pair of each cluster is identical. A threshold above 0.34 drops
 * the last cluster; the version shifts the AOP numbers.
 */
function overlapData(version, minJaccard) {
    const sizes = Number(minJaccard || 0.34) > 0.34 ? OVERLAP_SIZES.slice(0, -1) : OVERLAP_SIZES;
    const shift = version ? VERSIONS.indexOf(version) * 100 : 0;
    const cols = { cluster_id: [], cluster_size: [], aop1: [], aop1_id: [], aop1_title: [], n_kes_aop1: [],
        aop2: [], aop2_id: [], aop2_title: [], n_kes_aop2: [], jaccard: [], shared_kes: [],
        identical_ke_set: [], matrix_row: [], matrix_col: [], Version: [] };
    let start = 0;
    sizes.forEach((size, k) => {
        const pairs = [];
        for (let i = 0; i + 1 < size; i++) pairs.push([i, i + 1]);
        for (let i = 3; i < size; i += 3) pairs.push([0, i]);
        pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1]).forEach(([a, b], n) => {
            const pa = start + a, pb = start + b;
            const na = 10 + pa + shift, nb = 10 + pb + shift;
            cols.cluster_id.push(k + 1);
            cols.cluster_size.push(size);
            cols.aop1.push(`https://identifiers.org/aop/${na}`);
            cols.aop1_id.push(`AOP ${na}`);
            cols.aop1_title.push(`AOP number ${na}`);
            cols.n_kes_aop1.push(4 + (pa % 5));
            cols.aop2.push(`https://identifiers.org/aop/${nb}`);
            cols.aop2_id.push(`AOP ${nb}`);
            cols.aop2_title.push(`AOP number ${nb}`);
            cols.n_kes_aop2.push(4 + (pb % 5));
            cols.jaccard.push(n === 0 ? 1 : 0.4 + ((pa + pb) % 5) / 10);
            cols.shared_kes.push(3);
            cols.identical_ke_set.push(n === 0);
            cols.matrix_row.push(pa);
            cols.matrix_col.push(pb);
            cols.Version.push(version || VERSIONS[0]);
        });
        start += size;
    });
    const types = { cluster_id: 'integer', cluster_size: 'integer', n_kes_aop1: 'integer', n_kes_aop2: 'integer',
        jaccard: 'number', shared_kes: 'integer', identical_ke_set: 'boolean', matrix_row: 'integer', matrix_col: 'integer' };
    const columns = Object.keys(cols).map((name) => ({ name, type: types[name] || 'string' }));
    return { meta: { id: 'latest_aop_aop_overlap', rows: cols.aop1.length, columns }, data: cols };
}

/** A Plotly heatmap of overlap rows, as plot_latest_aop_aop_overlap draws it, as /api/plot HTML. */
function overlapMatrixHtml(payload) {
    const d = payload.data;
    const n = Math.max(...d.matrix_col) + 1;
    const labels = Array(n).fill('');
    d.matrix_row.forEach((r, i) => { labels[r] = d.aop1_id[i]; labels[d.matrix_col[i]] = d.aop2_id[i]; });
    const z = Array.from({ length: n }, () => Array(n).fill(null));
    d.matrix_row.forEach((r, i) => { const c = d.matrix_col[i]; z[r][c] = z[c][r] = d.jaccard[i]; });
    const fig = { data: [{ type: 'heatmap', x: labels, y: labels, z, zmin: 0.34, zmax: 1, hoverongaps: false }],
        layout: { height: 420, xaxis: { type: 'category', constrain: 'domain' }, yaxis: { type: 'category', autorange: 'reversed', scaleanchor: 'x', constrain: 'domain' } } };
    return `<div id="overlap-matrix" class="plotly-graph-div" style="height:420px;width:100%"></div>`
        + `<script>Plotly.newPlot("overlap-matrix", ${JSON.stringify(fig.data)}, ${JSON.stringify(fig.layout)}, {displayModeBar:false});</script>`;
}

const BUCKETS = ['Nervous', 'Cardiovascular', 'Respiratory', 'Digestive', 'Hepatobiliary', 'Renal/Urinary',
    'Reproductive', 'Endocrine', 'Immune/Haematopoietic', 'Integumentary', 'Musculoskeletal', 'Sensory'];

/** Deterministic AOP bucket sets; a fifth of the AOPs have no signal. The version shifts the counts. */
function upsetData(version) {
    const n = 600 - (version ? VERSIONS.indexOf(version) * 40 : 0);
    const cols = { aop: [], aop_id: [], title: [], systems_all_kes: [], systems_apical_kes: [],
        systems_adverse_outcome: [], n_systems_all_kes: [], version: [] };
    for (let i = 1; i <= n; i++) {
        let sets = [];
        if (i % 5 !== 0) {
            sets.push(BUCKETS[i % 7]);
            if (i % 3 === 0) sets.push(BUCKETS[7 + (i % 5)]);
            if (i % 17 === 0) sets.push(BUCKETS[(i % 4) + 1]);
            sets = [...new Set(sets)].sort((a, b) => BUCKETS.indexOf(a) - BUCKETS.indexOf(b));
        }
        cols.aop.push(`https://identifiers.org/aop/${i}`);
        cols.aop_id.push(`AOP ${i}`);
        cols.title.push(`AOP number ${i}`);
        cols.systems_all_kes.push(sets.join('; '));
        cols.systems_apical_kes.push(sets.slice(0, 1).join('; '));
        cols.systems_adverse_outcome.push(i % 2 ? sets.slice(0, 1).join('; ') : '');
        cols.n_systems_all_kes.push(sets.length);
        cols.version.push(version || VERSIONS[0]);
    }
    const columns = Object.keys(cols).map((name) => ({ name, type: name === 'n_systems_all_kes' ? 'integer' : 'string' }));
    return { meta: { id: 'latest_organ_upset', rows: n, columns }, data: cols };
}

const LIFELINE_SIZES = { AOPs: 60, KEs: 300, KERs: 2907, Stressors: 80 };
const STATUS_OF = (p) => (p.endsWith('-') ? 'removed' : /x-+x/.test(p) ? 'returned' : p.startsWith('x') ? 'always present' : 'added');

function lifelineData() {
    const n = VERSIONS.length;
    const cols = { entity_type: [], id: [], label: [], first_seen: [], last_seen: [], snapshots_present: [],
        returns: [], status: [], presence: [], uri: [] };
    const asc = VERSIONS.slice().reverse();
    Object.entries(LIFELINE_SIZES).forEach(([type, size]) => {
        for (let i = 0; i < size; i++) {
            const first = i % n;
            let p = '-'.repeat(first) + 'x'.repeat(n - first);
            if (i % 7 === 3) p = p.slice(0, n - 1) + '-';           // removed
            if (i % 11 === 5 && first < n - 3) p = p.slice(0, first + 1) + '-' + p.slice(first + 2);  // returned
            const number = i + 1;
            cols.entity_type.push(type);
            cols.id.push(`${type.replace(/s$/, '')} ${number}`);
            cols.label.push(`${type} number ${number}`);
            cols.first_seen.push(asc[p.indexOf('x')]);
            cols.last_seen.push(asc[p.lastIndexOf('x')]);
            cols.snapshots_present.push(p.split('x').length - 1);
            cols.returns.push((p.slice(p.indexOf('x'), p.lastIndexOf('x') + 1).match(/-+/g) || []).length);
            cols.status.push(STATUS_OF(p));
            cols.presence.push(p);
            cols.uri.push(`https://identifiers.org/aop.events/${number}`);
        }
    });
    const columns = Object.keys(cols).map((name) => ({ name, type: 'string' }));
    return { meta: { id: 'entity_lifeline', rows: cols.id.length, columns }, data: cols };
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

module.exports = { beeswarmData, installStubs, labelFor, overlapData, overlapMatrixHtml, pickVersion, upsetData, VERSIONS };
