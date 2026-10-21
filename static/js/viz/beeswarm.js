/**
 * AOP completeness beeswarm (#186). Renderer `plotly-beeswarm` for the plot
 * runtime.
 *
 * - Data: /api/v2/data/aop_completeness_beeswarm, one row per AOP per
 *   snapshot with both cohorts' composite scores (beeswarm_rows in
 *   plots/trends_plots.py, from the same scores as the boxplots).
 * - One dot per AOP, for the snapshot at the end of the /trends range (the
 *   version scrubber's right handle, which Play moves forward). Dots carry the
 *   AOP IRI as their Plotly id, so when the snapshot changes each AOP's dot
 *   slides to its new score: individual AOPs can be followed as they mature.
 *   With prefers-reduced-motion the dots jump instead.
 * - Layout: the "dodge" beeswarm, in pixels, the same as swarm_offsets() on
 *   the server. The vertical extent is fixed per cohort across all snapshots
 *   so the axis does not jump while playing.
 * - Cohort (Optional properties / All properties) lives in the page state
 *   (?cohort=all), and so does the picked AOP (?aop=<number>).
 * - Picking an AOP (click a dot, the arrow keys on the focused chart, or its
 *   number in the box) marks its dot in magenta in every snapshot, names it
 *   with its score and rank, links it to its identifiers.org IRI (which
 *   resolves to AOP-Wiki), and lists its scores across the snapshots in a
 *   table. Enter opens the link, Escape clears the pick.
 *
 * The pure helpers are exported for node --test.
 */
(function (root) {
    'use strict';

    const RADIUS = 4.5;          // px, = BEESWARM_RADIUS in plots/trends_plots.py
    const MIN_HEIGHT = 240;
    const MARGIN = { l: 20, r: 20, t: 10, b: 50 };
    const TRANSITION_MS = 600;
    const COHORTS = {
        optional: { column: 'completeness_optional', label: 'Optional properties' },
        all: { column: 'completeness_all', label: 'All properties' },
    };

    // ------------------------------------------------------------ pure parts

    /** Vertical offsets so that dots of `r` px at pixel positions `xs` do not overlap. */
    function swarmOffsets(xs, r) {
        const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b] || a - b);
        const placed = [];
        const out = new Array(xs.length).fill(0);
        const d = 2 * r;
        let start = 0;
        order.forEach((i) => {
            const x = xs[i];
            while (start < placed.length && placed[start][0] < x - d) start++;
            const near = placed.slice(start);
            const candidates = [0];
            near.forEach(([px, py]) => {
                const dy = Math.sqrt(d * d - (x - px) ** 2);
                candidates.push(py + dy, py - dy);
            });
            candidates.sort((a, b) => Math.abs(a) - Math.abs(b) || b - a);
            let y = candidates[0];
            for (const c of candidates) {
                y = c;
                if (near.every(([px, py]) => (x - px) ** 2 + (c - py) ** 2 >= d * d - 1e-6)) break;
            }
            out[i] = y;
            placed.push([x, y]);
        });
        return out;
    }

    /** The snapshots in the data, oldest first. */
    function versionsOf(data) {
        return Array.from(new Set(data.version || [])).sort();
    }

    /** The snapshot to show for the range end `end` (null = the newest). */
    function shownVersion(versions, end) {
        if (!versions.length) return null;
        if (!end) return versions[versions.length - 1];
        const earlier = versions.filter((v) => v <= end);
        return earlier.length ? earlier[earlier.length - 1] : versions[0];
    }

    /** Row indices of `version` that have a `column` score, lowest score first (ties by AOP number). */
    function rowsOf(data, version, column) {
        const out = [];
        for (let i = 0; i < data.version.length; i++) {
            if (data.version[i] === version && data[column][i] !== null && data[column][i] !== undefined) out.push(i);
        }
        return out.sort((a, b) => data[column][a] - data[column][b] || numberOf(data.aop_id[a]) - numberOf(data.aop_id[b]));
    }

    function numberOf(id) {
        const m = String(id || '').match(/(\d+)$/);
        return m ? Number(m[1]) : Infinity;
    }

    /** 1-based rank of row `i` among `rows` (1 = most complete), or null. */
    function rankOf(rows, i) {
        const at = rows.indexOf(i);
        return at < 0 ? null : rows.length - at;
    }

    /**
     * The row a key moves the pick to among `rows` (lowest score first), null
     * to clear it, or undefined for a key that does nothing. Without a pick
     * the arrows start at the median.
     */
    function stepPick(rows, current, key) {
        if (key === 'Escape') return null;
        if (!rows.length) return undefined;
        const at = rows.indexOf(current);
        const mid = Math.floor((rows.length - 1) / 2);
        const to = {
            ArrowRight: at < 0 ? mid : at + 1, ArrowUp: at < 0 ? mid : at + 1,
            ArrowLeft: at < 0 ? mid : at - 1, ArrowDown: at < 0 ? mid : at - 1,
            Home: 0, End: rows.length - 1,
        }[key];
        if (to === undefined) return undefined;
        return rows[Math.max(0, Math.min(rows.length - 1, to))];
    }

    /** Row indices of the AOP with IRI `aop`, oldest snapshot first. */
    function historyOf(data, aop) {
        const out = [];
        for (let i = 0; i < data.aop.length; i++) if (data.aop[i] === aop) out.push(i);
        return out.sort((a, b) => (data.version[a] < data.version[b] ? -1 : 1));
    }

    /**
     * The rows of an AOP's history worth listing: the first, every one whose
     * scores differ from the one before, and `keep` (the snapshot shown).
     */
    function changesOf(data, rows, keep) {
        return rows.filter((r, n) => n === 0 || r === keep
            || data.completeness_optional[r] !== data.completeness_optional[rows[n - 1]]
            || data.completeness_all[r] !== data.completeness_all[rows[n - 1]]
            || data.entity_count[r] !== data.entity_count[rows[n - 1]]);
    }

    /** The IRI of AOP number `n` (a string or number), or null. */
    function aopByNumber(data, n) {
        const want = `AOP ${Number(n)}`;
        const i = data.aop_id.indexOf(want);
        return i < 0 ? null : data.aop[i];
    }

    function median(values) {
        if (!values.length) return null;
        const s = values.slice().sort((a, b) => a - b);
        const m = Math.floor(s.length / 2);
        return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
    }

    const api = { swarmOffsets, versionsOf, shownVersion, rowsOf, rankOf, stepPick, historyOf, changesOf, aopByNumber, median };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
        return;
    }
    root.DashBeeswarm = api;

    // ------------------------------------------------------------- the view

    const views = new WeakMap();
    let tokens = null;

    function getTokens() {
        if (!tokens) {
            tokens = fetch('/static/data/tokens.json').then((r) => r.json())
                .catch(() => ({ color: { primary: '#29235C', magenta: '#E6007E' } }));
        }
        return tokens;
    }

    const reducedMotion = () => !!(root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches);
    const pct = (v) => `${Number(v).toFixed(1)}%`;
    const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

    async function mount(element, payload) {
        destroy(element);
        const Plotly = root.Plotly;
        if (!Plotly) {
            element.innerHTML = '<p class="beeswarm-note">The beeswarm needs Plotly, which did not load. '
                + 'Use "Show Raw Data" below or the CSV download.</p>';
            return;
        }
        const data = payload.data;
        const versions = versionsOf(data);
        const colour = (await getTokens()).color;
        const state = root.DashState;

        element.dataset.a11y = 'self';
        element.setAttribute('role', 'figure');
        element.innerHTML = `
            <div class="beeswarm-controls">
                <div class="segmented-control" role="group" aria-label="Properties counted">
                    ${Object.entries(COHORTS).map(([k, c]) => `<button type="button" class="segmented-button" data-cohort="${k}" aria-pressed="false">${c.label}</button>`).join('')}
                </div>
                <form class="beeswarm-find">
                    <label>Find AOP <input type="number" min="1" step="1" inputmode="numeric" class="beeswarm-find-input" placeholder="number"></label>
                    <button type="submit" class="segmented-button">Show</button>
                </form>
            </div>
            <p class="beeswarm-caption"></p>
            <div class="beeswarm-chart" tabindex="0" role="group" aria-roledescription="beeswarm">
                <div class="beeswarm-plot"></div>
            </div>
            <p class="beeswarm-readout"></p>
            <p class="sr-only beeswarm-live" aria-live="polite"></p>
            <div class="beeswarm-history"></div>`;
        const q = (s) => element.querySelector(s);
        const chart = q('.beeswarm-chart');
        const div = q('.beeswarm-plot');
        const live = q('.beeswarm-live');

        let width = 0;
        let drawnKey = null;   // cohort + width the dots were laid out for
        let layouts = {};      // version -> {rows, ys} for the drawn cohort and width
        let reach = 0;

        const cohort = () => (state && state.get('cohort') === 'all' ? 'all' : 'optional');
        const version = () => shownVersion(versions, state ? state.get('end') : null);
        const picked = () => (state && state.get('aop') ? aopByNumber(data, state.get('aop')) : null);
        const plotWidth = () => Math.max(1, width - MARGIN.l - MARGIN.r);
        const toPx = (v) => (v + 2) / 104 * plotWidth();

        // Lay every snapshot out once per cohort and width, so the y-range is the same for all.
        function layoutAll() {
            const key = `${cohort()}:${width}`;
            if (key === drawnKey) return false;
            const column = COHORTS[cohort()].column;
            layouts = {};
            reach = 0;
            versions.forEach((v) => {
                const rows = rowsOf(data, v, column);
                const ys = swarmOffsets(rows.map((i) => toPx(data[column][i])), RADIUS);
                ys.forEach((y) => { reach = Math.max(reach, Math.abs(y)); });
                layouts[v] = { rows, ys };
            });
            reach += 2 * RADIUS;
            drawnKey = key;
            return true;
        }

        function traces() {
            const column = COHORTS[cohort()].column;
            const { rows, ys } = layouts[version()] || { rows: [], ys: [] };
            const aop = picked();
            const at = rows.findIndex((i) => data.aop[i] === aop);
            const dots = {
                type: 'scatter', mode: 'markers',
                x: rows.map((i) => data[column][i]), y: ys,
                ids: rows.map((i) => data.aop[i]),
                customdata: rows.map((i) => [data.aop[i], data.aop_id[i], data.aop_title[i] || '']),
                marker: { size: 2 * RADIUS, color: colour.primary, opacity: 0.75, line: { width: 0.5, color: 'white' } },
                hovertemplate: '<b>%{customdata[1]}</b> %{customdata[2]}<br>%{x:.1f}%<extra></extra>',
            };
            const mark = {
                type: 'scatter', mode: 'markers',
                x: at < 0 ? [] : [data[column][rows[at]]], y: at < 0 ? [] : [ys[at]],
                ids: at < 0 ? [] : [aop],
                customdata: at < 0 ? [] : [[aop, data.aop_id[rows[at]], data.aop_title[rows[at]] || '']],
                marker: { size: 2 * RADIUS + 6, color: colour.magenta, line: { width: 2, color: 'white' } },
                hovertemplate: '<b>%{customdata[1]}</b> %{customdata[2]}<br>%{x:.1f}%<extra></extra>',
            };
            return [dots, mark];
        }

        function layout() {
            return {
                // One y unit is one pixel, so the dots stay round and apart at any width.
                height: Math.max(MIN_HEIGHT, Math.ceil(2 * reach) + MARGIN.t + MARGIN.b), margin: MARGIN, showlegend: false, hovermode: 'closest',
                xaxis: { title: { text: 'Composite completeness (%)' }, range: [-2, 102], zeroline: false, fixedrange: true },
                yaxis: { visible: false, range: [-reach, reach], fixedrange: true },
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            };
        }

        const config = { displaylogo: false, displayModeBar: false, responsive: false };

        async function draw(animate) {
            const relaid = layoutAll();
            if (!div.classList.contains('js-plotly-plot')) {
                await Plotly.newPlot(div, traces(), layout(), config);
                div.on('plotly_click', (ev) => {
                    const p = ev.points && ev.points[0];
                    if (p && p.customdata) pick(p.customdata[0]);
                });
            } else if (animate && !relaid && !reducedMotion()) {
                await Plotly.animate(div, { data: traces(), traces: [0, 1] }, {
                    transition: { duration: TRANSITION_MS, easing: 'cubic-in-out' },
                    frame: { duration: TRANSITION_MS, redraw: false },
                    mode: 'immediate',
                });
            } else {
                await Plotly.react(div, traces(), layout(), config);
            }
            describe();
        }

        function describe() {
            const v = version();
            const column = COHORTS[cohort()].column;
            const { rows } = layouts[v] || { rows: [] };
            const med = median(rows.map((i) => data[column][i]));
            const label = COHORTS[cohort()].label.toLowerCase();
            q('.beeswarm-caption').textContent = `${rows.length} AOPs in the ${v} snapshot`
                + (med === null ? '.' : `, median ${pct(med)} (${label}).`)
                + (state && state.get('end') ? '' : ' Move the end of the range on the timeline, or press Play, to follow them over time.');
            element.querySelectorAll('[data-cohort]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.cohort === cohort())));
            const name = `Beeswarm of the composite completeness of ${rows.length} AOPs in the ${v} snapshot, ${label}, one dot per AOP`;
            element.setAttribute('aria-label', name);
            chart.setAttribute('aria-label', `${name}. Left and right arrows pick the next less or more complete AOP, Enter opens it, Escape clears the pick.`);
            showPick(rows, v, column);
        }

        function showPick(rows, v, column) {
            const aop = picked();
            const readout = q('.beeswarm-readout');
            const history = q('.beeswarm-history');
            if (!aop) {
                readout.textContent = 'Click a dot, use the arrow keys on the chart, or type an AOP number to follow one AOP across the snapshots.';
                history.innerHTML = '';
                return;
            }
            const hist = historyOf(data, aop);
            const shownRow = hist.find((r) => data.version[r] === v);
            const listed = changesOf(data, hist, shownRow);
            const changes = changesOf(data, hist, -1).length - 1;
            const first = hist[0];
            const id = data.aop_id[first];
            const title = data.aop_title[first];
            const i = rows.find((r) => data.aop[r] === aop);
            const link = `<a href="${esc(aop)}" target="_blank" rel="noopener">${esc(id)}${title ? ` · ${esc(title)}` : ''}</a>`;
            readout.innerHTML = i === undefined
                ? `${link} is not in the ${esc(v)} snapshot.`
                : `${link}: ${pct(data[column][i])} in the ${esc(v)} snapshot, rank ${rankOf(rows, i)} of ${rows.length} (1 = most complete).`;
            history.innerHTML = `
                <table class="data-table beeswarm-rows">
                    <caption>${esc(id)} across the snapshots: in ${hist.length}, with ${changes} change${changes === 1 ? '' : 's'} in its scores (unchanged snapshots left out)</caption>
                    <thead><tr><th scope="col">Snapshot</th><th scope="col">Optional properties</th><th scope="col">All properties</th><th scope="col">Entities averaged</th></tr></thead>
                    <tbody>${listed.map((r) => `<tr${data.version[r] === v ? ' class="beeswarm-current" aria-current="true"' : ''}>
                        <th scope="row">${esc(data.version[r])}</th>
                        <td>${data.completeness_optional[r] === null ? '' : pct(data.completeness_optional[r])}</td>
                        <td>${data.completeness_all[r] === null ? '' : pct(data.completeness_all[r])}</td>
                        <td>${data.entity_count[r]}</td></tr>`).join('')}</tbody>
                </table>
                <p class="beeswarm-history"><a class="history-link" href="/aop?aop=${numberOf(id)}">${esc(id)}'s Key Events in every version</a></p>`;
        }

        function pick(aop) {
            if (!state) return;
            const i = aop ? data.aop.indexOf(aop) : -1;
            state.set({ aop: i < 0 ? null : String(numberOf(data.aop_id[i])) });
        }

        function announce() {
            const aop = picked();
            const v = version();
            const column = COHORTS[cohort()].column;
            const { rows } = layouts[v] || { rows: [] };
            const i = rows.find((r) => data.aop[r] === aop);
            if (!aop) live.textContent = 'Pick cleared.';
            else if (i === undefined) live.textContent = `${data.aop_id[data.aop.indexOf(aop)]} is not in the ${v} snapshot.`;
            else live.textContent = `${data.aop_id[i]}: ${pct(data[column][i])}, rank ${rankOf(rows, i)} of ${rows.length}.`;
        }

        chart.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                const aop = picked();
                if (aop) { event.preventDefault(); root.open(aop, '_blank', 'noopener'); }
                return;
            }
            const { rows } = layouts[version()] || { rows: [] };
            const current = rows.find((r) => data.aop[r] === picked());
            const next = stepPick(rows, current === undefined ? -1 : current, event.key);
            if (next === undefined) return;
            event.preventDefault();
            pick(next === null ? null : data.aop[next]);
        });
        element.querySelectorAll('[data-cohort]').forEach((b) => b.addEventListener('click', () => {
            if (state) state.set({ cohort: b.dataset.cohort === 'optional' ? null : b.dataset.cohort });
        }));
        q('.beeswarm-find').addEventListener('submit', (event) => {
            event.preventDefault();
            const n = q('.beeswarm-find-input').value;
            const aop = n ? aopByNumber(data, n) : null;
            if (aop) pick(aop);
            else live.textContent = n ? `AOP ${n} is in none of the snapshots.` : '';
        });

        const unsubscribe = state ? state.subscribe((changed) => {
            if (changed.includes('cohort')) draw(false);
            else if (changed.includes('end')) draw(true);
            else if (changed.includes('aop')) { draw(true).then(announce); }
        }) : () => {};

        function resize() {
            const w = Math.floor(chart.getBoundingClientRect().width);
            if (!w || w === width) return;
            width = w;
            if (div.classList.contains('js-plotly-plot')) Plotly.relayout(div, { width: w });
            draw(false);
        }
        width = Math.floor(chart.getBoundingClientRect().width) || 800;
        const observer = root.ResizeObserver ? new ResizeObserver(resize) : null;
        if (observer) observer.observe(chart);

        await draw(false);
        views.set(element, {
            destroy() {
                unsubscribe();
                if (observer) observer.disconnect();
                if (div.classList.contains('js-plotly-plot')) Plotly.purge(div);
            },
        });
    }

    function destroy(element) {
        const view = views.get(element);
        if (view) view.destroy();
        views.delete(element);
    }

    const renderer = {
        endpoint(plotName, qs) { return `/api/v2/data/${plotName}${qs}`; },
        mount,
        update: mount,
        destroy,
    };
    if (root.PlotRuntime) root.PlotRuntime.registerRenderer('plotly-beeswarm', renderer);
    root.DashBeeswarm.renderer = renderer;
})(typeof window !== 'undefined' ? window : this);
