/**
 * Organ-system UpSet chart (#184): how the 13 organ-system buckets of the
 * AOPs overlap. Renderer `d3-upset` for the plot runtime.
 *
 * - Data: /api/v2/data/latest_organ_upset, one row per AOP in the snapshot
 *   with its buckets under each scope ("; "-separated). See
 *   plot_latest_organ_upset in plots/latest_plots.py, which orders
 *   intersections and buckets the same way for its PNG/SVG.
 * - The scope (all KEs / apical KEs / adverse outcome) and the version come
 *   from the page state, like the other coverage plots; the runtime refetches
 *   when either changes.
 * - Columns are the exclusive intersections (each AOP counts once, in the
 *   combination that is exactly its bucket set), largest first. Rows are the
 *   buckets, largest first, with their total AOPs on the left. The 25 largest
 *   columns are shown; a button shows all of them.
 * - Clicking a column, or a bucket, lists its AOPs in a table below, linked to
 *   their identifiers.org IRIs. On the focused chart, Left/Right step through
 *   the columns, Up/Down through the buckets, Home/End jump to the first and
 *   last column, Escape clears.
 * - Colours: brand primary (bars, dots), blue (bucket totals), magenta (the
 *   selection), from static/data/tokens.json. No transitions. Colours are SVG
 *   attributes, so "SVG/PNG (as shown)" (viz/export.js) saves what is drawn.
 *
 * The pure helpers are exported for node --test.
 */
(function (root) {
    'use strict';

    const SCOPES = {
        all: { column: 'systems_all_kes', label: 'All KEs' },
        apical: { column: 'systems_apical_kes', label: 'Apical KEs' },
        ao: { column: 'systems_adverse_outcome', label: 'Adverse Outcome KE only' },
    };
    const SEPARATOR = '; ';
    const TOP = 25;          // columns shown before "Show all"
    const TABLE_MAX = 200;   // rows the linked table lists

    // ------------------------------------------------------------ pure parts

    function scopeOf(value) {
        return Object.prototype.hasOwnProperty.call(SCOPES, value) ? value : 'all';
    }

    function setsOf(value) {
        return value ? String(value).split(SEPARATOR) : [];
    }

    /**
     * Exclusive intersections of `column`: [{key: [buckets], members: [row
     * indices]}], largest first; ties go to fewer buckets, then by name.
     */
    function intersections(data, column) {
        const byKey = new Map();
        (data[column] || []).forEach((value, i) => {
            if (!value) return;
            if (!byKey.has(value)) byKey.set(value, { key: setsOf(value), members: [] });
            byKey.get(value).members.push(i);
        });
        return Array.from(byKey.values()).sort((a, b) => (b.members.length - a.members.length)
            || (a.key.length - b.key.length)
            || (a.key.join(SEPARATOR) < b.key.join(SEPARATOR) ? -1 : 1));
    }

    /** [{name, members}] of every bucket used in `column`, largest first, then by name. */
    function setSizes(data, column) {
        const byName = new Map();
        (data[column] || []).forEach((value, i) => {
            setsOf(value).forEach((name) => {
                if (!byName.has(name)) byName.set(name, { name, members: [] });
                byName.get(name).members.push(i);
            });
        });
        return Array.from(byName.values()).sort((a, b) => (b.members.length - a.members.length)
            || (a.name < b.name ? -1 : 1));
    }

    /**
     * The selection a key leaves: {kind: 'combo'|'set', i}, null (cleared),
     * or undefined for a key that does nothing here.
     */
    function step(sel, key, nCombos, nSets) {
        if (key === 'Escape') return null;
        const clamp = (i, n) => Math.max(0, Math.min(n - 1, i));
        const combo = sel && sel.kind === 'combo' ? sel.i : null;
        const set = sel && sel.kind === 'set' ? sel.i : null;
        switch (key) {
            case 'ArrowRight':
                return nCombos ? { kind: 'combo', i: combo === null ? 0 : clamp(combo + 1, nCombos) } : undefined;
            case 'ArrowLeft':
                return nCombos ? { kind: 'combo', i: combo === null ? nCombos - 1 : clamp(combo - 1, nCombos) } : undefined;
            case 'Home':
                return nCombos ? { kind: 'combo', i: 0 } : undefined;
            case 'End':
                return nCombos ? { kind: 'combo', i: nCombos - 1 } : undefined;
            case 'ArrowDown':
                return nSets ? { kind: 'set', i: set === null ? 0 : clamp(set + 1, nSets) } : undefined;
            case 'ArrowUp':
                return nSets ? { kind: 'set', i: set === null ? nSets - 1 : clamp(set - 1, nSets) } : undefined;
            default:
                return undefined;
        }
    }

    /** "432 of 599 AOPs have an organ-system signal (All KEs), in 79 combinations; 167 have none." */
    function summary(data, scope) {
        const column = SCOPES[scopeOf(scope)].column;
        const values = data[column] || [];
        const classified = values.filter(Boolean).length;
        const fmt = (n) => n.toLocaleString('en-US');
        const n = intersections(data, column).length;
        return `${fmt(classified)} of ${fmt(values.length)} AOPs have an organ-system signal `
            + `(${SCOPES[scopeOf(scope)].label}), in ${fmt(n)} combination${n === 1 ? '' : 's'}; `
            + `${fmt(values.length - classified)} have none.`;
    }

    /** Only identifiers.org IRIs become links. */
    function linkable(uri) {
        return typeof uri === 'string' && uri.startsWith('https://identifiers.org/');
    }

    const api = { SCOPES, TOP, scopeOf, intersections, setSizes, step, summary, linkable };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
        return;
    }
    root.DashUpset = api;

    // ------------------------------------------------------------- the view

    const BAR_H = 200;       // intersection bars
    const ROW_H = 22;        // one bucket row of the matrix
    const LABEL_W = 170;     // bucket names
    const TOTAL_W = 90;      // bucket-total bars
    const MIN_COL = 16;      // narrowest column; wider charts scroll sideways
    const GAP = 8;

    let tokens = null;
    function getTokens() {
        if (!tokens) {
            tokens = fetch('/static/data/tokens.json').then((r) => r.json())
                .catch(() => ({ color: { primary: '#29235C', magenta: '#E6007E', blue: '#307BBF' } }));
        }
        return tokens;
    }

    function escapeHtml(text) {
        return String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    const views = new WeakMap();   // container -> {destroy}
    const prefs = new WeakMap();   // container -> {showAll}, kept across versions and scopes

    async function mount(element, payload) {
        destroy(element);
        const d3 = root.d3;
        if (!d3) {
            element.innerHTML = '<p class="upset-note">The overlap chart needs JavaScript charting that did not load. '
                + 'Use "Show Raw Data" below or the CSV download.</p>';
            return;
        }
        const data = payload.data;
        const colour = (await getTokens()).color;
        const C = { bar: colour.primary, total: colour.blue, pick: colour.magenta, off: '#E6E6E6', text: colour.primary };
        const state = root.DashState;
        const scope = scopeOf(state && state.get('scope'));
        const column = SCOPES[scope].column;
        const combos = intersections(data, column);
        const sets = setSizes(data, column);
        const pref = prefs.get(element) || { showAll: false };
        prefs.set(element, pref);

        element.dataset.a11y = 'self';
        element.setAttribute('role', 'figure');
        element.innerHTML = `
            <div class="upset-controls">
                <p class="upset-summary"></p>
                <button type="button" class="segmented-button upset-show-all" aria-pressed="false" hidden></button>
            </div>
            <div class="upset-scroll">
                <div class="upset-chart" tabindex="0" role="group" aria-roledescription="UpSet chart">
                    <svg class="upset-svg" data-export="svg" aria-hidden="true"></svg>
                </div>
            </div>
            <p class="sr-only upset-live" aria-live="polite"></p>
            <div class="upset-table"></div>`;

        const q = (s) => element.querySelector(s);
        const chart = q('.upset-chart');
        const scroller = q('.upset-scroll');
        const live = q('.upset-live');
        const table = q('.upset-table');
        const toggle = q('.upset-show-all');
        const svg = d3.select(q('.upset-svg'));
        let selection = null;
        let width = 0;

        const shown = () => (pref.showAll ? combos : combos.slice(0, TOP));
        const comboName = (c) => c.key.join(' + ');

        function picked() {
            if (!selection) return null;
            if (selection.kind === 'combo') {
                const c = shown()[selection.i];
                return c && { title: `${comboName(c)} (exactly)`, members: c.members };
            }
            const s = sets[selection.i];
            return s && { title: `${s.name} (alone or with others)`, members: s.members };
        }

        function draw() {
            const cols = shown();
            const left = LABEL_W + TOTAL_W + GAP;
            const colW = Math.max(MIN_COL, Math.floor((width - left) / Math.max(1, cols.length)));
            const w = left + colW * cols.length;
            const h = BAR_H + GAP + ROW_H * sets.length + 24;
            const font = getComputedStyle(element).fontFamily || 'sans-serif';
            svg.attr('width', w).attr('height', h).attr('viewBox', `0 0 ${w} ${h}`)
                .attr('font-family', font).attr('font-size', 12).attr('fill', C.text);
            svg.selectAll('*').remove();

            const yBar = d3.scaleLinear().domain([0, d3.max(cols, (c) => c.members.length) || 1]).range([BAR_H, 16]);
            const xTot = d3.scaleLinear().domain([0, d3.max(sets, (s) => s.members.length) || 1]).range([0, TOTAL_W - 28]);
            const rowY = (i) => BAR_H + GAP + i * ROW_H + ROW_H / 2;
            const colX = (j) => left + j * colW + colW / 2;
            const setIndex = new Map(sets.map((s, i) => [s.name, i]));
            const selSet = selection && selection.kind === 'set' ? sets[selection.i].name : null;
            const isPicked = (c, j) => (selection && selection.kind === 'combo' && selection.i === j)
                || (selSet !== null && c.key.includes(selSet));

            // Zebra rows, so a dot can be followed to its bucket.
            const rows = svg.append('g');
            sets.forEach((s, i) => {
                if (i % 2 === 0) {
                    rows.append('rect').attr('x', 0).attr('y', rowY(i) - ROW_H / 2).attr('width', w)
                        .attr('height', ROW_H).attr('fill', '#F7F7F9');
                }
            });

            // Intersection bars with their counts.
            const bars = svg.append('g');
            cols.forEach((c, j) => {
                const n = c.members.length;
                bars.append('rect').attr('x', left + j * colW + 2).attr('width', Math.max(1, colW - 4))
                    .attr('y', yBar(n)).attr('height', BAR_H - yBar(n))
                    .attr('fill', isPicked(c, j) ? C.pick : C.bar);
                if (colW >= 14) {
                    bars.append('text').attr('x', colX(j)).attr('y', yBar(n) - 4).attr('text-anchor', 'middle')
                        .attr('font-size', colW >= 22 ? 11 : 9).text(n);
                }
            });
            // Axis label in the empty corner left of the bars, clear of their counts.
            svg.append('text').attr('x', left - GAP).attr('y', BAR_H - 4).attr('text-anchor', 'end')
                .attr('font-size', 11).text('AOPs with exactly this combination');

            // Matrix: light dots everywhere, members dark (magenta when picked) and joined.
            const matrix = svg.append('g');
            cols.forEach((c, j) => {
                const colour = isPicked(c, j) ? C.pick : C.bar;
                const ys = c.key.map((b) => rowY(setIndex.get(b))).sort((a, b) => a - b);
                sets.forEach((s, i) => {
                    matrix.append('circle').attr('cx', colX(j)).attr('cy', rowY(i)).attr('r', 4.5).attr('fill', C.off);
                });
                if (ys.length > 1) {
                    matrix.append('line').attr('x1', colX(j)).attr('x2', colX(j)).attr('y1', ys[0])
                        .attr('y2', ys[ys.length - 1]).attr('stroke', colour).attr('stroke-width', 2);
                }
                ys.forEach((y) => matrix.append('circle').attr('cx', colX(j)).attr('cy', y).attr('r', 4.5).attr('fill', colour));
            });

            // Bucket names and totals.
            const side = svg.append('g');
            sets.forEach((s, i) => {
                const on = selSet === s.name;
                side.append('text').attr('x', LABEL_W - 6).attr('y', rowY(i) + 4).attr('text-anchor', 'end')
                    .attr('font-weight', on ? 700 : 400).text(s.name);
                const bw = xTot(s.members.length);
                side.append('rect').attr('x', LABEL_W + TOTAL_W - bw).attr('y', rowY(i) - 7)
                    .attr('width', bw).attr('height', 14).attr('fill', on ? C.pick : C.total);
                side.append('text').attr('x', LABEL_W + TOTAL_W - bw - 3).attr('y', rowY(i) + 4)
                    .attr('text-anchor', 'end').attr('font-size', 10).text(s.members.length);
            });
            svg.append('text').attr('x', LABEL_W + TOTAL_W).attr('y', h - 6).attr('text-anchor', 'end')
                .attr('font-size', 11).text('AOPs per organ system');

            // Hit areas last, so they sit on top. Not exported as visible marks.
            const hits = svg.append('g').attr('class', 'upset-hits');
            cols.forEach((c, j) => {
                hits.append('rect').attr('x', left + j * colW).attr('y', 0).attr('width', colW)
                    .attr('height', BAR_H + GAP + ROW_H * sets.length).attr('fill', 'transparent')
                    .style('cursor', 'pointer')
                    .on('click', () => select({ kind: 'combo', i: j }))
                    .append('title').text(`${comboName(c)}: ${c.members.length} AOPs`);
            });
            sets.forEach((s, i) => {
                hits.append('rect').attr('x', 0).attr('y', rowY(i) - ROW_H / 2).attr('width', left)
                    .attr('height', ROW_H).attr('fill', 'transparent').style('cursor', 'pointer')
                    .on('click', () => select({ kind: 'set', i }))
                    .append('title').text(`${s.name}: ${s.members.length} AOPs in total`);
            });
        }

        function renderTable() {
            const p = picked();
            if (!p) {
                table.innerHTML = '<p class="upset-hint">Click a column or an organ system, or focus the chart and use the arrow keys, to list its AOPs here.</p>';
                return;
            }
            const rows = p.members.slice(0, TABLE_MAX);
            const cell = (k, i) => escapeHtml(data[k][i] == null ? '' : data[k][i]);
            table.innerHTML = `
                <div class="data-table-wrapper"><table class="raw-data-table upset-rows">
                    <caption>${p.members.length} AOPs: ${escapeHtml(p.title)}</caption>
                    <thead><tr><th scope="col">AOP</th><th scope="col">Title</th><th scope="col">Organ systems (${escapeHtml(SCOPES[scope].label)})</th></tr></thead>
                    <tbody>${rows.map((i) => `<tr>
                        <td>${linkable(data.aop[i]) ? `<a href="${escapeHtml(data.aop[i])}" target="_blank" rel="noopener">${cell('aop_id', i)}</a>` : cell('aop_id', i)}</td>
                        <td>${cell('title', i)}</td><td>${escapeHtml(setsOf(data[column][i]).join(', '))}</td></tr>`).join('')}
                    </tbody></table></div>
                ${p.members.length > TABLE_MAX ? `<div class="data-table-truncated">Showing the first ${TABLE_MAX} of ${p.members.length}; download the CSV for all.</div>` : ''}`;
        }

        function select(next) {
            selection = next;
            draw();
            renderTable();
            const p = picked();
            live.textContent = p ? `${p.title}: ${p.members.length} AOPs.` : 'Selection cleared.';
        }

        function apply() {
            q('.upset-summary').textContent = summary(data, scope);
            const extra = combos.length - TOP;
            toggle.hidden = extra <= 0;
            toggle.textContent = pref.showAll ? `Show the ${TOP} largest` : `Show all ${combos.length} combinations`;
            toggle.setAttribute('aria-pressed', pref.showAll ? 'true' : 'false');
            const name = `Organ-system overlap of ${data.aop.length} AOPs (${SCOPES[scope].label}): `
                + `${shown().length} of ${combos.length} combinations shown`;
            element.setAttribute('aria-label', name);
            chart.setAttribute('aria-label', `${name}. Left and right arrows step through the combinations, `
                + 'up and down through the organ systems; each lists its AOPs below the chart. Escape clears.');
            draw();
            renderTable();
        }

        function layout() {
            const w = Math.floor(scroller.getBoundingClientRect().width);
            if (!w || w === width) return;
            width = w;
            draw();
        }

        chart.addEventListener('keydown', (event) => {
            const next = step(selection, event.key, shown().length, sets.length);
            if (next === undefined) return;
            event.preventDefault();
            select(next);
        });
        toggle.addEventListener('click', () => {
            pref.showAll = !pref.showAll;
            if (selection && selection.kind === 'combo' && selection.i >= shown().length) selection = null;
            apply();
        });
        const resize = root.ResizeObserver ? new ResizeObserver(layout) : null;
        if (resize) resize.observe(scroller);

        width = Math.floor(scroller.getBoundingClientRect().width) || 900;
        apply();
        views.set(element, { destroy() { if (resize) resize.disconnect(); } });
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
    if (root.PlotRuntime) root.PlotRuntime.registerRenderer('d3-upset', renderer);
    root.DashUpset.renderer = renderer;
})(typeof window !== 'undefined' ? window : this);
