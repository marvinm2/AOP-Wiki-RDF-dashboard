/**
 * Entity lifeline (#183): when each AOP, KE, KER and stressor appears,
 * vanishes and comes back. Renderer `d3-lifeline` for the plot runtime.
 *
 * - Data: /api/v2/data/entity_lifeline, one row per entity ever seen with a
 *   presence string ("x" present, "-" absent, oldest snapshot first). See
 *   plot_entity_lifeline in plots/trends_plots.py.
 * - Drawing: the presence grid is painted at one pixel per cell into an
 *   offscreen canvas, then scaled into the chart column by column, so rows
 *   average into a density when there are more rows than pixels while the
 *   columns stay crisp. 100K cells draw in a few milliseconds
 *   (data-draw-ms on the container records it).
 * - One entity type at a time (buttons), optionally one status (select); both
 *   live in the page state (?entity=, ?status=).
 * - Selecting rows (a y-brush, or the arrow keys on the focused chart) lists
 *   them in a table below, with links to AOP-Wiki. Shift with the arrows
 *   resizes the selection, Page Up/Down moves it by its own height, Escape
 *   clears it. Hovering a cell names the entity and the snapshot.
 * - Colours: the entity type's brand colour from static/data/tokens.json, in
 *   the same order as the birth/death plot. No transitions.
 *
 * The pure helpers are exported for node --test.
 */
(function (root) {
    'use strict';

    const TYPES = ['AOPs', 'KEs', 'KERs', 'Stressors'];
    const STATUSES = ['always present', 'added', 'returned', 'removed'];
    const PRESENT = 'x';
    const DEFAULT_TYPE = 'KEs';
    const WINDOW = 20;       // rows a keyboard selection starts with
    const TABLE_MAX = 200;   // rows the linked table lists

    // ------------------------------------------------------------ pure parts

    /** Row indices of `type` (and `status`, if given), in data order. */
    function rowsFor(data, type, status) {
        const out = [];
        for (let i = 0; i < data.entity_type.length; i++) {
            if (data.entity_type[i] === type && (!status || data.status[i] === status)) out.push(i);
        }
        return out;
    }

    /** Rows [a, b) under the pixel span [y0, y1] of a chart `height` px tall holding `n` rows. */
    function rowSpan(y0, y1, height, n) {
        const a = Math.max(0, Math.min(n - 1, Math.floor((Math.min(y0, y1) / height) * n)));
        const b = Math.min(n, Math.max(a + 1, Math.ceil((Math.max(y0, y1) / height) * n)));
        return [a, b];
    }

    /**
     * The selection a key leaves: [a, b), null (cleared), or undefined for a
     * key that does nothing here. Without a selection, a key starts one at the
     * top (or the bottom, for End).
     */
    function moveWindow(win, key, shift, n) {
        if (key === 'Escape') return null;
        const keys = ['ArrowDown', 'ArrowUp', 'PageDown', 'PageUp', 'Home', 'End'];
        if (!keys.includes(key) || n === 0) return undefined;
        const size = Math.min(WINDOW, n);
        if (!win) return key === 'End' ? [n - size, n] : [0, size];
        let [a, b] = win;
        const len = b - a;
        const shiftBy = (d) => {
            const na = Math.max(0, Math.min(n - len, a + d));
            return [na, na + len];
        };
        switch (key) {
            case 'ArrowDown': return shift ? [a, Math.min(n, b + 1)] : shiftBy(1);
            case 'ArrowUp': return shift ? [a, Math.max(a + 1, b - 1)] : shiftBy(-1);
            case 'PageDown': return shiftBy(len);
            case 'PageUp': return shiftBy(-len);
            case 'Home': return [0, len];
            default: return [n - len, n];  // End
        }
    }

    /** [r, g, b] from "#rrggbb". */
    function rgb(hex) {
        const h = String(hex).replace('#', '');
        return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
    }

    /**
     * Paint presence strings into RGBA pixels: one row per string, one
     * column per character.
     */
    function fillPixels(pixels, presences, on, off) {
        const cols = presences.length ? presences[0].length : 0;
        presences.forEach((p, r) => {
            for (let c = 0; c < cols; c++) {
                const k = (r * cols + c) * 4;
                const col = p[c] === PRESENT ? on : off;
                pixels[k] = col[0]; pixels[k + 1] = col[1]; pixels[k + 2] = col[2]; pixels[k + 3] = 255;
            }
        });
        return pixels;
    }

    /** "2,907 KERs ever: 880 always present, 1,487 added, 14 returned, 526 removed." */
    function summary(data, type) {
        const rows = rowsFor(data, type, null);
        const counts = STATUSES.map((s) => [s, rows.filter((i) => data.status[i] === s).length]);
        const fmt = (n) => n.toLocaleString('en-US');
        return `${fmt(rows.length)} ${type} ever: `
            + counts.map(([s, c]) => `${fmt(c)} ${s}`).join(', ') + '.';
    }

    const api = { TYPES, STATUSES, rowsFor, rowSpan, moveWindow, rgb, fillPixels, summary };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
        return;
    }
    root.DashLifeline = api;

    // ------------------------------------------------------------- the view

    const HEIGHT = 420;
    const AXIS = 22;

    let tokens = null;
    function getTokens() {
        if (!tokens) {
            tokens = fetch('/static/data/tokens.json').then((r) => r.json())
                .catch(() => ({ color: { palette: ['#29235C', '#E6007E', '#307BBF', '#009FE3'] } }));
        }
        return tokens;
    }

    function absentColour() {
        const v = getComputedStyle(document.documentElement).getPropertyValue('--color-border-light').trim();
        return /^#[0-9a-f]{6}$/i.test(v) ? v : '#f0f0f0';
    }

    function escapeHtml(text) {
        return String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    const views = new WeakMap();  // container -> {destroy}

    async function mount(element, payload) {
        destroy(element);
        const d3 = root.d3;
        if (!d3) {
            element.innerHTML = '<p class="lifeline-note">The lifeline needs JavaScript charting that did not load. '
                + 'Use "Show Raw Data" below or the CSV download.</p>';
            return;
        }
        const data = payload.data;
        const versions = Array.from(data.presence[0] || '', (_, i) => i);
        const palette = (await getTokens()).color.palette;
        const colourOf = (type) => palette[TYPES.indexOf(type)] || palette[0];
        const state = root.DashState;
        const snapshots = (await fetch('/api/versions').then((r) => r.json()).catch(() => ({ versions: [] })))
            .versions.map((v) => v.version).sort();
        const label = (j) => snapshots[j] || `snapshot ${j + 1}`;

        element.dataset.a11y = 'self';
        element.setAttribute('role', 'figure');
        element.innerHTML = `
            <div class="lifeline-controls">
                <div class="segmented-control" role="group" aria-label="Entity type">
                    ${TYPES.map((t) => `<button type="button" class="segmented-button" data-type="${t}" aria-pressed="false">${t} <span class="lifeline-count">${rowsFor(data, t, null).length}</span></button>`).join('')}
                </div>
                <label class="lifeline-status-label">Status
                    <select class="lifeline-status">
                        <option value="">All</option>
                        ${STATUSES.map((s) => `<option value="${s}">${s[0].toUpperCase() + s.slice(1)}</option>`).join('')}
                    </select>
                </label>
            </div>
            <p class="lifeline-summary"></p>
            <div class="lifeline-chart" tabindex="0" role="group" aria-roledescription="row selector">
                <svg class="lifeline-axis" height="${AXIS}" aria-hidden="true"></svg>
                <div class="lifeline-stage">
                    <canvas class="lifeline-canvas" aria-hidden="true"></canvas>
                    <svg class="lifeline-overlay" aria-hidden="true"></svg>
                    <div class="lifeline-tip" hidden></div>
                </div>
            </div>
            <ul class="lifeline-legend" aria-hidden="true">
                <li><span class="lifeline-swatch lifeline-swatch--on"></span>In the snapshot</li>
                <li><span class="lifeline-swatch lifeline-swatch--off"></span>Not in the snapshot</li>
            </ul>
            <p class="sr-only lifeline-live" aria-live="polite"></p>
            <div class="lifeline-table"></div>`;

        const q = (s) => element.querySelector(s);
        const chart = q('.lifeline-chart');
        const stage = q('.lifeline-stage');
        const canvas = q('.lifeline-canvas');
        const tip = q('.lifeline-tip');
        const live = q('.lifeline-live');
        const table = q('.lifeline-table');
        const statusSelect = q('.lifeline-status');
        const axis = d3.select(q('.lifeline-axis'));
        const overlay = d3.select(q('.lifeline-overlay'));

        let rows = [];
        let selection = null;  // [a, b) into rows
        let width = 0;
        const x = d3.scaleBand().domain(versions).paddingInner(0.12);
        const brush = d3.brushY();
        let moving = false;

        function current() {
            const t = state && state.get('entity');
            return {
                type: TYPES.includes(t) ? t : DEFAULT_TYPE,
                status: state && STATUSES.includes(state.get('status')) ? state.get('status') : null,
            };
        }

        function draw() {
            const t0 = performance.now();
            const { type } = current();
            const dpr = root.devicePixelRatio || 1;
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(HEIGHT * dpr);
            canvas.style.width = `${width}px`;
            canvas.style.height = `${HEIGHT}px`;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (rows.length) {
                const cols = versions.length;
                const off = document.createElement('canvas');
                off.width = cols;
                off.height = rows.length;
                const octx = off.getContext('2d');
                const img = octx.createImageData(cols, rows.length);
                fillPixels(img.data, rows.map((i) => data.presence[i]), rgb(colourOf(type)), rgb(absentColour()));
                octx.putImageData(img, 0, 0);
                // More rows than pixels: average them (a density); fewer: keep edges sharp.
                ctx.imageSmoothingEnabled = rows.length > canvas.height;
                ctx.imageSmoothingQuality = 'high';
                versions.forEach((j) => {
                    ctx.drawImage(off, j, 0, 1, rows.length,
                        x(j) * dpr, 0, Math.max(1, x.bandwidth() * dpr), canvas.height);
                });
            }
            element.dataset.drawMs = (performance.now() - t0).toFixed(1);
        }

        function drawAxis() {
            const years = versions.filter((j, i) => i === 0 || label(j).slice(0, 4) !== label(j - 1).slice(0, 4));
            const step = width < 600 ? 2 : 1;
            axis.attr('width', width).selectAll('*').remove();
            axis.append('g').attr('transform', `translate(0,${AXIS - 2})`)
                .call(d3.axisTop(x).tickValues(years.filter((_, i) => i % step === 0))
                    .tickFormat((j) => label(j).slice(0, 4)).tickSizeOuter(0));
        }

        function rowsToPx([a, b]) {
            return [(a / rows.length) * HEIGHT, (b / rows.length) * HEIGHT];
        }

        function showSelection() {
            moving = true;
            overlay.select('.lifeline-brush').call(brush.move, selection ? rowsToPx(selection) : null);
            moving = false;
            renderTable();
        }

        function renderTable() {
            const { type } = current();
            if (!selection) {
                table.innerHTML = `<p class="lifeline-hint">Drag over the rows, or focus the chart and use the arrow keys, to list ${type} here.</p>`;
                return;
            }
            const [a, b] = selection;
            const picked = rows.slice(a, b);
            const shown = picked.slice(0, TABLE_MAX);
            const cell = (k, i) => escapeHtml(data[k][i] == null ? '' : data[k][i]);
            table.innerHTML = `
                <div class="data-table-wrapper"><table class="raw-data-table lifeline-rows">
                    <caption>${picked.length.toLocaleString('en-US')} ${type} selected (rows ${a + 1} to ${b} of ${rows.length.toLocaleString('en-US')})</caption>
                    <thead><tr><th scope="col">ID</th><th scope="col">Name</th><th scope="col">First seen</th><th scope="col">Last seen</th><th scope="col">Snapshots</th><th scope="col">Returns</th><th scope="col">Status</th></tr></thead>
                    <tbody>${shown.map((i) => `<tr>
                        <td>${data.aopwiki_url[i] ? `<a href="${escapeHtml(data.aopwiki_url[i])}" target="_blank" rel="noopener">${cell('id', i)}</a>` : cell('id', i)}</td>
                        <td>${cell('label', i)}</td><td>${cell('first_seen', i)}</td><td>${cell('last_seen', i)}</td>
                        <td>${cell('snapshots_present', i)}</td><td>${cell('returns', i)}</td><td>${cell('status', i)}</td></tr>`).join('')}
                    </tbody></table></div>
                ${picked.length > TABLE_MAX ? `<div class="data-table-truncated">Showing the first ${TABLE_MAX} of ${picked.length}; narrow the selection or download the CSV.</div>` : ''}`;
        }

        function apply() {
            const { type, status } = current();
            rows = rowsFor(data, type, status);
            selection = null;
            element.querySelectorAll('[data-type]').forEach((btn) => {
                btn.setAttribute('aria-pressed', btn.dataset.type === type ? 'true' : 'false');
                btn.classList.toggle('is-active', btn.dataset.type === type);
            });
            statusSelect.value = status || '';
            q('.lifeline-summary').textContent = summary(data, type)
                + (status ? ` Showing ${rows.length.toLocaleString('en-US')} ${status}.` : '');
            element.style.setProperty('--lifeline-on', colourOf(type));
            const name = `Entity lifeline of ${rows.length.toLocaleString('en-US')} ${status ? `${status} ` : ''}${type}`;
            element.setAttribute('aria-label', name);
            chart.setAttribute('aria-label', `${name}, one row each, oldest snapshot on the left. `
                + 'Up and down arrows select rows, with Shift to resize the selection; Escape clears it.');
            draw();
            showSelection();
        }

        function layout() {
            const w = Math.floor(stage.getBoundingClientRect().width);
            if (!w || w === width) return;
            width = w;
            x.range([0, width]);
            overlay.attr('width', width).attr('height', HEIGHT);
            brush.extent([[0, 0], [width, HEIGHT]]);
            let g = overlay.select('.lifeline-brush');
            if (g.empty()) g = overlay.append('g').attr('class', 'lifeline-brush');
            g.call(brush);
            drawAxis();
            draw();
            showSelection();
        }

        brush.on('end', (event) => {
            if (moving || !event.sourceEvent) return;
            selection = event.selection ? rowSpan(event.selection[0], event.selection[1], HEIGHT, rows.length) : null;
            showSelection();
        });

        overlay.on('pointermove', (event) => {
            if (!rows.length) return;
            const [px, py] = d3.pointer(event, stage);
            const r = Math.min(rows.length - 1, Math.max(0, Math.floor((py / HEIGHT) * rows.length)));
            const j = versions.find((v) => px >= x(v) && px <= x(v) + x.bandwidth());
            if (j === undefined) { tip.hidden = true; return; }
            const i = rows[r];
            tip.textContent = `${data.id[i]} · ${data.label[i]} · ${label(j)}: ${data.presence[i][j] === PRESENT ? 'present' : 'absent'}`;
            tip.hidden = false;
            tip.style.left = `${Math.min(px + 12, width - 260)}px`;
            tip.style.top = `${py + 12}px`;
        }).on('pointerleave', () => { tip.hidden = true; });

        chart.addEventListener('keydown', (event) => {
            const next = moveWindow(selection, event.key, event.shiftKey, rows.length);
            if (next === undefined) return;
            event.preventDefault();
            selection = next;
            showSelection();
            const { type } = current();
            live.textContent = selection
                ? `Rows ${selection[0] + 1} to ${selection[1]} of ${rows.length} ${type} selected.`
                : 'Selection cleared.';
        });

        element.querySelectorAll('[data-type]').forEach((btn) => btn.addEventListener('click', () => {
            if (state) state.set({ entity: btn.dataset.type === DEFAULT_TYPE ? null : btn.dataset.type });
        }));
        statusSelect.addEventListener('change', () => {
            if (state) state.set({ status: statusSelect.value || null });
        });
        const unsubscribe = state ? state.subscribe((changed) => {
            if (changed.includes('entity') || changed.includes('status')) apply();
        }) : () => {};
        const resize = root.ResizeObserver ? new ResizeObserver(layout) : null;
        if (resize) resize.observe(stage);

        layout();
        apply();
        views.set(element, {
            destroy() { unsubscribe(); if (resize) resize.disconnect(); },
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
    if (root.PlotRuntime) root.PlotRuntime.registerRenderer('d3-lifeline', renderer);
    root.DashLifeline.renderer = renderer;
})(typeof window !== 'undefined' ? window : this);
