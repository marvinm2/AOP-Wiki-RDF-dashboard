/**
 * AOP-AOP overlap (#185): a cluster-ordered matrix and a graph of the same
 * pairs, with one cluster picker and a linked table.
 *
 * - Data: /api/v2/data/latest_aop_aop_overlap, one row per AOP pair whose Key
 *   Event sets reach the Jaccard threshold, with its cluster and its position
 *   in the matrix (plot_latest_aop_aop_overlap in plots/latest_plots.py). The
 *   threshold (min_jaccard) and the version come from the page state; the
 *   runtime refetches when either changes.
 * - Matrix tab: the server's Plotly heatmap (the default renderer). Graph tab:
 *   renderer `cytoscape-overlap`, which loads Cytoscape the first time the tab
 *   is opened. The tab is kept in the URL (?overlap=graph).
 * - Picking a cluster (the "Cluster" select, a matrix cell or a graph node)
 *   zooms the matrix to the cluster's block, centres the graph on it, fades
 *   the rest, and lists its pairs below, linked to identifiers.org. Escape
 *   clears. Everything is reachable from the keyboard: the tabs (arrow keys),
 *   the select, the graph's zoom buttons and the table.
 * - Colours: clusters of 3+ AOPs take the brand palette in cluster order,
 *   without magenta; pairs are grey; magenta marks identical KE sets. The
 *   server's outlines follow the same rule (overlap_cluster_colours).
 * - No animation anywhere: Cytoscape layouts and fits run without it.
 *
 * The pure helpers are exported for node --test.
 */
(function (root) {
    'use strict';

    const PLOT = 'latest_aop_aop_overlap';
    const PAIR_GREY = '#B8B8B8';
    const MAGENTA = '#E6007E';

    // ------------------------------------------------------------ pure parts

    /**
     * Clusters of the rows, in cluster order: [{id, size, pairs, identical,
     * lo, hi, aops: [AOP ids in matrix order]}]. `lo`..`hi` is the cluster's
     * block on the matrix diagonal.
     */
    function clusters(data) {
        const byId = new Map();
        const ids = data.cluster_id || [];
        ids.forEach((id, i) => {
            if (!byId.has(id)) {
                byId.set(id, { id, size: data.cluster_size[i], pairs: 0, identical: 0, lo: Infinity, hi: -Infinity, at: new Map() });
            }
            const c = byId.get(id);
            c.pairs += 1;
            if (data.identical_ke_set[i]) c.identical += 1;
            c.lo = Math.min(c.lo, data.matrix_row[i]);
            c.hi = Math.max(c.hi, data.matrix_col[i]);
            c.at.set(data.matrix_row[i], data.aop1_id[i]);
            c.at.set(data.matrix_col[i], data.aop2_id[i]);
        });
        return Array.from(byId.values()).sort((a, b) => a.id - b.id).map((c) => ({
            id: c.id, size: c.size, pairs: c.pairs, identical: c.identical, lo: c.lo, hi: c.hi,
            aops: Array.from(c.at.keys()).sort((a, b) => a - b).map((k) => c.at.get(k)),
        }));
    }

    /** {cluster id: colour}: palette (no magenta) for clusters of 3+, grey for pairs. */
    function colours(list, palette) {
        const hues = (palette || []).filter((c) => c.toUpperCase() !== MAGENTA);
        const out = {};
        let k = 0;
        list.forEach((c) => {
            out[c.id] = c.size >= 3 && hues.length ? hues[k++ % hues.length] : PAIR_GREY;
        });
        return out;
    }

    /** The cluster of the pair (a, b) of AOP ids, either way round, or null. */
    function clusterOfPair(data, a, b) {
        const ids1 = data.aop1_id || [];
        for (let i = 0; i < ids1.length; i++) {
            const x = ids1[i], y = data.aop2_id[i];
            if ((x === a && y === b) || (x === b && y === a)) return data.cluster_id[i];
        }
        return null;
    }

    /** The cluster an AOP id belongs to, or null. */
    function clusterOfAop(data, aop) {
        const ids1 = data.aop1_id || [];
        for (let i = 0; i < ids1.length; i++) {
            if (ids1[i] === aop || data.aop2_id[i] === aop) return data.cluster_id[i];
        }
        return null;
    }

    /** Row indices of a cluster's pairs, in matrix order. */
    function pairsOf(data, id) {
        return (data.cluster_id || []).map((c, i) => (c === id ? i : -1)).filter((i) => i >= 0);
    }

    /** "Cluster 1 · 25 AOPs" or, for a pair, "Cluster 37 · AOP 30 + AOP 31". */
    function optionLabel(c) {
        return c.size === 2 ? `Cluster ${c.id} · ${c.aops.join(' + ')}` : `Cluster ${c.id} · ${c.size} AOPs`;
    }

    /** The matrix ranges that show a cluster's block (y is drawn top-down). */
    function blockRanges(c) {
        return { x: [c.lo - 0.5, c.hi + 0.5], y: [c.hi + 0.5, c.lo - 0.5] };
    }

    /**
     * Where each cluster goes in the graph: shelves of cells left to right, a
     * cell's side growing with the square root of its size. {id: {x, y, w, h}}.
     */
    function cells(list, width) {
        const out = {};
        const unit = 46;
        let x = 0, y = 0, shelf = 0;
        list.forEach((c) => {
            // Pairs get room for their two labels side by side.
            const side = unit * Math.max(2.2, Math.sqrt(c.size) * 1.5);
            if (x > 0 && x + side > width) { x = 0; y += shelf; shelf = 0; }
            out[c.id] = { x, y, w: side, h: side };
            x += side;
            shelf = Math.max(shelf, side);
        });
        return out;
    }

    /** The tab a key moves to from tab `i` of `n` (ARIA tabs pattern), or undefined. */
    function tabStep(key, i, n) {
        switch (key) {
            case 'ArrowRight': return (i + 1) % n;
            case 'ArrowLeft': return (i - 1 + n) % n;
            case 'Home': return 0;
            case 'End': return n - 1;
            default: return undefined;
        }
    }

    /** Only identifiers.org IRIs become links. */
    function linkable(uri) {
        return typeof uri === 'string' && uri.startsWith('https://identifiers.org/');
    }

    const api = { clusters, colours, clusterOfPair, clusterOfAop, pairsOf, optionLabel, blockRanges, cells, tabStep, linkable };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
        return;
    }
    root.DashOverlap = api;

    // ------------------------------------------------------------- the page

    function escapeHtml(text) {
        return String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    let tokens = null;
    function getTokens() {
        if (!tokens) {
            tokens = fetch('/static/data/tokens.json').then((r) => r.json())
                .catch(() => ({ color: { primary: '#29235C', magenta: MAGENTA, palette: ['#29235C', '#307BBF'] } }));
        }
        return tokens;
    }

    const payloads = new Map();   // data URL -> promise of the payload
    function dataFor(url) {
        if (!payloads.has(url)) {
            payloads.set(url, fetch(url).then((r) => {
                if (!r.ok) throw new Error(`${url}: ${r.status}`);
                return r.json();
            }).catch((e) => { payloads.delete(url); throw e; }));
        }
        return payloads.get(url);
    }

    function dataUrl() {
        const query = root.plotLoader ? root.plotLoader.query(PLOT) : '';
        return `/api/v2/data/${PLOT}${query}`;
    }

    /** The one controller of the card: tabs, picker, table, and both views. */
    const card = {
        root: null,
        url: null,        // data URL the picker and table were built from
        data: null,
        list: [],
        picked: null,     // cluster id or null
        graph: null,      // {focus(id), fit()} once the graph is drawn
    };

    function q(sel) { return card.root.querySelector(sel); }
    function matrixContainer() { return q('.overlap-panel--matrix .lazy-plot'); }
    function matrixDiv() { const m = matrixContainer(); return m && m.querySelector('.js-plotly-plot'); }

    async function refresh() {
        const url = dataUrl();
        if (card.url === url) return;
        card.url = url;
        card.picked = null;
        const select = q('.overlap-cluster');
        select.disabled = true;
        let payload;
        try {
            payload = await dataFor(url);
        } catch (e) {
            if (card.url === url) q('.overlap-table').innerHTML = '<p class="overlap-hint">The cluster list is unavailable; use "Show Raw Data" below or the CSV download.</p>';
            return;
        }
        if (card.url !== url) return;  // the state moved on
        card.data = payload.data;
        card.list = clusters(card.data);
        select.innerHTML = '<option value="">All clusters</option>'
            + card.list.map((c) => `<option value="${c.id}">${escapeHtml(optionLabel(c))}</option>`).join('');
        select.disabled = !card.list.length;
        show();
    }

    function pick(id, announce) {
        const c = card.list.find((x) => x.id === id) || null;
        card.picked = c ? c.id : null;
        show();
        if (announce !== false) {
            q('.overlap-live').textContent = c
                ? `${optionLabel(c)}: ${c.pairs} pair${c.pairs === 1 ? '' : 's'} listed below.`
                : 'Selection cleared.';
        }
    }

    function show() {
        const select = q('.overlap-cluster');
        select.value = card.picked === null ? '' : String(card.picked);
        const c = card.list.find((x) => x.id === card.picked) || null;
        zoomMatrix(c);
        if (card.graph) card.graph.focus(c ? c.id : null);
        renderTable(c);
    }

    function zoomMatrix(c) {
        const div = matrixDiv();
        if (!div || !root.Plotly) return;
        const update = c
            ? { 'xaxis.autorange': false, 'yaxis.autorange': false,
                'xaxis.range': blockRanges(c).x, 'yaxis.range': blockRanges(c).y }
            : { 'xaxis.autorange': true, 'yaxis.autorange': 'reversed' };
        Promise.resolve(root.Plotly.relayout(div, update)).catch(() => {});
    }

    function renderTable(c) {
        const table = q('.overlap-table');
        if (!c) {
            table.innerHTML = '<p class="overlap-hint">Pick a cluster above, click a cell in the matrix or an AOP in the graph to list its pairs here.</p>';
            return;
        }
        const d = card.data;
        const link = (uri, id) => (linkable(uri)
            ? `<a href="${escapeHtml(uri)}" target="_blank" rel="noopener">${escapeHtml(id)}</a>` : escapeHtml(id));
        const rows = pairsOf(d, c.id).map((i) => `<tr>
            <td>${link(d.aop1[i], d.aop1_id[i])}</td><td>${escapeHtml(d.aop1_title[i] || '')}</td>
            <td>${link(d.aop2[i], d.aop2_id[i])}</td><td>${escapeHtml(d.aop2_title[i] || '')}</td>
            <td>${Number(d.jaccard[i]).toFixed(2)}</td><td>${d.shared_kes[i]}</td>
            <td>${d.identical_ke_set[i] ? 'yes' : ''}</td></tr>`).join('');
        table.innerHTML = `
            <div class="data-table-wrapper"><table class="raw-data-table overlap-rows">
                <caption>Cluster ${c.id}: ${c.size} AOPs, ${c.pairs} pair${c.pairs === 1 ? '' : 's'} at or above the threshold</caption>
                <thead><tr><th scope="col">AOP</th><th scope="col">Title</th><th scope="col">AOP</th><th scope="col">Title</th>
                    <th scope="col">Jaccard</th><th scope="col">Shared KEs</th><th scope="col">Identical KE set</th></tr></thead>
                <tbody>${rows}</tbody></table></div>`;
    }

    // ---- tabs

    function tabs() { return Array.from(card.root.querySelectorAll('.overlap-tabs [role="tab"]')); }

    function selectTab(name, focus) {
        tabs().forEach((tab) => {
            const on = tab.dataset.tab === name;
            tab.setAttribute('aria-selected', on ? 'true' : 'false');
            tab.tabIndex = on ? 0 : -1;
            tab.classList.toggle('is-active', on);
            document.getElementById(tab.getAttribute('aria-controls')).hidden = !on;
            if (on && focus) tab.focus();
        });
        // The graph only loads once its panel is shown; a re-shown matrix needs its size back.
        if (root.plotLoader) root.plotLoader.loadVisible();
        const div = matrixDiv();
        if (name === 'matrix' && div && root.Plotly) Promise.resolve(root.Plotly.Plots.resize(div)).catch(() => {});
        if (name === 'graph' && card.graph) card.graph.resize();
    }

    function wireTabs() {
        const state = root.DashState;
        const list = tabs();
        list.forEach((tab, i) => {
            tab.addEventListener('click', () => state.set({ overlap: tab.dataset.tab === 'graph' ? 'graph' : null }));
            tab.addEventListener('keydown', (event) => {
                const next = tabStep(event.key, i, list.length);
                if (next === undefined) return;
                event.preventDefault();
                state.set({ overlap: list[next].dataset.tab === 'graph' ? 'graph' : null });
                list[next].focus();
            });
        });
        const apply = () => selectTab(state.get('overlap') === 'graph' ? 'graph' : 'matrix');
        state.subscribe((changed) => { if (changed.includes('overlap')) apply(); });
        apply();
    }

    function init() {
        card.root = document.querySelector('.overlap-view');
        if (!card.root || !root.DashState) return;
        wireTabs();
        q('.overlap-cluster').addEventListener('change', (event) => {
            pick(event.target.value === '' ? null : Number(event.target.value));
        });
        card.root.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && card.picked !== null) { event.preventDefault(); pick(null); }
        });
        renderTable(null);

        document.addEventListener('plot:rendered', (event) => {
            if (event.detail.plotName !== PLOT) return;
            const container = event.target;
            if (container === matrixContainer()) {
                container.setAttribute('role', 'figure');
                container.setAttribute('aria-label', 'AOP-AOP overlap matrix, ordered by cluster. '
                    + 'Use the Cluster list above to zoom to a cluster and list its pairs below.');
                const div = matrixDiv();
                if (div && typeof div.on === 'function') {
                    div.on('plotly_click', (e) => {
                        const p = e && e.points && e.points[0];
                        if (!p || !card.data) return;
                        const id = clusterOfPair(card.data, String(p.x), String(p.y));
                        if (id !== null) pick(id);
                    });
                }
            }
            refresh().then(() => { if (container === matrixContainer() && card.picked !== null) show(); });
        });
    }

    // ---------------------------------------------------- the graph renderer

    let cytoscapeLoad = null;
    function loadCytoscape(element) {
        if (root.cytoscape) return Promise.resolve(root.cytoscape);
        if (!cytoscapeLoad) {
            cytoscapeLoad = new Promise((resolve, reject) => {
                const s = document.createElement('script');
                s.src = element.dataset.libSrc;
                if (element.dataset.libIntegrity) {
                    s.integrity = element.dataset.libIntegrity;
                    s.crossOrigin = 'anonymous';
                }
                s.onload = () => resolve(root.cytoscape);
                s.onerror = () => { cytoscapeLoad = null; reject(new Error('Cytoscape did not load')); };
                document.head.appendChild(s);
            });
        }
        return cytoscapeLoad;
    }

    const views = new WeakMap();   // container -> {cy, ro}

    async function mountGraph(element, payload) {
        destroyGraph(element);
        const data = payload.data;
        element.dataset.a11y = 'self';
        if (!data.cluster_id || !data.cluster_id.length) {
            element.innerHTML = '<p class="overlap-hint">No AOP pairs reach this threshold.</p>';
            return;
        }
        let cytoscape;
        try {
            cytoscape = await loadCytoscape(element);
        } catch (e) {
            element.innerHTML = '<p class="overlap-hint">The graph needs a script that did not load. The matrix tab and the CSV hold the same pairs.</p>';
            return;
        }
        const colour = (await getTokens()).color;
        const list = clusters(data);
        const hue = colours(list, colour.palette);

        element.innerHTML = `
            <div class="overlap-graph-tools" role="group" aria-label="Graph zoom">
                <button type="button" class="segmented-button" data-zoom="in" aria-label="Zoom in">+</button>
                <button type="button" class="segmented-button" data-zoom="out" aria-label="Zoom out">&minus;</button>
                <button type="button" class="segmented-button" data-zoom="fit">Fit</button>
            </div>
            <div class="overlap-graph-canvas"></div>`;
        const canvas = element.querySelector('.overlap-graph-canvas');
        const nAops = new Set(data.aop1.concat(data.aop2)).size;
        element.setAttribute('role', 'figure');
        element.setAttribute('aria-label', `AOP-AOP overlap graph: ${nAops} AOPs in ${list.length} clusters, `
            + `${data.aop1.length} pairs. Use the Cluster list above to centre a cluster and list its pairs below.`);
        canvas.setAttribute('aria-hidden', 'true');

        const nodes = new Map();
        const addNode = (uri, id, title, nKes, cluster) => {
            if (!nodes.has(uri)) nodes.set(uri, { data: { id: uri, label: id, title, kes: nKes, cluster, colour: hue[cluster] } });
        };
        const edges = data.aop1.map((a, i) => {
            addNode(a, data.aop1_id[i], data.aop1_title[i], data.n_kes_aop1[i], data.cluster_id[i]);
            addNode(data.aop2[i], data.aop2_id[i], data.aop2_title[i], data.n_kes_aop2[i], data.cluster_id[i]);
            return { data: { id: `e${i}`, source: a, target: data.aop2[i], jaccard: data.jaccard[i],
                identical: data.identical_ke_set[i] ? 1 : 0 } };
        });
        const cy = cytoscape({
            container: canvas,
            elements: { nodes: Array.from(nodes.values()), edges },
            userZoomingEnabled: false,   // the page scrolls under the wheel; zoom with the buttons
            boxSelectionEnabled: false,
            autoungrabify: true,
            style: [
                { selector: 'node', style: {
                    'background-color': 'data(colour)', width: 'mapData(kes, 1, 40, 10, 30)', height: 'mapData(kes, 1, 40, 10, 30)',
                    label: 'data(label)', 'font-size': 9, color: colour.primary, 'text-valign': 'top', 'text-margin-y': -2,
                    'min-zoomed-font-size': 7, 'border-width': 1, 'border-color': '#FFFFFF' } },
                { selector: 'edge', style: {
                    width: 'mapData(jaccard, 0.3, 1, 1, 4)', 'line-color': 'rgba(50,50,80,0.35)', 'curve-style': 'straight' } },
                { selector: 'edge[identical = 1]', style: { 'line-color': colour.magenta || MAGENTA, width: 3 } },
                { selector: '.faded', style: { opacity: 0.12 } },
            ],
        });

        // Lay each cluster out in its own cell, deterministically: a circle to
        // start from, then CoSE (no randomising, no animation) for larger ones.
        const width = Math.max(600, canvas.clientWidth || 900) * 1.4;
        const place = cells(list, width);
        list.forEach((c) => {
            const box = place[c.id];
            const members = cy.nodes().filter((n) => n.data('cluster') === c.id).sort((a, b) => (a.id() < b.id() ? -1 : 1));
            const cx = box.x + box.w / 2, cyy = box.y + box.h / 2, r = Math.min(box.w, box.h) * 0.32;
            members.forEach((n, k) => {
                const t = (2 * Math.PI * k) / members.length - Math.PI / 2;  // a pair stands upright, labels clear of the next cell
                n.position({ x: cx + r * Math.cos(t), y: cyy + r * Math.sin(t) });
            });
            if (members.length > 4) {
                members.union(members.edgesWith(members)).layout({
                    name: 'cose', randomize: false, animate: false, fit: false,
                    boundingBox: { x1: box.x + 8, y1: box.y + 8, w: box.w - 16, h: box.h - 16 },
                    nodeRepulsion: () => 6000, idealEdgeLength: () => 40, numIter: 600,
                }).run();
            }
        });
        cy.fit(undefined, 20);

        cy.on('tap', 'node', (e) => pick(e.target.data('cluster')));
        cy.on('tap', (e) => { if (e.target === cy) pick(null); });
        element.querySelector('.overlap-graph-tools').addEventListener('click', (event) => {
            const b = event.target.closest('[data-zoom]');
            if (!b) return;
            const centre = { x: canvas.clientWidth / 2, y: canvas.clientHeight / 2 };
            if (b.dataset.zoom === 'fit') graph.focus(card.picked);
            else cy.zoom({ level: cy.zoom() * (b.dataset.zoom === 'in' ? 1.4 : 1 / 1.4), renderedPosition: centre });
        });

        const graph = {
            focus(id) {
                cy.elements().removeClass('faded');
                if (id === null || id === undefined) { cy.fit(undefined, 20); return; }
                const members = cy.nodes().filter((n) => n.data('cluster') === id);
                if (!members.length) { cy.fit(undefined, 20); return; }
                cy.elements().difference(members.union(members.connectedEdges())).addClass('faded');
                cy.fit(members, 40);
                // A pair fits at a silly zoom; cap it around the pair's centre.
                if (cy.zoom() > 2.5) {
                    const bb = members.boundingBox();
                    cy.zoom({ level: 2.5, position: { x: (bb.x1 + bb.x2) / 2, y: (bb.y1 + bb.y2) / 2 } });
                }
            },
            resize() { cy.resize(); },
        };
        card.graph = graph;
        // The view as drawn, for "SVG/PNG (as shown)" (viz/export.js).
        element.exportSvg = () => graphSvg(cy, canvas, colour);
        // A new canvas size (phone rotation, the tab shown) refits what was in view.
        let size = `${canvas.clientWidth}x${canvas.clientHeight}`;
        const ro = root.ResizeObserver ? new ResizeObserver(() => {
            const now = `${canvas.clientWidth}x${canvas.clientHeight}`;
            if (now === size || !canvas.clientWidth) return;
            size = now;
            cy.resize();
            graph.focus(card.picked);
        }) : null;
        if (ro) ro.observe(canvas);
        views.set(element, { cy, ro });
        if (card.list.length) graph.focus(card.picked);
    }

    /** An SVG of the graph's current viewport, from Cytoscape's rendered positions. */
    function graphSvg(cy, canvas, colour) {
        const NS = 'http://www.w3.org/2000/svg';
        const w = canvas.clientWidth, h = canvas.clientHeight;
        const svg = document.createElementNS(NS, 'svg');
        svg.setAttribute('width', w);
        svg.setAttribute('height', h);
        svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
        svg.setAttribute('font-family', getComputedStyle(canvas).fontFamily || 'sans-serif');
        const el = (name, attrs, parent) => {
            const n = document.createElementNS(NS, name);
            Object.keys(attrs).forEach((k) => n.setAttribute(k, attrs[k]));
            (parent || svg).appendChild(n);
            return n;
        };
        const opacity = (e) => (e.hasClass('faded') ? 0.12 : 1);
        cy.edges().forEach((e) => {
            const s = e.source().renderedPosition(), t = e.target().renderedPosition();
            el('line', { x1: s.x, y1: s.y, x2: t.x, y2: t.y, stroke: e.data('identical') ? (colour.magenta || MAGENTA) : '#323250',
                'stroke-opacity': (e.data('identical') ? 1 : 0.35) * opacity(e), 'stroke-width': parseFloat(e.renderedStyle('width')) || 1 });
        });
        const zoom = cy.zoom();
        cy.nodes().forEach((n) => {
            const p = n.renderedPosition();
            el('circle', { cx: p.x, cy: p.y, r: n.renderedWidth() / 2, fill: n.data('colour'), stroke: '#FFFFFF',
                'stroke-width': 1, opacity: opacity(n) });
            if (9 * zoom >= 7) {
                el('text', { x: p.x, y: p.y - n.renderedHeight() / 2 - 2 * zoom, 'text-anchor': 'middle',
                    'font-size': 9 * zoom, fill: colour.primary, opacity: opacity(n) }).textContent = n.data('label');
            }
        });
        return svg;
    }

    function destroyGraph(element) {
        const view = views.get(element);
        if (!view) return;
        if (view.ro) view.ro.disconnect();
        view.cy.destroy();
        views.delete(element);
        delete element.exportSvg;
        if (card.graph) card.graph = null;
    }

    const renderer = {
        endpoint(plotName, qs) { return `/api/v2/data/${plotName}${qs}`; },
        mount: mountGraph,
        update: mountGraph,
        destroy: destroyGraph,
    };
    if (root.PlotRuntime) root.PlotRuntime.registerRenderer('cytoscape-overlap', renderer);
    root.DashOverlap.renderer = renderer;

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})(typeof window !== 'undefined' ? window : this);
