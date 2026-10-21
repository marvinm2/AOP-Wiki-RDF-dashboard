/**
 * Network evolution on /network (#189): every snapshot's KE network on one layout.
 *
 * Renderer `cy-evolution` draws /api/v2/data/network_evolution with
 * Cytoscape: one node per Key Event ever seen at its fixed position (the
 * layout of the union of all snapshots, computed on the server), one edge per
 * relationship. The page's `version` (the version scrubber; null = the
 * latest) picks the snapshot: its elements are shown, those new since the
 * snapshot before are magenta, those gone since it light grey, the rest are
 * hidden. Only classes change, never positions, and Cytoscape animates
 * nothing here, so prefers-reduced-motion needs nothing extra.
 *
 * The keyboard path is the scrubber (a slider: arrows, Page Up/Down, Home,
 * End); the counts are read out in a live region, and "Show Raw Data" lists
 * the elements. Clicking a node names it under the chart.
 *
 * The pure helpers are exported for node --test.
 */
(function (root) {
    'use strict';

    const PRESENT = 'x';

    /** 'shown' | 'new' | 'gone' | 'hidden' for a presence string at snapshot i. */
    function stateAt(presence, i) {
        const here = presence[i] === PRESENT;
        const before = i > 0 && presence[i - 1] === PRESENT;
        if (here) return i > 0 && !before ? 'new' : 'shown';
        return before ? 'gone' : 'hidden';
    }

    /** An edge is in a snapshot when it and both its Key Events are. */
    function edgePresence(edge, source, target) {
        let out = '';
        for (let i = 0; i < edge.length; i++) {
            out += edge[i] === PRESENT && source[i] === PRESENT && target[i] === PRESENT ? PRESENT : '-';
        }
        return out;
    }

    /** Nodes and edges from the columnar rows; edges with an unknown Key Event are dropped. */
    function elementsOf(payload) {
        const d = payload.data;
        const n = payload.meta.rows;
        const nodes = [];
        const edges = [];
        const presence = {};
        for (let i = 0; i < n; i++) {
            if (d.Element[i] === 'KE') {
                presence[d.ID[i]] = d.Presence[i];
                nodes.push({ id: d.ID[i], label: d.Label[i], presence: d.Presence[i],
                    first: d['First Seen'][i], last: d['Last Seen'][i], x: d.X[i], y: d.Y[i] });
            }
        }
        for (let i = 0; i < n; i++) {
            if (d.Element[i] !== 'KER') continue;
            const s = presence[d.Source[i]];
            const t = presence[d.Target[i]];
            if (s === undefined || t === undefined) continue;
            edges.push({ id: d.ID[i], source: d.Source[i], target: d.Target[i],
                presence: edgePresence(d.Presence[i], s, t) });
        }
        return { nodes, edges };
    }

    /** Counts per state for nodes and edges at snapshot i. */
    function countsAt(elements, i) {
        const count = (list) => {
            const c = { shown: 0, new: 0, gone: 0, hidden: 0 };
            list.forEach((e) => { c[stateAt(e.presence, i)]++; });
            return c;
        };
        return { kes: count(elements.nodes), kers: count(elements.edges) };
    }

    const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
    const both = (kes, kers) => `${plural(kes, 'Key Event')} and ${plural(kers, 'relationship')}`;

    /** The sentence read out for snapshot i. */
    function describe(elements, versions, i) {
        const c = countsAt(elements, i);
        let text = `${versions[i]}: ${both(c.kes.shown + c.kes.new, c.kers.shown + c.kers.new)}.`;
        if (i > 0) {
            text += ` Since ${versions[i - 1]}: ${both(c.kes.new, c.kers.new)} new,`
                + ` ${both(c.kes.gone, c.kers.gone)} gone.`;
        }
        return text;
    }

    const STYLE = [
        { selector: 'node', style: { width: 5, height: 5, 'background-color': '#307BBF', 'border-width': 0, label: '' } },
        { selector: 'edge', style: { width: 0.7, 'line-color': '#A9AEC2', 'curve-style': 'haystack', opacity: 0.7 } },
        { selector: 'node.new', style: { width: 9, height: 9, 'background-color': '#E6007E', 'z-index': 10 } },
        { selector: 'edge.new', style: { width: 1.4, 'line-color': '#E6007E', opacity: 1, 'z-index': 10 } },
        { selector: 'node.gone', style: { 'background-color': '#D3D3DA', opacity: 0.7 } },
        { selector: 'edge.gone', style: { 'line-color': '#E2E2E8', opacity: 0.7 } },
        { selector: '.hidden', style: { display: 'none' } },
        { selector: 'node.picked', style: { width: 12, height: 12, 'border-width': 2, 'border-color': '#29235C' } },
    ];

    const views = new WeakMap();   // container -> {cy, elements, versions}
    let versionsPromise = null;

    function versionList() {
        if (!versionsPromise) {
            versionsPromise = fetch('/api/versions').then((r) => r.json())
                .then((d) => (d.versions || []).map((v) => v.version).sort())
                .catch((e) => { versionsPromise = null; throw e; });
        }
        return versionsPromise;
    }

    function card(element) { return element.closest('.plot-box'); }

    function indexFor(view) {
        const wanted = root.DashState && root.DashState.get('version');
        const i = wanted ? view.versions.indexOf(wanted) : -1;
        return i >= 0 ? i : view.versions.length - 1;
    }

    function apply(element) {
        const view = views.get(element);
        if (!view) return;
        const i = indexFor(view);
        view.cy.batch(() => {
            view.cy.elements().forEach((ele) => {
                const s = stateAt(ele.data('presence'), i);
                ele.classes(s === 'shown' ? [] : [s].concat(ele.hasClass('picked') ? ['picked'] : []));
            });
        });
        const summary = card(element).querySelector('.evolution-summary');
        if (summary) summary.textContent = describe(view.elements, view.versions, i);
    }

    async function mount(element, payload) {
        if (!root.cytoscape) {
            element.innerHTML = '<p class="plot-error">The network needs a script that did not load. '
                + 'The CSV and "Show Raw Data" hold the same elements.</p>';
            return;
        }
        const versions = await versionList();
        const elements = elementsOf(payload);
        const width = elements.nodes.length ? elements.nodes[0].presence.length : 0;
        if (width !== versions.length) {
            element.innerHTML = `<p class="plot-error">The network covers ${width} snapshots but the endpoint lists `
                + `${versions.length}; reload the page.</p>`;
            return;
        }
        const old = views.get(element);
        if (old) old.cy.destroy();
        element.innerHTML = '';
        const box = document.createElement('div');
        box.className = 'evolution-canvas';
        element.appendChild(box);
        const cy = root.cytoscape({
            container: box,
            elements: elements.nodes.map((n) => ({ data: n, position: { x: n.x, y: -n.y } }))
                .concat(elements.edges.map((e) => ({ data: e }))),
            layout: { name: 'preset' },
            style: STYLE,
            minZoom: 0.05,
            maxZoom: 6,
            textureOnViewport: true,
            hideEdgesOnViewport: true,
            autoungrabify: true,
        });
        views.set(element, { cy, elements, versions });
        element.setAttribute('role', 'img');
        element.setAttribute('aria-label', 'The Key Event network of the selected snapshot on a fixed layout. '
            + 'Pick the snapshot with the timeline above; the counts are read out below it, '
            + 'and "Show Raw Data" lists every element.');
        cy.on('tap', 'node', (event) => {
            const n = event.target;
            cy.nodes('.picked').removeClass('picked');
            n.addClass('picked');
            const picked = card(element).querySelector('.evolution-picked');
            const seen = n.data('presence').split('').filter((c) => c === PRESENT).length;
            if (picked) {
                picked.textContent = `${n.id()}: ${n.data('label') || 'no title'}. In ${seen} of ${versions.length} `
                    + `snapshots, first ${n.data('first')}, last ${n.data('last')}.`;
            }
        });
        apply(element);
        cy.fit(undefined, 20);
    }

    function destroy(element) {
        const view = views.get(element);
        if (view) view.cy.destroy();
        views.delete(element);
    }

    const renderer = {
        endpoint(plotName) { return `/api/v2/data/${plotName}`; },
        shared: true,
        mount,
        update: mount,
        destroy,
    };

    function init() {
        if (root.DashState) {
            root.DashState.subscribe((changed) => {
                if (!changed.includes('version')) return;
                document.querySelectorAll('[data-renderer="cy-evolution"]').forEach(apply);
            });
        }
        // Cytoscape measures its box when drawn; a tab that was hidden needs a new measure.
        document.querySelectorAll('.tab-btn[data-tab="evolution-view"]').forEach((btn) => {
            btn.addEventListener('click', () => setTimeout(() => {
                document.querySelectorAll('[data-renderer="cy-evolution"]').forEach((el) => {
                    const view = views.get(el);
                    if (view) { view.cy.resize(); view.cy.fit(undefined, 20); }
                });
            }, 60));
        });
    }

    const api = { stateAt, edgePresence, elementsOf, countsAt, describe };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    if (root.PlotRuntime) root.PlotRuntime.registerRenderer('cy-evolution', renderer);
    if (typeof document !== 'undefined') {
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
        else init();
    }
})(typeof window !== 'undefined' ? window : globalThis);
