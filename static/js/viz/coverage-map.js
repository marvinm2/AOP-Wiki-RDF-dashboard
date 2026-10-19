/**
 * Ontology coverage map on the Snapshot page (#187).
 *
 * The map is a server Plotly figure: an icicle drawn as horizontal bars, one
 * row per level of the organ-system branch, one block per term. This module
 * adds what the figure does not do itself: a click zooms the x axis to the
 * block (double-click zooms out, as on any Plotly chart), and the keyboard
 * walks the tree. Down goes to the first sub-branch, Up to the parent,
 * Left/Right along the level (only blocks inside the zoom), Home/End to the
 * ends of the level, Enter zooms to the focused block, Escape zooms out. Each
 * step is read out in the live region. A zoom is a plain relayout with no
 * transition, so prefers-reduced-motion needs nothing extra.
 *
 * The pure helpers are exported for node --test.
 */
(function (root) {
    'use strict';

    const PLOT = 'latest_ontology_coverage_map';
    const FULL = [0, 100];
    const DOUBLE_CLICK_MS = 300;

    /**
     * One node per block from the figure's traces. The customdata layout is
     * set in plots/latest_plots.py (_render_ontology_coverage_map):
     * [term, id, status, subtree size, parent label, also-under note,
     * parent id, level].
     */
    function buildIndex(traces) {
        const nodes = [];
        traces.forEach((trace, curve) => {
            (trace.customdata || []).forEach((cd, point) => {
                const start = Number(trace.base[point]);
                nodes.push({
                    curve, point, label: cd[0], id: cd[1], status: cd[2], terms: Number(cd[3]),
                    parentId: cd[6] || null, level: Number(cd[7]),
                    start, end: start + Number(trace.x[point]),
                });
            });
        });
        nodes.sort((a, b) => a.level - b.level || a.start - b.start);
        const top = nodes.find((n) => n.level === 0);
        const byId = new Map(nodes.filter((n) => n.level > 0).map((n) => [n.id, n]));
        nodes.forEach((n, i) => { n.index = i; n.children = []; });
        nodes.forEach((n) => {
            if (n.level === 0) { n.parent = null; return; }
            n.parent = n.parentId ? byId.get(n.parentId) || top : top;
            if (n.parent) n.parent.children.push(n);
        });
        return nodes;
    }

    /** The blocks on one level that show inside the x range [lo, hi]. */
    function onLevel(nodes, level, range) {
        const [lo, hi] = range || FULL;
        return nodes.filter((n) => n.level === level && n.end > lo && n.start < hi);
    }

    /** The node a key moves to from `current` (undefined: not a move key). */
    function step(key, nodes, current, range) {
        if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(key)) return undefined;
        if (!nodes.length) return undefined;
        if (!current) return nodes.find((n) => n.level === 0) || nodes[0];
        if (key === 'ArrowUp') return current.parent || current;
        if (key === 'ArrowDown') {
            const [lo, hi] = range || FULL;
            return current.children.find((c) => c.end > lo && c.start < hi) || current;
        }
        const row = onLevel(nodes, current.level, range);
        if (!row.length) return current;
        const i = row.indexOf(current);
        if (key === 'Home') return row[0];
        if (key === 'End') return row[row.length - 1];
        if (i < 0) return row[0];
        if (key === 'ArrowLeft') return row[Math.max(0, i - 1)];
        return row[Math.min(row.length - 1, i + 1)];
    }

    const PHRASES = {
        'Annotated': 'annotated',
        'Contains annotated terms': 'not annotated itself, but terms below it are',
        'Coverage hole': 'a coverage hole: nothing in it is annotated',
        'Inside a hole': 'inside a coverage hole',
    };

    /** The sentence read out for one block. */
    function describe(node) {
        const where = node.level === 0 ? 'the whole branch' : `level ${node.level}`;
        const kids = node.children.length;
        const below = kids ? ` ${kids} sub-branch${kids === 1 ? '' : 'es'} below.` : '';
        return `${node.label} (${node.id}), ${where}, ${PHRASES[node.status] || node.status}. `
            + `${node.terms} term${node.terms === 1 ? '' : 's'} in its subtree.${below}`;
    }

    /** The x range that shows a block across the full width. */
    function zoomRange(node) {
        return node.level === 0 ? FULL.slice() : [node.start, node.end];
    }

    const api = { PLOT, buildIndex, onLevel, step, describe, zoomRange };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
        return;
    }
    root.DashCoverageMap = api;

    // ------------------------------------------------------------- the page

    const focus = new WeakMap();  // plot container -> focused node

    function announce(container, text) {
        const live = container.closest('.plot-box').querySelector('.coverage-map-live');
        if (live) live.textContent = text;
    }

    function currentRange(div) {
        const axis = div._fullLayout && div._fullLayout.xaxis;
        return axis && Array.isArray(axis.range) ? axis.range.map(Number) : FULL.slice();
    }

    function zoomTo(div, range) {
        return Promise.resolve(root.Plotly.relayout(div, { 'xaxis.range': range, 'xaxis.autorange': false }))
            .catch(() => {});
    }

    function wire(container) {
        const div = container.querySelector('.js-plotly-plot');
        if (!div || !root.Plotly) return;
        const nodes = buildIndex(div.data || []);
        focus.delete(container);
        container.setAttribute('role', 'figure');
        container.tabIndex = 0;
        container.setAttribute('aria-label', 'Icicle of the organ-system branch: the branch on top, its '
            + 'sub-branches below, each block as wide as its subtree and coloured by annotation. Use the '
            + 'arrow keys to move between blocks, Enter to zoom in, Escape to zoom out, or "Show Raw Data" '
            + 'below for the table.');

        // A double-click also fires two clicks: hold each click back long
        // enough for a double-click to cancel it, or it would zoom right back in.
        let pending = null;
        div.on('plotly_click', (event) => {
            const p = event && event.points && event.points[0];
            if (!p) return;
            const node = nodes.find((n) => n.curve === p.curveNumber && n.point === p.pointNumber);
            if (!node) return;
            clearTimeout(pending);
            pending = setTimeout(() => {
                focus.set(container, node);
                zoomTo(div, zoomRange(node));
                announce(container, `Zoomed to ${node.label}. ${describe(node)}`);
            }, DOUBLE_CLICK_MS);
        });
        div.on('plotly_doubleclick', () => {
            clearTimeout(pending);
            zoomTo(div, FULL.slice());
            announce(container, 'Zoomed out to the whole branch.');
            return false;
        });

        container.onkeydown = (event) => {
            const current = focus.get(container);
            if (event.key === 'Escape') {
                event.preventDefault();
                root.Plotly.Fx.unhover(div);
                zoomTo(div, FULL.slice());
                announce(container, 'Zoomed out to the whole branch.');
                return;
            }
            if (event.key === 'Enter') {
                if (!current) return;
                event.preventDefault();
                zoomTo(div, zoomRange(current));
                announce(container, `Zoomed to ${current.label}.`);
                return;
            }
            const next = step(event.key, nodes, current, currentRange(div));
            if (!next) return;
            event.preventDefault();
            focus.set(container, next);
            try {
                root.Plotly.Fx.hover(div, [{ curveNumber: next.curve, pointNumber: next.point }]);
            } catch (e) { /* not drawn yet */ }
            announce(container, describe(next));
        };
    }

    document.addEventListener('plot:rendered', (event) => {
        if (!event.detail || event.detail.plotName !== PLOT) return;
        wire(event.target);
    });
})(typeof window !== 'undefined' ? window : globalThis);
