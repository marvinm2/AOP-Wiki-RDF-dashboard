/**
 * Organ-system bump chart on /trends (#187).
 *
 * The chart is a server Plotly figure: one line per organ system through its
 * rank by AOP count at each snapshot. This module adds two things. The
 * "Highlight" select picks one system and dims the other lines; the pick
 * lives in the page state as `organ`, so a highlighted view can be shared.
 * On the focused chart, Left/Right step through the snapshots the /trends
 * range shows and Up/Down move one rank up or down at that snapshot, so the
 * keyboard follows one system across time; Home/End jump to the ends, Enter
 * opens the snapshot, Escape clears. Each step is read out in the live
 * region. Dimming is a restyle with no transition.
 *
 * The pure helpers are exported for node --test.
 */
(function (root) {
    'use strict';

    const PLOT = 'organ_coverage_bump';
    const DIM = 0.15;

    /** [{curve, name, ranks}] of the traces; ranks[col] is null where unranked. */
    function linesOf(traces) {
        return traces.map((t, curve) => ({
            curve, name: t.name, ranks: Array.from(t.y || [], (y) => (y == null ? null : Number(y))),
        }));
    }

    /** The lines ranked at column `col`, best rank first. */
    function rankedAt(lines, col) {
        return lines.filter((l) => l.ranks[col] != null).sort((a, b) => a.ranks[col] - b.ranks[col]);
    }

    /**
     * The {curve, col} a key moves to within the visible columns [lo, hi].
     * Left/Right keep the system and move to the nearest snapshot where it is
     * ranked; Up/Down keep the snapshot and move one rank. The first key lands
     * on rank 1 at the newest visible snapshot. Other keys give undefined.
     */
    function bumpStep(key, focus, lines, lo, hi) {
        if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(key)) return undefined;
        if (!lines.length || hi < lo) return undefined;
        const top = (col) => { const r = rankedAt(lines, col); return r.length ? { curve: r[0].curve, col } : undefined; };
        if (!focus) {
            for (let col = hi; col >= lo; col--) { const t = top(col); if (t) return t; }
            return undefined;
        }
        const line = lines[focus.curve];
        const col = Math.min(hi, Math.max(lo, focus.col));
        const rankedCols = [];
        for (let c = lo; c <= hi; c++) if (line.ranks[c] != null) rankedCols.push(c);
        if (key === 'ArrowUp' || key === 'ArrowDown') {
            const row = rankedAt(lines, col);
            const i = row.findIndex((l) => l.curve === focus.curve);
            if (i < 0) return top(col) || focus;
            const j = key === 'ArrowUp' ? Math.max(0, i - 1) : Math.min(row.length - 1, i + 1);
            return { curve: row[j].curve, col };
        }
        if (!rankedCols.length) return focus;
        if (key === 'Home') return { curve: focus.curve, col: rankedCols[0] };
        if (key === 'End') return { curve: focus.curve, col: rankedCols[rankedCols.length - 1] };
        const next = key === 'ArrowLeft'
            ? rankedCols.filter((c) => c < col).pop()
            : rankedCols.find((c) => c > col);
        return { curve: focus.curve, col: next === undefined ? col : next };
    }

    /** The sentence read out for one point. */
    function describePoint(trace, col) {
        const rank = trace.y[col];
        const cd = (trace.customdata && trace.customdata[col]) || [];
        if (rank == null) return `${trace.name}, ${trace.x[col]}: not ranked, no AOPs.`;
        return `${trace.name}, ${trace.x[col]}: rank ${rank} of ${cd[2]}, ${cd[0]} AOPs, ${cd[1]}.`;
    }

    /** Opacity per trace: all 1 with no pick (or an unknown one), else the pick alone. */
    function opacities(names, picked) {
        if (!picked || !names.includes(picked)) return names.map(() => 1);
        return names.map((n) => (n === picked ? 1 : DIM));
    }

    const api = { PLOT, linesOf, rankedAt, bumpStep, describePoint, opacities };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
        return;
    }
    root.DashBump = api;

    // ------------------------------------------------------------- the page

    const focus = new WeakMap();  // plot container -> {curve, col}
    let card = null;

    function announce(text) {
        const live = card.querySelector('.bump-live');
        if (live) live.textContent = text;
    }

    function plotDiv() {
        return card.querySelector('.lazy-plot .js-plotly-plot');
    }

    /** Dim every line but the picked one; fill the select from the drawn traces. */
    function applyPick() {
        const picked = root.DashState.get('organ');
        const select = card.querySelector('#organ-bump-highlight');
        const div = plotDiv();
        if (!div || !root.Plotly) {
            if (select) select.value = picked || '';
            return;
        }
        const names = (div.data || []).map((t) => t.name);
        if (select && select.options.length <= 1) {
            names.forEach((n) => select.add(new Option(n, n)));
        }
        if (select) select.value = names.includes(picked) ? picked : '';
        const op = opacities(names, picked);
        Promise.resolve(root.Plotly.restyle(div, { opacity: op })).catch(() => {});
    }

    function visible(div, xs) {
        const heat = root.DashPropertyHeatmap;  // shares the /trends range reading
        return heat ? heat.visibleColumns(xs, div._fullLayout && div._fullLayout.xaxis) : [0, xs.length - 1];
    }

    function onKey(event) {
        const container = event.currentTarget;
        const div = container.querySelector('.js-plotly-plot');
        if (!div || !div.data || !root.Plotly) return;
        const current = focus.get(container);
        if (event.key === 'Escape') {
            if (!current) return;
            focus.delete(container);
            root.Plotly.Fx.unhover(div);
            announce('Point cleared.');
            return;
        }
        if (event.key === 'Enter') {
            if (!current) return;
            event.preventDefault();
            window.location.assign(`/snapshot?version=${encodeURIComponent(div.data[current.curve].x[current.col])}`);
            return;
        }
        const xs = Array.from(div.data[0].x || []);
        const [lo, hi] = visible(div, xs);
        const next = bumpStep(event.key, current, linesOf(div.data), lo, hi);
        if (!next) return;
        event.preventDefault();
        focus.set(container, next);
        try {
            root.Plotly.Fx.hover(div, [{ curveNumber: next.curve, pointNumber: next.col }]);
        } catch (e) { /* not drawn yet */ }
        announce(describePoint(div.data[next.curve], next.col));
    }

    function init() {
        card = document.getElementById('organ-bump');
        if (!card || !root.DashState) return;
        const state = root.DashState;
        const select = card.querySelector('#organ-bump-highlight');
        if (select) select.addEventListener('change', () => state.set({ organ: select.value || null }));
        state.subscribe((changed) => { if (changed.includes('organ')) applyPick(); });
        document.addEventListener('plot:rendered', (event) => {
            if (!event.detail || event.detail.plotName !== PLOT) return;
            const container = event.target;
            container.setAttribute('role', 'figure');
            container.tabIndex = 0;
            container.setAttribute('aria-label', 'Bump chart of the organ systems ranked by number of AOPs, one line '
                + 'per system across the snapshots. Use left and right to move between snapshots, up and down to '
                + 'move between ranks, Enter to open the snapshot, or "Show Raw Data" below for the table.');
            focus.delete(container);
            if (!container.dataset.bumpKeys) {
                container.dataset.bumpKeys = '1';
                container.addEventListener('keydown', onKey);
            }
            applyPick();
        });
        applyPick();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})(typeof window !== 'undefined' ? window : globalThis);
