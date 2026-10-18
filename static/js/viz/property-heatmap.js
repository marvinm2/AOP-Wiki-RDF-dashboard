/**
 * Property x snapshot heatmap on /trends (#187).
 *
 * The four heatmaps (AOPs, KEs, KERs, Stressors) are server Plotly figures,
 * one per tab; the tab lives in the page state as `props` (AOPs = default).
 * This module adds what a 2-D figure needs beyond viz/a11y.js, which only
 * steps along one axis: arrow keys move a cell at a time (columns limited to
 * the snapshots the /trends range shows), Home/End jump along the row, Enter
 * opens the focused snapshot, Escape clears. Each step is read out in the
 * live region. The download links follow the tab.
 *
 * The pure helpers are exported for node --test.
 */
(function (root) {
    'use strict';

    const ENTITIES = ['aop', 'ke', 'ker', 'stressor'];
    const NOUNS = { aop: 'AOPs', ke: 'KEs', ker: 'KERs', stressor: 'Stressors' };

    /** The entity a `props` state value stands for (unknown or empty = AOPs). */
    function entityOf(value) {
        return ENTITIES.includes(value) ? value : 'aop';
    }

    /** The state value for an entity: the default is left out of the URL. */
    function stateValue(entity) {
        return entity === 'aop' ? null : entity;
    }

    /** Next tab index for a key on the tab list, or undefined for other keys. */
    function tabStep(key, i, n) {
        if (key === 'ArrowRight') return (i + 1) % n;
        if (key === 'ArrowLeft') return (i - 1 + n) % n;
        if (key === 'Home') return 0;
        if (key === 'End') return n - 1;
        return undefined;
    }

    /**
     * The cell a key moves to: [row, col] within nRows x the visible columns
     * [lo, hi]. The first key lands on the top-left visible cell (Left and
     * End on the newest column). Other keys give undefined.
     */
    function cellStep(key, cell, nRows, lo, hi) {
        if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(key)) return undefined;
        if (!nRows || hi < lo) return undefined;
        if (!cell) return [0, key === 'ArrowLeft' || key === 'End' ? hi : lo];
        let [r, c] = cell;
        c = Math.min(hi, Math.max(lo, c));
        if (key === 'ArrowUp') r = Math.max(0, r - 1);
        if (key === 'ArrowDown') r = Math.min(nRows - 1, r + 1);
        if (key === 'ArrowLeft') c = Math.max(lo, c - 1);
        if (key === 'ArrowRight') c = Math.min(hi, c + 1);
        if (key === 'Home') c = lo;
        if (key === 'End') c = hi;
        return [r, c];
    }

    /**
     * [lo, hi] column indices inside an x range (date axis, as the /trends
     * range sets it); all columns when the axis is autoranged.
     */
    function visibleColumns(xs, axis) {
        const all = [0, xs.length - 1];
        if (!axis || axis.autorange !== false || !Array.isArray(axis.range)) return all;
        const time = (v) => (typeof v === 'number' ? v : Date.parse(String(v).replace(' ', 'T') + (/[zZ]$|[+-]\d\d:?\d\d$/.test(String(v)) ? '' : 'Z')));
        const a = time(axis.range[0]);
        const b = time(axis.range[1]);
        if (isNaN(a) || isNaN(b)) return all;
        const lo = Math.min(a, b);
        const hi = Math.max(a, b);
        const inside = xs.map((x, i) => [i, time(x)]).filter(([, t]) => t >= lo && t <= hi).map(([i]) => i);
        return inside.length ? [inside[0], inside[inside.length - 1]] : all;
    }

    /** The sentence read out for one cell. */
    function describeCell(trace, r, c, entity) {
        const label = trace.y[r];
        const version = trace.x[c];
        const z = trace.z[r][c];
        const extra = (trace.customdata && trace.customdata[r] && trace.customdata[r][c]) || [];
        const count = trace.text && trace.text[r] ? trace.text[r][c] : '';
        const pct = typeof z === 'number' ? `${Math.round(z * 10) / 10}%` : 'no value';
        const first = extra[1] && extra[1] !== 'never' ? `first filled in ${extra[1]}` : 'never filled in';
        return `${label}${extra[0] ? ` (${extra[0]})` : ''}, ${version}: ${pct} of ${NOUNS[entity]}`
            + `${count ? ` (${count})` : ''}, ${first}.`;
    }

    /** A download href with its plot name switched to `entity`, query kept. */
    function downloadHref(href, entity) {
        return href.replace(/\/download\/trend\/[a-z]+_property_heatmap/, `/download/trend/${entity}_property_heatmap`);
    }

    const api = { ENTITIES, entityOf, stateValue, tabStep, cellStep, visibleColumns, describeCell, downloadHref };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
        return;
    }
    root.DashPropertyHeatmap = api;

    // ------------------------------------------------------------- the page

    let view = null;
    const cells = new WeakMap();  // plot container -> [row, col]

    function q(sel) { return view.querySelector(sel); }
    function tabs() { return Array.from(view.querySelectorAll('.props-tabs [role="tab"]')); }
    function announce(text) { q('.props-live').textContent = text; }

    function selectTab(entity, focus) {
        tabs().forEach((tab) => {
            const on = tab.dataset.tab === entity;
            tab.setAttribute('aria-selected', on ? 'true' : 'false');
            tab.tabIndex = on ? 0 : -1;
            tab.classList.toggle('is-active', on);
            document.getElementById(tab.getAttribute('aria-controls')).hidden = !on;
            if (on && focus) tab.focus();
        });
        document.querySelectorAll('.props-downloads a').forEach((a) => {
            a.setAttribute('href', downloadHref(a.getAttribute('href'), entity));
        });
        // A panel loads once it is shown; one drawn while hidden needs its size back.
        if (root.plotLoader) root.plotLoader.loadVisible();
        const div = q(`#props-panel-${entity} .js-plotly-plot`);
        if (div && root.Plotly) Promise.resolve(root.Plotly.Plots.resize(div)).catch(() => {});
    }

    function wireTabs() {
        const state = root.DashState;
        const list = tabs();
        list.forEach((tab, i) => {
            tab.addEventListener('click', () => state.set({ props: stateValue(tab.dataset.tab) }));
            tab.addEventListener('keydown', (event) => {
                const next = tabStep(event.key, i, list.length);
                if (next === undefined) return;
                event.preventDefault();
                state.set({ props: stateValue(list[next].dataset.tab) });
                list[next].focus();
            });
        });
        const apply = () => selectTab(entityOf(state.get('props')));
        state.subscribe((changed) => { if (changed.includes('props')) apply(); });
        apply();
    }

    function onKey(event) {
        const container = event.currentTarget;
        const div = container.querySelector('.js-plotly-plot');
        const trace = div && div._fullData && div._fullData[0];
        if (!trace || !root.Plotly) return;
        const entity = container.closest('.props-panel').dataset.entity;
        const cell = cells.get(container);
        if (event.key === 'Escape') {
            if (!cell) return;
            cells.delete(container);
            root.Plotly.Fx.unhover(div);
            announce('Cell cleared.');
            return;
        }
        if (event.key === 'Enter') {
            if (!cell) return;
            event.preventDefault();
            window.location.assign(`/snapshot?version=${encodeURIComponent(trace.x[cell[1]])}`);
            return;
        }
        const [lo, hi] = visibleColumns(Array.from(trace.x), div._fullLayout && div._fullLayout.xaxis);
        const next = cellStep(event.key, cell, trace.y.length, lo, hi);
        if (!next) return;
        event.preventDefault();
        cells.set(container, next);
        try {
            root.Plotly.Fx.hover(div, [{ curveNumber: 0, pointNumber: next }]);
        } catch (e) { /* not drawn yet */ }
        announce(describeCell(trace, next[0], next[1], entity));
    }

    function init() {
        view = document.querySelector('.props-view');
        if (!view || !root.DashState) return;
        wireTabs();
        document.addEventListener('plot:rendered', (event) => {
            const container = event.target;
            if (!container.closest || !container.closest('.props-view')) return;
            const entity = container.closest('.props-panel').dataset.entity;
            container.setAttribute('role', 'figure');
            container.tabIndex = 0;
            container.setAttribute('aria-label', `Heatmap of how many ${NOUNS[entity]} fill in each property, `
                + 'one row per property and one column per snapshot. Use the arrow keys to read the cells, '
                + 'Enter to open the snapshot, or "Show Raw Data" below for the table.');
            cells.delete(container);
            if (!container.dataset.propsKeys) {
                container.dataset.propsKeys = '1';
                container.addEventListener('keydown', onKey);
            }
        });
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})(typeof window !== 'undefined' ? window : globalThis);
