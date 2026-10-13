/**
 * Linked trend plots on /trends (#181).
 *
 * - Crosshair: hovering a snapshot on one plot shows the same snapshot on every
 *   other drawn trend plot.
 * - Shared zoom: zooming the x-axis of one plot (or resetting it) applies to all.
 * - Click a snapshot to open it on the Snapshot page (/snapshot?version=...).
 *
 * Only points whose x is a snapshot version take part, so the creation-date
 * plots (AOP lifetime) are left alone. Plots are hooked as the runtime draws
 * them (`plot:rendered`).
 */
(function () {
    'use strict';

    let versions = new Set();
    let syncing = false;  // true while this module drives other plots
    const linked = new Set();

    fetch('/api/versions')
        .then((r) => r.json())
        .then((d) => { versions = new Set((d.versions || []).map((v) => v.version)); })
        .catch(() => {});

    function snapshotOf(point) {
        const x = point && point.x;
        return typeof x === 'string' && versions.has(x.slice(0, 10)) ? x.slice(0, 10) : null;
    }

    function shown(div) {
        return div.isConnected && div.offsetParent !== null && div._fullData;
    }

    /** Every (trace, point) on `div` whose x is `snapshot`. */
    function pointsAt(div, snapshot) {
        const points = [];
        (div._fullData || []).forEach((trace, curveNumber) => {
            if (trace.visible === false || !trace.x) return;
            const xs = Array.from(trace.x, (x) => String(x).slice(0, 10));
            const pointNumber = xs.indexOf(snapshot);
            if (pointNumber !== -1) points.push({ curveNumber, pointNumber });
        });
        return points;
    }

    function others(div) {
        return Array.from(linked).filter((d) => d !== div && shown(d));
    }

    function withSync(fn) {
        syncing = true;
        try { fn(); } finally { syncing = false; }
    }

    /** True if `div` already shows the x-range `update` asks for. */
    function sameRange(div, update) {
        const axis = (div._fullLayout || {}).xaxis || {};
        if (update['xaxis.autorange']) return axis.autorange === true;
        const want = update['xaxis.range'];
        return !axis.autorange && Array.isArray(axis.range)
            && String(axis.range[0]) === String(want[0]) && String(axis.range[1]) === String(want[1]);
    }

    function link(div) {
        if (linked.has(div) || typeof div.on !== 'function') return;
        linked.add(div);

        div.on('plotly_hover', (event) => {
            if (syncing) return;
            const snapshot = snapshotOf(event.points && event.points[0]);
            if (!snapshot) return;
            withSync(() => others(div).forEach((other) => {
                const points = pointsAt(other, snapshot);
                try {
                    if (points.length) window.Plotly.Fx.hover(other, points);
                    else window.Plotly.Fx.unhover(other);
                } catch (e) { /* not drawn yet */ }
            }));
        });
        div.on('plotly_unhover', () => {
            if (syncing) return;
            withSync(() => others(div).forEach((other) => {
                try { window.Plotly.Fx.unhover(other); } catch (e) { /* gone */ }
            }));
        });
        div.on('plotly_click', (event) => {
            const snapshot = snapshotOf(event.points && event.points[0]);
            if (snapshot) window.location.assign(`/snapshot?version=${encodeURIComponent(snapshot)}`);
        });
        div.on('plotly_relayout', (change) => {
            if (syncing || !change) return;
            let update = null;
            if ('xaxis.range[0]' in change && 'xaxis.range[1]' in change) {
                update = { 'xaxis.range': [change['xaxis.range[0]'], change['xaxis.range[1]']], 'xaxis.autorange': false };
            } else if (change['xaxis.autorange'] === true) {
                update = { 'xaxis.autorange': true };
            }
            if (!update) return;
            others(div).forEach((other) => {
                const first = (other._fullData || [])[0];
                if (!first || !first.x || !snapshotOf({ x: String(first.x[0]) })) return;
                // Relayout is asynchronous, so its own plotly_relayout event
                // arrives after any flag is cleared. Skipping plots that already
                // show the range is what stops the echo bouncing between plots.
                if (sameRange(other, update)) return;
                window.Plotly.relayout(other, update).catch(() => {});
            });
        });
    }

    document.addEventListener('plot:rendered', (event) => {
        const div = event.target.querySelector('.js-plotly-plot');
        if (!div) return;
        // Forget plots the runtime has replaced.
        Array.from(linked).forEach((d) => { if (!d.isConnected) linked.delete(d); });
        link(div);
    });
})();
