/**
 * Date-range selector for the /trends page (issue #44).
 *
 * - Populates two <select> dropdowns from /api/versions.
 * - On change, clamps the x-axis of every snapshot-keyed trend plot to the
 *   chosen sub-range using Plotly.relayout. Trend plots use a categorical
 *   x-axis (string version dates), so the range is expressed in index space
 *   (start_idx - 0.5, end_idx + 0.5).
 * - Mirrors the selection to ?start=...&end=... in the URL.
 * - Appends start/end query params to every download link on the page
 *   (CSV/PNG/SVG/bulk) so server-side exports honour the same constraint.
 * - Skips plots that are not snapshot-iterated (lifetime/creation-date plots).
 */

(function () {
    'use strict';

    // Trend plot containers that don't iterate over snapshot versions —
    // their x-axis is creation/modification year, not a snapshot date — so
    // a "version range" doesn't apply.
    const EXCLUDED_PLOTS = new Set([
        'aops_created_over_time',
        'aops_modified_over_time',
        'aop_creation_vs_modification_timeline'
    ]);

    let allVersions = []; // [{version, graph_uri, date}, ...] newest first
    let startVersion = null;
    let endVersion = null;

    async function init() {
        const container = document.getElementById('trends-range-selector');
        if (!container) return; // not on /trends

        try {
            const resp = await fetch('/api/versions');
            const data = await resp.json();
            if (!data.versions || data.versions.length === 0) {
                container.style.display = 'none';
                return;
            }
            allVersions = data.versions
                .slice()
                .sort((a, b) => (a.version < b.version ? -1 : 1)); // ascending
        } catch (e) {
            console.error('trends-range-selector: /api/versions failed', e);
            container.style.display = 'none';
            return;
        }

        populateSelects();
        restoreFromURL();
        wireEvents();
        applyRange({ skipPlotRelayout: false });
    }

    function populateSelects() {
        const startSel = document.getElementById('trends-range-start');
        const endSel = document.getElementById('trends-range-end');
        startSel.innerHTML = '';
        endSel.innerHTML = '';

        allVersions.forEach((v) => {
            const o1 = new Option(v.version, v.version);
            const o2 = new Option(v.version, v.version);
            startSel.add(o1);
            endSel.add(o2);
        });

        // Default: full range
        startSel.value = allVersions[0].version;
        endSel.value = allVersions[allVersions.length - 1].version;
    }

    function restoreFromURL() {
        const params = new URLSearchParams(window.location.search);
        const s = params.get('start');
        const e = params.get('end');
        const startSel = document.getElementById('trends-range-start');
        const endSel = document.getElementById('trends-range-end');
        const known = new Set(allVersions.map((v) => v.version));
        if (s && known.has(s)) startSel.value = s;
        if (e && known.has(e)) endSel.value = e;
    }

    function wireEvents() {
        const startSel = document.getElementById('trends-range-start');
        const endSel = document.getElementById('trends-range-end');
        const reset = document.getElementById('trends-range-reset');

        startSel.addEventListener('change', () => applyRange({ skipPlotRelayout: false }));
        endSel.addEventListener('change', () => applyRange({ skipPlotRelayout: false }));
        reset.addEventListener('click', () => {
            startSel.value = allVersions[0].version;
            endSel.value = allVersions[allVersions.length - 1].version;
            applyRange({ skipPlotRelayout: false });
        });

        // When the lazy-loader replaces a placeholder with a real plot, the
        // current range needs to be re-applied to the new plotly div.
        // Hook into a MutationObserver on each snapshot-keyed container.
        document.querySelectorAll('.lazy-plot').forEach((el) => {
            const name = el.dataset.plotName;
            if (!name || EXCLUDED_PLOTS.has(stripSuffix(name))) return;
            const obs = new MutationObserver(() => {
                const plotly = el.querySelector('.plotly-graph-div');
                if (plotly) relayoutPlot(plotly);
            });
            obs.observe(el, { childList: true, subtree: true });
        });
    }

    function stripSuffix(name) {
        // Some plot names have _absolute / _delta / _percentage suffixes; the
        // exclusion list uses the bare name.
        return name.replace(/_(absolute|delta|percentage)$/, '');
    }

    function applyRange({ skipPlotRelayout }) {
        startVersion = document.getElementById('trends-range-start').value;
        endVersion = document.getElementById('trends-range-end').value;

        // Enforce start <= end
        if (startVersion > endVersion) {
            const tmp = startVersion;
            startVersion = endVersion;
            endVersion = tmp;
            document.getElementById('trends-range-start').value = startVersion;
            document.getElementById('trends-range-end').value = endVersion;
        }

        updateStatus();
        updateURLState();
        updateDownloadLinks();
        if (!skipPlotRelayout) relayoutAllPlots();
    }

    function isFullRange() {
        return (
            allVersions.length > 0 &&
            startVersion === allVersions[0].version &&
            endVersion === allVersions[allVersions.length - 1].version
        );
    }

    function updateStatus() {
        const status = document.getElementById('trends-range-status');
        if (!status) return;
        if (isFullRange()) {
            status.textContent =
                'Showing all snapshots. Lifetime/creation-date plots are not affected.';
        } else {
            status.textContent = `Constrained to ${startVersion} → ${endVersion}. Lifetime/creation-date plots are not affected.`;
        }
    }

    function updateURLState() {
        const params = new URLSearchParams(window.location.search);
        if (isFullRange()) {
            params.delete('start');
            params.delete('end');
        } else {
            params.set('start', startVersion);
            params.set('end', endVersion);
        }
        const qs = params.toString();
        const newUrl = window.location.pathname + (qs ? '?' + qs : '');
        history.replaceState(null, '', newUrl);
    }

    function updateDownloadLinks() {
        // Walk every <a href="/download/..."> on the page and add/remove the
        // start/end query params so server-side CSV/PNG/SVG exports — and the
        // bulk-download ZIPs — honour the selected window.
        document.querySelectorAll('a[href^="/download/"]').forEach((a) => {
            const url = new URL(a.getAttribute('href'), window.location.origin);
            // Bulk download links carry a `category=trends-*` query param.
            // For per-plot links, the URL path encodes the plot name. We
            // intentionally apply the range to *all* trends download links
            // (not lifetime plots, which don't appear under /download/trend/).
            if (isFullRange()) {
                url.searchParams.delete('start');
                url.searchParams.delete('end');
            } else {
                url.searchParams.set('start', startVersion);
                url.searchParams.set('end', endVersion);
            }
            // Preserve any path-only links (no host)
            const newHref = url.pathname + (url.search || '');
            a.setAttribute('href', newHref);
        });
    }

    function relayoutAllPlots() {
        document.querySelectorAll('.lazy-plot').forEach((el) => {
            const name = el.dataset.plotName;
            if (!name || EXCLUDED_PLOTS.has(stripSuffix(name))) return;
            const plotly = el.querySelector('.plotly-graph-div');
            if (plotly) relayoutPlot(plotly);
        });
    }

    function relayoutPlot(plotDiv) {
        if (!window.Plotly) return;
        if (isFullRange()) {
            window.Plotly.relayout(plotDiv, { 'xaxis.autorange': true });
            return;
        }
        const startIdx = allVersions.findIndex((v) => v.version === startVersion);
        const endIdx = allVersions.findIndex((v) => v.version === endVersion);
        if (startIdx === -1 || endIdx === -1) return;
        // Plotly indexes categorical x-axes by ordinal position. ±0.5 gives a
        // half-step pad so the first/last markers aren't clipped.
        window.Plotly.relayout(plotDiv, {
            'xaxis.range': [startIdx - 0.5, endIdx + 0.5],
            'xaxis.autorange': false
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
