/**
 * Version selector for the Snapshot page.
 *
 * Fills the selector from /api/versions and keeps the selected version in the
 * page state (static/js/state.js; null = the latest). The runtime reloads the
 * versioned plots when it changes; this module keeps the surrounding UI in
 * step: the banner, the snapshot download links, the methodology queries and
 * the raw-data tables (#43, #49, #171, #181).
 */
(function () {
    'use strict';

    const state = window.DashState;
    if (!state) return;
    let latest = null;

    async function init() {
        const selector = document.getElementById('version-selector');
        if (!selector) return;
        let versions = [];
        try {
            const data = await (await fetch('/api/versions')).json();
            versions = data.versions || [];
        } catch (error) {
            console.error('Version list unavailable:', error);
        }
        if (!versions.length) return;
        latest = versions[0].version;

        // Pre-select with real dates so the dropdown never shows an empty
        // placeholder (#49); "Current" is the latest version.
        selector.innerHTML = '';
        versions.forEach((v, i) => {
            selector.add(new Option(i === 0 ? `Current (${v.version})` : v.version, v.version));
        });

        // A ?version= naming the latest, or an unknown one, means the latest.
        const wanted = state.get('version');
        if (wanted && (wanted === latest || !versions.some((v) => v.version === wanted))) {
            state.set({ version: null });
        }
        selector.value = state.get('version') || latest;

        selector.addEventListener('change', () => {
            state.set({ version: selector.value === latest ? null : selector.value });
        });
        state.subscribe((changed) => {
            if (!changed.includes('version')) return;
            selector.value = state.get('version') || latest;
            applyVersion();
            document.dispatchEvent(new CustomEvent('version-changed', { detail: { version: state.get('version') } }));
        });
        applyVersion();
    }

    /** Banner, download links and methodology queries for the selected version. */
    function applyVersion() {
        const version = state.get('version');
        const banner = document.getElementById('version-banner');
        if (banner) {
            banner.style.display = version ? 'block' : 'none';
            const name = banner.querySelector('.version-name');
            if (name && version) name.textContent = version;
        }
        updateDownloadLinks(version);
        updateMethodologyQueryUris(version || latest);
    }

    /** Point every snapshot download link (per-plot and bulk) at the selected version. */
    function updateDownloadLinks(version) {
        document.querySelectorAll('a[href^="/download/"]').forEach((link) => {
            const url = new URL(link.getAttribute('href'), window.location.origin);
            const isSnapshot = url.pathname.startsWith('/download/latest')
                || (url.pathname === '/download/bulk' && (url.searchParams.get('category') || '').startsWith('latest'));
            if (!isSnapshot) return;
            if (version) url.searchParams.set('version', version);
            else url.searchParams.delete('version');
            link.setAttribute('href', url.pathname + url.search);
        });
    }

    /**
     * Point the single-graph methodology queries (<…/graph/YYYY-MM-DD>) at the
     * selected version, in the text and in the "Run on Endpoint" link (#43).
     * Multi-graph trend queries have no such IRI and are left alone.
     */
    function updateMethodologyQueryUris(version) {
        if (!version) return;
        const iri = `<http://aopwiki.org/graph/${version}>`;
        const pattern = /<http:\/\/aopwiki\.org\/graph\/\d{4}-\d{2}-\d{2}>/g;
        document.querySelectorAll('details.sparql-query').forEach((panel) => {
            const code = panel.querySelector('code');
            if (!code) return;
            const updated = code.textContent.replace(pattern, iri);
            if (updated === code.textContent) return;
            code.textContent = updated;
            const link = panel.querySelector('a.sparql-run-btn');
            if (link && link.href) {
                try {
                    const url = new URL(link.href);
                    url.searchParams.set('query', updated);
                    link.href = url.toString();
                } catch (e) { /* leave the link as it was */ }
            }
        });
    }

    /** "Return to Current Version" in the banner. */
    window.resetToCurrentVersion = function () { state.set({ version: null }); };

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
