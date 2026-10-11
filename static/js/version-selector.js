/**
 * Version Selector for AOP-Wiki RDF Dashboard
 *
 * Enables users to select and view Latest Data plots for any historical database version.
 * Dynamically regenerates plots when version changes.
 */

(function() {
    'use strict';

    // Verbose init tracing, off by default so a normal load is quiet (#131).
    // Enable with localStorage.setItem('plotDebug','1') then reload.
    const DEBUG = (() => {
        try { return localStorage.getItem('plotDebug') === '1'; } catch (e) { return false; }
    })();
    const dbg = (...args) => { if (DEBUG) console.log(...args); };

    let selectedVersion = null;  // null = latest version
    let availableVersions = [];

    // Which plots take a version comes from the server (window.VERSIONED_PLOTS,
    // rendered from LATEST_PLOT_FUNCTIONS), not from a list kept here (#171).

    // Plots driven by the domain-lens selector — one control, three charts.
    const domainLensPlots = [
        'latest_domain_coverage',
        'latest_domain_completeness',
        'latest_domain_status'
    ];

    /**
     * Update the URL with the current version selection (without page reload)
     */
    function updateURLState(version) {
        var params = new URLSearchParams(window.location.search);
        var latest = availableVersions[0] && availableVersions[0].version;
        // Treat "no version" and "latest version" the same in the URL so
        // bookmarking /snapshot stays clean (#49).
        if (version && version !== '' && version !== latest) {
            params.set('version', version);
        } else {
            params.delete('version');
        }
        var newUrl = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
        history.replaceState(null, '', newUrl);
    }

    /**
     * Restore version selection from URL ?version= parameter
     */
    function restoreVersionFromURL() {
        var params = new URLSearchParams(window.location.search);
        var version = params.get('version');
        if (version) {
            var selector = document.getElementById('version-selector');
            if (selector) {
                var option = Array.from(selector.options).find(function(o) { return o.value === version; });
                if (option) {
                    selector.value = version;
                    // Plots already load this version (the lazy loader reads the
                    // URL), so only the surrounding UI needs to follow.
                    applyVersion(version, { reload: false });
                }
            }
        }
    }

    /**
     * Initialize version selector on page load
     */
    async function initVersionSelector() {
        dbg('Initializing version selector...');
        dbg('Looking for version selector element...');

        const selector = document.getElementById('version-selector');
        if (!selector) {
            console.error('Version selector element not found!');
            return;
        }

        dbg('Version selector element found:', selector);

        try {
            // Fetch available versions from API
            dbg('Fetching versions from /api/versions...');
            const response = await fetch('/api/versions');
            const data = await response.json();

            if (!data.versions || data.versions.length === 0) {
                console.error('No versions available');
                return;
            }

            availableVersions = data.versions;
            dbg(`Loaded ${availableVersions.length} versions`);

            // Populate version selector dropdown
            populateVersionSelector();

            // Set up event listeners
            setupEventListeners();

            // Restore version from URL if present (must happen after options are populated)
            restoreVersionFromURL();

            dbg('Version selector initialization complete!');

        } catch (error) {
            console.error('Error initializing version selector:', error);
        }
    }

    /**
     * Populate the version selector dropdown with available versions
     */
    function populateVersionSelector() {
        const selector = document.getElementById('version-selector');
        if (!selector) {
            console.warn('Version selector element not found');
            return;
        }

        // Clear existing options
        selector.innerHTML = '';

        // Pre-select the latest version with its actual date value so the
        // dropdown never shows an empty <option value=""> placeholder (#49).
        // updateURLState below drops the version param when it equals the
        // latest, so the URL stays clean.
        availableVersions.forEach((versionObj, index) => {
            const option = document.createElement('option');
            option.value = versionObj.version;
            option.textContent = index === 0
                ? `Current (${versionObj.version})`
                : versionObj.version;
            if (index === 0) option.selected = true;
            selector.appendChild(option);
        });

        dbg('Version selector populated');
    }

    /**
     * Set up event listeners for version selection
     */
    function setupEventListeners() {
        const selector = document.getElementById('version-selector');
        if (!selector) return;

        selector.addEventListener('change', handleVersionChange);
        dbg('Event listeners set up');

        setupAopAopThresholdSlider();
        setupOntologyBranchSelector();
        setupDomainLensSelector();
    }

    /**
     * Wire the ontology coverage-holes branch selector.
     * Same contract as the threshold slider: write the choice onto the plot
     * div's dataset, drop it from the loaded set, and let reloadPlot() refetch.
     */
    function setupOntologyBranchSelector() {
        const select = document.getElementById('ontology-branch-select');
        if (!select) return;

        const plotName = select.dataset.targetPlot || 'latest_ontology_coverage_holes';
        const plotDiv = document.querySelector(`[data-plot-name="${plotName}"]`);
        if (!plotDiv) {
            console.warn(`Branch selector target plot div not found: ${plotName}`);
            return;
        }

        select.addEventListener('change', () => {
            plotDiv.dataset.branch = select.value;
            reloadPlot(plotName);
        });

        dbg('Ontology branch selector wired');
    }

    /**
     * Wire the domain-lens selector. Unlike the branch selector it drives
     * several plots at once, so the choice is written onto every one of them
     * before they are dropped from the loaded set and refetched.
     */
    function setupDomainLensSelector() {
        const select = document.getElementById('domain-lens-select');
        if (!select) return;

        select.addEventListener('change', () => {
            domainLensPlots.forEach((plotName) => {
                const plotDiv = document.querySelector(`[data-plot-name="${plotName}"]`);
                if (!plotDiv) return;
                plotDiv.dataset.domain = select.value;
                reloadPlot(plotName);
            });
        });

        dbg('Domain lens selector wired');
    }

    /**
     * Wire the AOP-AOP overlap Jaccard-similarity slider.
     * - input event updates the visible value label live (no reload)
     * - change event (slider released) updates the plot div dataset and
     *   reuses reloadPlot() to refetch with the new threshold
     */
    function setupAopAopThresholdSlider() {
        const slider = document.getElementById('aop-aop-threshold-slider');
        const label = document.getElementById('aop-aop-threshold-value');
        if (!slider || !label) return;

        const plotName = slider.dataset.targetPlot || 'latest_aop_aop_overlap';
        const plotDiv = document.querySelector(`[data-plot-name="${plotName}"]`);
        if (!plotDiv) {
            console.warn(`Slider target plot div not found: ${plotName}`);
            return;
        }

        slider.addEventListener('input', () => {
            label.textContent = slider.value;
        });

        slider.addEventListener('change', () => {
            plotDiv.dataset.minJaccard = slider.value;
            reloadPlot(plotName);
        });

        dbg('AOP-AOP threshold slider wired');
    }

    /**
     * Handle version selection change
     */
    function handleVersionChange(event) {
        applyVersion(event.target.value || null, { reload: true });
    }

    /**
     * Make `version` the page's version: shared loader state, URL, banner,
     * methodology queries, download links, data tables, and (unless the plots
     * already show it) the versioned plots.
     */
    function applyVersion(version, { reload }) {
        const latest = availableVersions[0] && availableVersions[0].version;
        // "Current" is just the latest version; treat it as no selection so the
        // banner hides and the URL stays clean.
        const newVersion = version && version !== latest ? version : null;
        if (newVersion === selectedVersion && reload) {
            dbg('Version unchanged, skipping reload');
            return;
        }
        dbg(`Version changed: ${selectedVersion || 'latest'} -> ${newVersion || 'latest'}`);
        selectedVersion = newVersion;
        if (window.PlotLazyLoader) window.PlotLazyLoader.selectedVersion = newVersion;

        updateURLState(selectedVersion || '');
        updateVersionBanner();
        updateDownloadLinks(selectedVersion);

        // Data tables reset on this event.
        document.dispatchEvent(new CustomEvent('version-changed', { detail: { version: selectedVersion } }));

        // Re-render the SPARQL methodology previews so the displayed graph URI
        // matches the selected version (issue #43).
        updateMethodologyQueryUris(selectedVersion);

        if (reload) reloadVersionedPlots();
    }

    /**
     * Point every snapshot download link (per-plot and bulk) at the selected
     * version, so a download after a switch carries that version (#171).
     */
    function updateDownloadLinks(version) {
        document.querySelectorAll('a[href^="/download/"]').forEach(link => {
            const url = new URL(link.getAttribute('href'), window.location.origin);
            const isSnapshot = url.pathname.startsWith('/download/latest')
                || (url.pathname === '/download/bulk' && (url.searchParams.get('category') || '').startsWith('latest'));
            if (!isSnapshot) return;
            if (version) url.searchParams.set('version', version);
            else url.searchParams.delete('version');
            link.setAttribute('href', url.pathname + url.search);
        });
    }

    /** "Return to Current Version" in the banner. */
    window.resetToCurrentVersion = function () {
        const selector = document.getElementById('version-selector');
        const latest = availableVersions[0] && availableVersions[0].version;
        if (!selector || !latest) return;
        selector.value = latest;
        selector.dispatchEvent(new Event('change'));
    };

    /**
     * Update the graph URI in every methodology SPARQL preview to match the
     * selected version. Walks both the visible <code> text and the
     * "Run on Endpoint" anchor href, only touching URIs that look like
     * <http://aopwiki.org/graph/YYYY-MM-DD> so multi-graph trend queries
     * (which iterate across all snapshots) are left alone.
     */
    function updateMethodologyQueryUris(version) {
        const targetVersion = version || (availableVersions[0] && availableVersions[0].version);
        if (!targetVersion) return;
        const newUriBracketed = `<http://aopwiki.org/graph/${targetVersion}>`;
        const uriRegex = /<http:\/\/aopwiki\.org\/graph\/\d{4}-\d{2}-\d{2}>/g;

        const panels = document.querySelectorAll('details.sparql-query');
        panels.forEach(panel => {
            const code = panel.querySelector('code');
            if (!code) return;

            const original = code.textContent;
            if (!uriRegex.test(original)) return;
            uriRegex.lastIndex = 0;
            const updated = original.replace(uriRegex, newUriBracketed);
            if (updated === original) return;

            code.textContent = updated;

            const link = panel.querySelector('a.sparql-run-btn');
            if (link && link.href) {
                try {
                    const url = new URL(link.href);
                    url.searchParams.set('query', updated);
                    link.href = url.toString();
                } catch (e) {
                    console.warn('Could not rebuild Run-on-Endpoint href:', e);
                }
            }
        });
    }

    /**
     * Update version banner to show which version is selected
     */
    function updateVersionBanner() {
        const banner = document.getElementById('version-banner');
        if (!banner) return;

        if (selectedVersion) {
            banner.style.display = 'block';
            banner.querySelector('.version-name').textContent = selectedVersion;
        } else {
            banner.style.display = 'none';
        }
    }

    /**
     * Reload the versioned plots through the lazy loader: visible ones queue
     * now (at most a few requests at a time), the rest load when scrolled to.
     */
    function reloadVersionedPlots() {
        if (!window.plotLoader) return;
        document.querySelectorAll('.lazy-plot').forEach(element => {
            if (window.PlotLazyLoader.isVersioned(element.dataset.plotName)) {
                window.plotLoader.reload(element);
            }
        });
    }

    /** Reload one plot after one of its settings (slider, selector) changed. */
    function reloadPlot(plotName) {
        const element = document.querySelector(`[data-plot-name="${plotName}"]`);
        if (element && window.plotLoader) window.plotLoader.reload(element);
    }

    // Initialize when DOM is ready
    // Wait a bit longer to ensure lazy loading has started
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            // Give lazy loading time to initialize
            setTimeout(initVersionSelector, 1000);
        });
    } else {
        // Give lazy loading time to initialize
        setTimeout(initVersionSelector, 1000);
    }

})();

// Add spinner animation CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);