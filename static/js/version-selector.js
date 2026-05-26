/**
 * Version Selector for AOP-Wiki RDF Dashboard
 *
 * Enables users to select and view Latest Data plots for any historical database version.
 * Dynamically regenerates plots when version changes.
 */

(function() {
    'use strict';

    let selectedVersion = null;  // null = latest version
    let availableVersions = [];
    let loadingPlots = new Set();

    // List of plot names that support version selection (all 11 plots)
    const versionedPlots = [
        'latest_entity_counts',
        'latest_ke_components',
        'latest_aop_connectivity',
        'latest_avg_per_aop',
        'latest_process_usage',
        'latest_object_usage',
        'latest_aop_completeness',
        'latest_aop_completeness_unique',
        'latest_ontology_usage',
        'latest_database_summary',
        'latest_ke_annotation_depth',
        'latest_ke_by_bio_level',
        'latest_taxonomic_groups',
        'latest_entity_by_oecd_status',
        'latest_ke_reuse',
        'latest_ke_reuse_distribution',
        'latest_ontology_diversity',
        'latest_organ_coverage_unified',
        'latest_organ_coverage_pie',
        'latest_multi_organ_aops',
        'latest_life_stage',
        'latest_aop_aop_overlap'
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
                    // Trigger change event to load plots for this version
                    selector.dispatchEvent(new Event('change'));
                }
            }
        }
    }

    /**
     * Initialize version selector on page load
     */
    async function initVersionSelector() {
        console.log('Initializing version selector...');
        console.log('Looking for version selector element...');

        const selector = document.getElementById('version-selector');
        if (!selector) {
            console.error('Version selector element not found!');
            return;
        }

        console.log('Version selector element found:', selector);

        try {
            // Fetch available versions from API
            console.log('Fetching versions from /api/versions...');
            const response = await fetch('/api/versions');
            const data = await response.json();

            if (!data.versions || data.versions.length === 0) {
                console.error('No versions available');
                return;
            }

            availableVersions = data.versions;
            console.log(`Loaded ${availableVersions.length} versions`);

            // Populate version selector dropdown
            populateVersionSelector();

            // Set up event listeners
            setupEventListeners();

            // Restore version from URL if present (must happen after options are populated)
            restoreVersionFromURL();

            console.log('Version selector initialization complete!');

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

        console.log('Version selector populated');
    }

    /**
     * Set up event listeners for version selection
     */
    function setupEventListeners() {
        const selector = document.getElementById('version-selector');
        if (!selector) return;

        selector.addEventListener('change', handleVersionChange);
        console.log('Event listeners set up');

        setupAopAopThresholdSlider();
    }

    /**
     * Wire the AOP-AOP overlap threshold slider.
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
            plotDiv.dataset.minSharedKes = slider.value;
            if (window.plotLoader) {
                window.plotLoader.loadedPlots.delete(plotName);
            }
            reloadPlot(plotName);
        });

        console.log('AOP-AOP threshold slider wired');
    }

    /**
     * Handle version selection change
     */
    async function handleVersionChange(event) {
        console.log('VERSION CHANGE EVENT FIRED!');
        console.log('Event target:', event.target);
        console.log('Event target value:', event.target.value);

        const newVersion = event.target.value || null;  // Empty string = latest

        console.log(`New version: "${newVersion}", Current version: "${selectedVersion}"`);

        if (newVersion === selectedVersion) {
            console.log('Version unchanged, skipping reload');
            return; // No change
        }

        console.log(`Version changed: ${selectedVersion} → ${newVersion || 'latest'}`);
        selectedVersion = newVersion;

        // Update URL with version parameter
        updateURLState(selectedVersion || '');

        // Update UI to show selected version
        console.log('Updating version banner...');
        updateVersionBanner();

        // Dispatch version-changed event so data tables can reset
        document.dispatchEvent(new CustomEvent('version-changed', { detail: { version: selectedVersion } }));

        // Re-render the SPARQL methodology previews so the displayed graph URI
        // matches the selected version (issue #43).
        updateMethodologyQueryUris(selectedVersion);

        // Reload all versioned plots
        console.log('Starting plot reload...');
        await reloadVersionedPlots();
        console.log('Plot reload complete!');
    }

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
     * Reload all versioned plots with the selected version
     */
    async function reloadVersionedPlots() {
        console.log(`Reloading ${versionedPlots.length} plots for version: ${selectedVersion || 'latest'}`);

        const reloadPromises = versionedPlots.map(plotName => reloadPlot(plotName));

        try {
            await Promise.all(reloadPromises);
            console.log('All plots reloaded successfully');
        } catch (error) {
            console.error('Error reloading plots:', error);
        }
    }

    /**
     * Reload a single plot with the selected version
     */
    async function reloadPlot(plotName) {
        if (loadingPlots.has(plotName)) {
            console.log(`Plot ${plotName} already loading, skipping...`);
            return;
        }

        loadingPlots.add(plotName);
        const plotDiv = document.querySelector(`[data-plot-name="${plotName}"]`);

        if (!plotDiv) {
            console.warn(`Plot div not found for ${plotName}`);
            loadingPlots.delete(plotName);
            return;
        }

        // Mark this plot as manually loaded so lazy-loader won't reload it
        if (window.plotLoader) {
            window.plotLoader.loadedPlots.add(plotName);
            console.log(`Marked ${plotName} as loaded in lazy-loader cache`);
        }

        // Show loading spinner
        showLoadingSpinner(plotDiv);

        try {
            // Build API URL with version + any toggle parameters from the
            // plot container (scope/view drive the unified coverage views).
            const params = new URLSearchParams();
            if (selectedVersion) params.set('version', selectedVersion);
            if (plotDiv.dataset.scope) params.set('scope', plotDiv.dataset.scope);
            if (plotDiv.dataset.view) params.set('view', plotDiv.dataset.view);
            if (plotDiv.dataset.minSharedKes) params.set('min_shared_kes', plotDiv.dataset.minSharedKes);
            const qs = params.toString();
            const url = qs ? `/api/plot/${plotName}?${qs}` : `/api/plot/${plotName}`;

            console.log(`Fetching plot from: ${url}`);

            const response = await fetch(url);
            const data = await response.json();

            if (data.success && data.html) {
                // Replace plot content
                plotDiv.innerHTML = data.html;

                // Execute any script tags in the HTML (needed for Plotly plots)
                const scripts = plotDiv.querySelectorAll('script');
                scripts.forEach(script => {
                    const newScript = document.createElement('script');
                    if (script.src) {
                        newScript.src = script.src;
                    } else {
                        newScript.textContent = script.textContent;
                    }
                    script.parentNode.replaceChild(newScript, script);
                });

                console.log(`Plot ${plotName} loaded successfully`);
            } else {
                throw new Error(data.error || 'Failed to load plot');
            }

        } catch (error) {
            console.error(`Error loading plot ${plotName}:`, error);
            plotDiv.innerHTML = `<p style="color: red;">Error loading plot: ${error.message}</p>`;
        } finally {
            loadingPlots.delete(plotName);
        }
    }

    /**
     * Show loading spinner in plot container
     */
    function showLoadingSpinner(container) {
        container.innerHTML = `
            <div style="display: flex; justify-content: center; align-items: center; min-height: 300px;">
                <div style="text-align: center;">
                    <div class="spinner" style="
                        border: 4px solid #f3f3f3;
                        border-top: 4px solid #29235C;
                        border-radius: 50%;
                        width: 40px;
                        height: 40px;
                        animation: spin 1s linear infinite;
                        margin: 0 auto 10px;
                    "></div>
                    <p style="color: #666;">Loading plot...</p>
                </div>
            </div>
        `;
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