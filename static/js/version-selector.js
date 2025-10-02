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
        'latest_network_density',
        'latest_avg_per_aop',
        'latest_process_usage',
        'latest_object_usage',
        'latest_aop_completeness',
        'latest_aop_completeness_unique',
        'latest_ontology_usage',
        'latest_database_summary',
        'latest_ke_annotation_depth'
    ];

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

        // Add "Current" option
        const latestOption = document.createElement('option');
        latestOption.value = '';
        latestOption.textContent = `Current (${availableVersions[0].version})`;
        latestOption.selected = true;
        selector.appendChild(latestOption);

        // Add historical versions
        availableVersions.forEach((versionObj, index) => {
            if (index === 0) return; // Skip first (already added as "Current")

            const option = document.createElement('option');
            option.value = versionObj.version;
            option.textContent = versionObj.version;
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

        // Update UI to show selected version
        console.log('Updating version banner...');
        updateVersionBanner();

        // Reload all versioned plots
        console.log('Starting plot reload...');
        await reloadVersionedPlots();
        console.log('Plot reload complete!');
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
            // Build API URL with version parameter
            const url = selectedVersion
                ? `/api/plot/${plotName}?version=${encodeURIComponent(selectedVersion)}`
                : `/api/plot/${plotName}`;

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