/**
 * Lazy Loading System for AOP-Wiki Dashboard Plots
 *
 * This script implements progressive loading of plots to improve initial page load time
 * from 28+ seconds to under 2 seconds by loading plots on-demand.
 */

// Verbose per-plot tracing, off by default so a normal page load is quiet
// (was ~52 console.log lines per /snapshot, #131). Enable from devtools with
//   localStorage.setItem('plotDebug', '1')
// then reload. Warnings and errors always log.
const PLOT_DEBUG = (() => {
    try { return localStorage.getItem('plotDebug') === '1'; } catch (e) { return false; }
})();
function dbg(...args) { if (PLOT_DEBUG) console.log(...args); }

class PlotLazyLoader {
    constructor() {
        this.loadedPlots = new Set();
        this.loadingPlots = new Set();
        this.queue = [];
        this.activeCount = 0;
        this.observer = null;
        this.init();
    }

    init() {
        dbg("PlotLazyLoader: Initializing lazy loading system");

        // Set up Intersection Observer for lazy loading
        this.observer = new IntersectionObserver(
            (entries) => this.handleIntersection(entries),
            {
                root: null,
                rootMargin: '100px', // Start loading 100px before element is visible
                threshold: 0.1
            }
        );

        // Observe all lazy plot containers
        const lazyPlots = document.querySelectorAll('.lazy-plot');
        dbg(`PlotLazyLoader: Found ${lazyPlots.length} lazy-plot containers`);

        lazyPlots.forEach(element => {
            dbg(`PlotLazyLoader: Observing plot: ${element.dataset.plotName}`);
            this.observer.observe(element);
        });

        // Handle tab switching to ensure visible plots are loaded
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', () => {
                setTimeout(() => this.loadVisiblePlots(), 100);
            });
        });

        // Load initial visible plots
        this.loadVisiblePlots();
    }

    handleIntersection(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                this.loadPlot(entry.target);
                this.observer.unobserve(entry.target);
            }
        });
    }

    loadVisiblePlots() {
        document.querySelectorAll('.lazy-plot').forEach(element => {
            if (this.isElementVisible(element)) {
                this.loadPlot(element);
            }
        });
    }

    isElementVisible(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top < window.innerHeight &&
            rect.bottom > 0 &&
            getComputedStyle(element).display !== 'none'
        );
    }

    /**
     * Queue a plot for loading. The skeleton appears immediately; the request
     * itself waits for a free slot.
     */
    loadPlot(element) {
        const plotName = element.dataset.plotName;
        dbg(`PlotLazyLoader: Attempting to load plot: ${plotName}`);

        if (this.loadedPlots.has(plotName) || this.loadingPlots.has(plotName)) {
            dbg(`PlotLazyLoader: Plot ${plotName} already loaded or loading, skipping`);
            return;
        }

        this.loadingPlots.add(plotName);
        this.showLoadingState(element);
        this.queue.push(element);
        this.pumpQueue();
    }

    /**
     * Start queued fetches up to MAX_CONCURRENT.
     *
     * Every /api/plot/<name> call runs a SPARQL query server-side. Scrolling
     * quickly used to fire all of them at once, so the browser's per-host
     * connection cap queued them opaquely and the tail requests starved — which
     * is what produced apparently-stuck plots. Bounding it here keeps the wait
     * visible to the timeout in fetchPlot() instead.
     */
    pumpQueue() {
        while (this.activeCount < PlotLazyLoader.MAX_CONCURRENT && this.queue.length > 0) {
            const element = this.queue.shift();
            this.activeCount++;
            this.fetchPlot(element).finally(() => {
                this.activeCount--;
                this.pumpQueue();
            });
        }
    }

    async fetchPlot(element) {
        const plotName = element.dataset.plotName;
        dbg(`PlotLazyLoader: Starting load for plot: ${plotName}`);

        // Without an abort signal a hung backend leaves the promise unsettled
        // forever: the spinner stays up, the catch below never runs, and the
        // user gets no error and no Retry (issue #129).
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), PlotLazyLoader.LOAD_TIMEOUT_MS);

        try {
            // Forward any data-* attributes that map to plot kwargs (scope, view).
            // Lets the coverage toggles drive the initial render via markup alone.
            const params = new URLSearchParams();
            if (element.dataset.scope) params.set('scope', element.dataset.scope);
            if (element.dataset.view) params.set('view', element.dataset.view);
            const qs = params.toString();
            const url = qs ? `/api/plot/${plotName}?${qs}` : `/api/plot/${plotName}`;

            // Fetch plot data
            dbg(`PlotLazyLoader: Fetching data for plot: ${plotName} (${url})`);
            const response = await fetch(url, { signal: controller.signal });
            dbg(`PlotLazyLoader: Response status for ${plotName}: ${response.status}`);

            const data = await response.json();
            dbg(`PlotLazyLoader: Response data for ${plotName}:`, {
                success: data.success,
                hasHtml: !!data.html,
                htmlLength: data.html?.length || 0
            });

            if (data.success) {
                // Replace skeleton with actual plot
                dbg(`PlotLazyLoader: Rendering plot ${plotName}`);
                element.innerHTML = data.html;
                this.loadedPlots.add(plotName);

                // Run the inline init scripts from the fetched HTML by
                // re-injecting them as fresh <script> elements, rather than
                // eval() (#131). innerHTML-inserted <script> tags don't execute
                // on their own; a re-injected one runs normally in global scope,
                // which is exactly what Plotly's `Plotly.newPlot(<id>, …)` init
                // needs. (This removes the eval; it does not by itself enable a
                // strict CSP — inline scripts still run. That's a separate task.)
                const scripts = element.querySelectorAll('script');
                scripts.forEach(oldScript => {
                    if (!oldScript.textContent) return;
                    try {
                        dbg(`PlotLazyLoader: Executing script for ${plotName}`);
                        const s = document.createElement('script');
                        if (oldScript.type) s.type = oldScript.type;
                        s.textContent = oldScript.textContent;
                        document.body.appendChild(s);
                        document.body.removeChild(s);
                    } catch (error) {
                        console.error(`PlotLazyLoader: Script execution error for ${plotName}:`, error);
                    }
                });

                // Check if Plotly is available
                dbg(`PlotLazyLoader: Plotly available: ${!!window.Plotly}`);

                // Trigger Plotly resize if needed
                if (window.Plotly) {
                    const plotDiv = element.querySelector('.plotly-graph-div');
                    if (plotDiv) {
                        dbg(`PlotLazyLoader: Resizing Plotly plot: ${plotName}`);
                        setTimeout(() => {
                            window.Plotly.Plots.resize(plotDiv);
                        }, 100);
                    } else {
                        dbg(`PlotLazyLoader: No plotly-graph-div found for: ${plotName}`);
                    }
                } else {
                    console.warn(`PlotLazyLoader: Plotly not available for plot: ${plotName}`);
                }

                dbg(`PlotLazyLoader: Successfully loaded plot: ${plotName}`);
            } else {
                console.error(`PlotLazyLoader: Plot ${plotName} failed:`, data.error);
                this.showErrorState(element, data.error);
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                const seconds = Math.round(PlotLazyLoader.LOAD_TIMEOUT_MS / 1000);
                console.error(`PlotLazyLoader: Timeout loading plot ${plotName} after ${seconds}s`);
                // Phrased so showErrorState() picks its timeout branch.
                this.showErrorState(element, `Request timeout after ${seconds}s`);
            } else {
                console.error(`PlotLazyLoader: Exception loading plot ${plotName}:`, error);
                this.showErrorState(element, 'Failed to load plot: ' + error.message);
            }
        } finally {
            clearTimeout(timeoutId);
            this.loadingPlots.delete(plotName);
        }
    }

    showLoadingState(element) {
        element.innerHTML = `
            <div class="plot-skeleton">
                <div class="skeleton-header"></div>
                <div class="skeleton-chart">
                    <div class="loading-spinner"></div>
                    <p>Loading plot...</p>
                </div>
            </div>
        `;
    }

    showErrorState(element, error) {
        const errorLower = error.toLowerCase();
        const isTimeout = errorLower.includes('timeout');
        const isUnreachable = errorLower.includes('fetch') ||
                              errorLower.includes('network') ||
                              errorLower.includes('503');

        let icon, title, suggestion;
        if (isTimeout) {
            icon = '\u23F1';  // Stopwatch
            title = 'Query Timed Out';
            suggestion = 'This visualization requires a complex query. Try again in a moment.';
        } else if (isUnreachable) {
            icon = '\u26A0';  // Warning sign
            title = 'Service Unreachable';
            suggestion = 'The data service is currently unavailable. Please check back later.';
        } else {
            icon = '\u26A0';  // Warning sign
            title = 'Unable to Load Plot';
            suggestion = 'An unexpected error occurred while loading this visualization.';
        }

        const plotName = element.dataset.plotName;
        element.innerHTML = `
            <div class="plot-error">
                <div class="error-icon">${icon}</div>
                <h4 class="error-title">${title}</h4>
                <p class="error-suggestion">${suggestion}</p>
                <button class="error-retry-btn" onclick="plotLoader.retryPlot('${plotName}')">Retry</button>
            </div>
        `;
    }

    retryPlot(plotName) {
        this.loadedPlots.delete(plotName);
        this.loadingPlots.delete(plotName);
        const element = document.querySelector(`[data-plot-name="${plotName}"]`);
        if (element) {
            this.loadPlot(element);
        }
    }
}

/** Cap on in-flight /api/plot requests; each one runs a SPARQL query. */
PlotLazyLoader.MAX_CONCURRENT = 4;

/** Give up on a plot request after this long and show the retryable error. */
PlotLazyLoader.LOAD_TIMEOUT_MS = 45000;

// Initialize lazy loader when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.plotLoader = new PlotLazyLoader();
});

// Handle window resize for responsive plots
window.addEventListener('resize', () => {
    if (window.Plotly) {
        document.querySelectorAll('.plotly-graph-div').forEach(div => {
            window.Plotly.Plots.resize(div);
        });
    }
});