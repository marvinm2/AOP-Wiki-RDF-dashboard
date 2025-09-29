/**
 * Lazy Loading System for AOP-Wiki Dashboard Plots
 *
 * This script implements progressive loading of plots to improve initial page load time
 * from 28+ seconds to under 2 seconds by loading plots on-demand.
 */

class PlotLazyLoader {
    constructor() {
        this.loadedPlots = new Set();
        this.loadingPlots = new Set();
        this.observer = null;
        this.init();
    }

    init() {
        console.log("PlotLazyLoader: Initializing lazy loading system");

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
        console.log(`PlotLazyLoader: Found ${lazyPlots.length} lazy-plot containers`);

        lazyPlots.forEach(element => {
            console.log(`PlotLazyLoader: Observing plot: ${element.dataset.plotName}`);
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

    async loadPlot(element) {
        const plotName = element.dataset.plotName;
        console.log(`PlotLazyLoader: Attempting to load plot: ${plotName}`);

        if (this.loadedPlots.has(plotName) || this.loadingPlots.has(plotName)) {
            console.log(`PlotLazyLoader: Plot ${plotName} already loaded or loading, skipping`);
            return;
        }

        this.loadingPlots.add(plotName);
        console.log(`PlotLazyLoader: Starting load for plot: ${plotName}`);

        try {
            // Show loading state
            this.showLoadingState(element);

            // Fetch plot data
            console.log(`PlotLazyLoader: Fetching data for plot: ${plotName}`);
            const response = await fetch(`/api/plot/${plotName}`);
            console.log(`PlotLazyLoader: Response status for ${plotName}: ${response.status}`);

            const data = await response.json();
            console.log(`PlotLazyLoader: Response data for ${plotName}:`, {
                success: data.success,
                hasHtml: !!data.html,
                htmlLength: data.html?.length || 0
            });

            if (data.success) {
                // Replace skeleton with actual plot
                console.log(`PlotLazyLoader: Rendering plot ${plotName}`);
                element.innerHTML = data.html;
                this.loadedPlots.add(plotName);

                // Execute any script tags in the inserted HTML
                const scripts = element.querySelectorAll('script');
                scripts.forEach(script => {
                    if (script.innerHTML) {
                        try {
                            console.log(`PlotLazyLoader: Executing script for ${plotName}`);
                            eval(script.innerHTML);
                        } catch (error) {
                            console.error(`PlotLazyLoader: Script execution error for ${plotName}:`, error);
                        }
                    }
                });

                // Check if Plotly is available
                console.log(`PlotLazyLoader: Plotly available: ${!!window.Plotly}`);

                // Trigger Plotly resize if needed
                if (window.Plotly) {
                    const plotDiv = element.querySelector('.plotly-graph-div');
                    if (plotDiv) {
                        console.log(`PlotLazyLoader: Resizing Plotly plot: ${plotName}`);
                        setTimeout(() => {
                            window.Plotly.Plots.resize(plotDiv);
                        }, 100);
                    } else {
                        console.log(`PlotLazyLoader: No plotly-graph-div found for: ${plotName}`);
                    }
                } else {
                    console.warn(`PlotLazyLoader: Plotly not available for plot: ${plotName}`);
                }

                console.log(`PlotLazyLoader: Successfully loaded plot: ${plotName}`);
            } else {
                console.error(`PlotLazyLoader: Plot ${plotName} failed:`, data.error);
                this.showErrorState(element, data.error);
            }
        } catch (error) {
            console.error(`PlotLazyLoader: Exception loading plot ${plotName}:`, error);
            this.showErrorState(element, 'Failed to load plot: ' + error.message);
        } finally {
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
        element.innerHTML = `
            <div class="plot-error">
                <p>⚠️ Unable to load plot</p>
                <small>${error}</small>
                <button onclick="plotLoader.retryPlot('${element.dataset.plotName}')">Retry</button>
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