/**
 * The plot runtime: the one place plots are fetched and drawn (#181).
 *
 * - Containers are `.lazy-plot[data-plot-name]`. A plot loads when it scrolls
 *   into view, through a queue of at most MAX_CONCURRENT requests.
 * - The request for a plot is built from the page state (static/js/state.js):
 *   the selected version for versioned plots (window.VERSIONED_PLOTS) and the
 *   plot's own settings (window.PLOT_PARAMS), both rendered from
 *   plots/registry.py. Controls never fetch; they change the state, and the
 *   runtime reloads exactly the plots that depend on what changed.
 * - A request that the state has moved past is aborted, and a response for an
 *   older selection is never drawn, so rapid switching cannot leave a stale
 *   plot on screen.
 * - Drawing goes through a renderer (mount / update / destroy). `legacy-html`
 *   draws the HTML /api/plot returns and purges the previous Plotly figure.
 * - After every draw the container dispatches a bubbling `plot:rendered`
 *   event ({plotName, url}); page code hooks that instead of watching the DOM.
 */

// Verbose per-plot tracing, off by default (#131). Enable from devtools with
//   localStorage.setItem('plotDebug', '1')
const PLOT_DEBUG = (() => {
    try { return localStorage.getItem('plotDebug') === '1'; } catch (e) { return false; }
})();
function dbg(...args) { if (PLOT_DEBUG) console.log('[runtime]', ...args); }

const SUPERSEDED = 'superseded';

/** Draws /api/plot HTML: inject it, run its init scripts, size the figure. */
const legacyHtmlRenderer = {
    mount(element, payload) {
        element.innerHTML = payload.html;
        // innerHTML-inserted <script> tags don't run; re-inject them so
        // Plotly.newPlot(...) executes in global scope (no eval, #131).
        element.querySelectorAll('script').forEach((old) => {
            if (!old.textContent) return;
            try {
                const s = document.createElement('script');
                if (old.type) s.type = old.type;
                s.textContent = old.textContent;
                document.body.appendChild(s);
                document.body.removeChild(s);
            } catch (error) {
                console.error(`Plot script failed for ${element.dataset.plotName}:`, error);
            }
        });
        const div = element.querySelector('.plotly-graph-div');
        if (div && window.Plotly) setTimeout(() => window.Plotly.Plots.resize(div), 100);
    },
    update(element, payload) {
        this.destroy(element);
        this.mount(element, payload);
    },
    destroy(element) {
        if (!window.Plotly) return;
        element.querySelectorAll('.js-plotly-plot').forEach((div) => {
            try { window.Plotly.purge(div); } catch (e) { /* already gone */ }
        });
    },
};

class PlotRuntime {
    constructor(state) {
        this.state = state;
        this.queue = [];             // containers waiting for a slot
        this.active = 0;             // requests in flight
        this.inflight = new Map();   // container -> {controller, url}
        this.drawn = new Map();      // container -> url of what is on screen
        this.mounted = new Set();    // containers holding a drawn plot
        this.renderers = { 'legacy-html': legacyHtmlRenderer };

        this.observer = new IntersectionObserver(
            (entries) => entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    this.observer.unobserve(entry.target);
                    this.load(entry.target);
                }
            }),
            { root: null, rootMargin: '100px', threshold: 0.1 },
        );
        this.containers().forEach((el) => this.observer.observe(el));

        document.querySelectorAll('.tab-button').forEach((button) => {
            button.addEventListener('click', () => setTimeout(() => this.loadVisible(), 100));
        });
        if (state) state.subscribe((changed) => this.onStateChange(changed));
        this.loadVisible();
    }

    containers() {
        return Array.from(document.querySelectorAll('.lazy-plot[data-plot-name]'));
    }

    static isVersioned(plotName) {
        return Array.isArray(window.VERSIONED_PLOTS) && window.VERSIONED_PLOTS.includes(plotName);
    }

    static paramsOf(plotName) {
        return (window.PLOT_PARAMS && window.PLOT_PARAMS[plotName]) || [];
    }

    /** The state keys a plot's request depends on. */
    static dependencies(plotName) {
        const deps = PlotRuntime.paramsOf(plotName).slice();
        if (PlotRuntime.isVersioned(plotName)) deps.push('version');
        return deps;
    }

    /** The one place a plot request URL is built. */
    plotUrl(element) {
        const plotName = element.dataset.plotName;
        const params = new URLSearchParams();
        const get = (k) => (this.state ? this.state.get(k) : null);
        if (PlotRuntime.isVersioned(plotName) && get('version')) params.set('version', get('version'));
        PlotRuntime.paramsOf(plotName).forEach((k) => {
            const v = get(k);
            if (v !== null && v !== undefined) params.set(k, v);
        });
        const qs = params.toString();
        return `/api/plot/${plotName}` + (qs ? `?${qs}` : '');
    }

    isVisible(element) {
        const rect = element.getBoundingClientRect();
        return rect.top < window.innerHeight && rect.bottom > 0
            && getComputedStyle(element).display !== 'none';
    }

    loadVisible() {
        this.containers().forEach((el) => { if (this.isVisible(el)) this.load(el); });
    }

    onStateChange(changed) {
        this.containers().forEach((el) => {
            const deps = PlotRuntime.dependencies(el.dataset.plotName);
            if (deps.some((k) => changed.includes(k))) this.reload(el);
        });
    }

    /** Queue a container unless what it would fetch is already drawn, queued or in flight. */
    load(element) {
        const url = this.plotUrl(element);
        if (this.drawn.get(element) === url) return;
        const running = this.inflight.get(element);
        if (running && running.url === url) return;
        if (running) running.controller.abort(SUPERSEDED);
        if (this.queue.includes(element)) return;  // its URL is read when it starts
        // Never leave a plot for another selection on screen while its
        // replacement loads.
        this.unmount(element);
        this.showLoading(element);
        this.queue.push(element);
        this.pump();
    }

    /** Re-render for the current state: now if visible, else when scrolled to. */
    reload(element) {
        if (this.isVisible(element)) {
            this.load(element);
            return;
        }
        const running = this.inflight.get(element);
        if (running) running.controller.abort(SUPERSEDED);
        this.drawn.delete(element);
        this.observer.observe(element);
    }

    pump() {
        while (this.active < PlotRuntime.MAX_CONCURRENT && this.queue.length) {
            const element = this.queue.shift();
            this.active++;
            this.fetchAndDraw(element).finally(() => {
                this.active--;
                this.pump();
            });
        }
    }

    async fetchAndDraw(element) {
        const plotName = element.dataset.plotName;
        const url = this.plotUrl(element);
        const controller = new AbortController();
        this.inflight.set(element, { controller, url });
        // Abort a hung request so the user gets an error and a Retry (#129);
        // just under gunicorn's 120 s worker timeout (#170).
        const timer = setTimeout(() => controller.abort('timeout'), PlotRuntime.LOAD_TIMEOUT_MS);
        try {
            dbg('fetch', url);
            const response = await fetch(url, { signal: controller.signal });
            const data = await response.json();
            if (this.plotUrl(element) !== url) return;  // the state moved on
            if (data.success) {
                this.draw(element, data, url);
            } else {
                this.showError(element, data.error || 'Unknown error');
            }
        } catch (error) {
            if (controller.signal.aborted && controller.signal.reason === SUPERSEDED) {
                dbg('superseded', url);
            } else if (controller.signal.aborted) {
                const seconds = Math.round(PlotRuntime.LOAD_TIMEOUT_MS / 1000);
                this.showError(element, `Request timeout after ${seconds}s`);
            } else {
                console.error(`Plot ${plotName} failed:`, error);
                this.showError(element, 'Failed to load plot: ' + error.message);
            }
        } finally {
            clearTimeout(timer);
            if (this.inflight.get(element)?.controller === controller) this.inflight.delete(element);
            // The state changed while this ran: fetch what it asks for now.
            if (this.plotUrl(element) !== url && this.drawn.get(element) !== this.plotUrl(element)) {
                this.reload(element);
            }
        }
    }

    draw(element, payload, url) {
        const renderer = this.renderers[element.dataset.renderer || 'legacy-html'];
        if (this.mounted.has(element)) renderer.update(element, payload);
        else renderer.mount(element, payload);
        this.mounted.add(element);
        this.drawn.set(element, url);
        element.setAttribute('data-rendered-url', url);
        element.dispatchEvent(new CustomEvent('plot:rendered', {
            bubbles: true, detail: { plotName: element.dataset.plotName, url },
        }));
    }

    unmount(element) {
        if (!this.mounted.has(element)) return;
        const renderer = this.renderers[element.dataset.renderer || 'legacy-html'];
        renderer.destroy(element);
        this.mounted.delete(element);
        this.drawn.delete(element);
    }

    showLoading(element) {
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

    showError(element, error) {
        this.unmount(element);
        const lower = String(error).toLowerCase();
        let icon = '⚠', title = 'Unable to Load Plot',
            suggestion = 'An unexpected error occurred while loading this visualization.';
        if (lower.includes('timeout')) {
            icon = '⏱';
            title = 'Query Timed Out';
            suggestion = 'This visualization requires a complex query. Try again in a moment.';
        } else if (lower.includes('fetch') || lower.includes('network') || lower.includes('503')) {
            title = 'Service Unreachable';
            suggestion = 'The data service is currently unavailable. Please check back later.';
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
        const element = document.querySelector(`.lazy-plot[data-plot-name="${plotName}"]`);
        if (!element) return;
        this.drawn.delete(element);
        this.load(element);
    }
}

/** Cap on in-flight /api/plot requests. */
PlotRuntime.MAX_CONCURRENT = 4;
/** Give up on a request after this long (kept under gunicorn's 120 s timeout, #170). */
PlotRuntime.LOAD_TIMEOUT_MS = 110000;

window.PlotRuntime = PlotRuntime;

document.addEventListener('DOMContentLoaded', () => {
    window.plotLoader = new PlotRuntime(window.DashState || null);
});

window.addEventListener('resize', () => {
    if (!window.Plotly) return;
    document.querySelectorAll('.plotly-graph-div').forEach((div) => window.Plotly.Plots.resize(div));
});
