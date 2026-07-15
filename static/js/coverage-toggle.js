/**
 * Coverage Analysis toggle controller.
 *
 * Drives the unified organ-coverage stacked bar + bucket-totals bar via two toggles:
 *   - scope: all | apical | ao
 *   - view:  absolute | percentage (stacked bar only)
 *
 * Scope is shared between both views so the user reads a single coherent
 * slice across the section. Version is read from the URL so the
 * toggles play well with the version selector.
 */
(function () {
    'use strict';

    const BAR_PLOT = 'latest_organ_coverage_unified';
    const PIE_PLOT = 'latest_organ_coverage_pie';

    function getVersion() {
        const params = new URLSearchParams(window.location.search);
        return params.get('version') || '';
    }

    function buildUrl(plotName, scope, view) {
        const params = new URLSearchParams();
        const version = getVersion();
        if (version) params.set('version', version);
        if (scope) params.set('scope', scope);
        if (view) params.set('view', view);
        const qs = params.toString();
        return qs ? `/api/plot/${plotName}?${qs}` : `/api/plot/${plotName}`;
    }

    function showSpinner(container) {
        container.innerHTML = `
            <div style="display:flex; justify-content:center; align-items:center; min-height:300px;">
                <div style="text-align:center;">
                    <div class="spinner" style="
                        border:4px solid #f3f3f3;
                        border-top:4px solid #29235C;
                        border-radius:50%;
                        width:40px;
                        height:40px;
                        animation:spin 1s linear infinite;
                        margin:0 auto 10px;
                    "></div>
                    <p style="color:#666;">Loading plot...</p>
                </div>
            </div>
        `;
    }

    function executeScripts(container) {
        const scripts = container.querySelectorAll('script');
        scripts.forEach((script) => {
            const fresh = document.createElement('script');
            if (script.src) {
                fresh.src = script.src;
            } else {
                fresh.textContent = script.textContent;
            }
            script.parentNode.replaceChild(fresh, script);
        });
    }

    async function fetchPlot(plotName, scope, view) {
        const container = document.querySelector(`[data-plot-name="${plotName}"]`);
        if (!container) return;
        // Keep the dataset attributes in sync so version-selector reloads pick
        // up the current scope/view.
        if (scope !== undefined) container.dataset.scope = scope;
        if (view !== undefined) container.dataset.view = view;
        // Prevent the lazy-loader from clobbering this fetch with a stale one.
        if (window.plotLoader) {
            window.plotLoader.loadedPlots.add(plotName);
            window.plotLoader.loadingPlots.delete(plotName);
        }

        showSpinner(container);
        try {
            const response = await fetch(buildUrl(plotName, scope, view));
            const data = await response.json();
            if (data.success && data.html) {
                container.innerHTML = data.html;
                executeScripts(container);
            } else {
                container.innerHTML = `<p style="color:#c00;">Error loading plot: ${data.error || 'unknown'}</p>`;
            }
        } catch (err) {
            container.innerHTML = `<p style="color:#c00;">Error loading plot: ${err.message}</p>`;
        }
    }

    function setActive(buttons, value, attrName) {
        buttons.forEach((btn) => {
            if (btn.dataset[attrName] === value) {
                btn.classList.add('is-active');
                btn.setAttribute('aria-pressed', 'true');
            } else {
                btn.classList.remove('is-active');
                btn.setAttribute('aria-pressed', 'false');
            }
        });
    }

    function init() {
        const toggleRoot = document.querySelector('.coverage-toggle');
        if (!toggleRoot) return;

        const scopeButtons = Array.from(toggleRoot.querySelectorAll('[data-toggle-scope]'));
        const viewButtons = Array.from(toggleRoot.querySelectorAll('[data-toggle-view]'));

        let currentScope = toggleRoot.dataset.scope || 'all';
        let currentView = toggleRoot.dataset.view || 'absolute';

        setActive(scopeButtons, currentScope, 'toggleScope');
        setActive(viewButtons, currentView, 'toggleView');

        scopeButtons.forEach((btn) => {
            btn.addEventListener('click', () => {
                const next = btn.dataset.toggleScope;
                if (!next || next === currentScope) return;
                currentScope = next;
                toggleRoot.dataset.scope = next;
                setActive(scopeButtons, currentScope, 'toggleScope');
                // Both views follow scope.
                fetchPlot(BAR_PLOT, currentScope, currentView);
                fetchPlot(PIE_PLOT, currentScope, undefined);
            });
        });

        viewButtons.forEach((btn) => {
            btn.addEventListener('click', () => {
                const next = btn.dataset.toggleView;
                if (!next || next === currentView) return;
                currentView = next;
                toggleRoot.dataset.view = next;
                setActive(viewButtons, currentView, 'toggleView');
                // Only the bar has an abs/% distinction.
                fetchPlot(BAR_PLOT, currentScope, currentView);
            });
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
