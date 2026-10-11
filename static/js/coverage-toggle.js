/**
 * Coverage Analysis toggle controller.
 *
 * Drives the unified organ-coverage stacked bar + bucket-totals bar via two toggles:
 *   - scope: all | apical | ao
 *   - view:  absolute | percentage (stacked bar only)
 *
 * Scope is shared between both views so the user reads a single coherent
 * slice across the section. Requests go through the lazy loader, which adds
 * the selected version.
 */
(function () {
    'use strict';

    const BAR_PLOT = 'latest_organ_coverage_unified';
    const PIE_PLOT = 'latest_organ_coverage_pie';

    /**
     * Put the new scope/view on the plot container and let the lazy loader
     * refetch it. The loader adds the selected version and drops responses for
     * an older selection (#171).
     */
    function fetchPlot(plotName, scope, view) {
        const container = document.querySelector(`[data-plot-name="${plotName}"]`);
        if (!container) return;
        if (scope !== undefined) container.dataset.scope = scope;
        if (view !== undefined) container.dataset.view = view;
        if (window.plotLoader) window.plotLoader.reload(container);
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
