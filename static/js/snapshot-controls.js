/**
 * Plot settings on the Snapshot page (#181): the coverage scope/view toggles,
 * the overlap threshold slider, the ontology-branch selector and the domain
 * selector.
 *
 * Each control only writes its value into the page state (static/js/state.js);
 * the runtime reloads the plots that take that setting. A control's initial
 * value comes from the URL, so a shared link opens with the same settings. A
 * value equal to the control's default is stored as null, which keeps it out
 * of the URL and lets the server apply the same default.
 */
(function () {
    'use strict';

    const state = window.DashState;
    if (!state) return;

    /** Store `value`, or null when it is the control's default. */
    function setSetting(key, value, defaultValue) {
        state.set({ [key]: value === defaultValue ? null : value });
    }

    function setPressed(buttons, attr, value) {
        buttons.forEach((btn) => {
            const on = btn.dataset[attr] === value;
            btn.classList.toggle('is-active', on);
            btn.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
    }

    function wireCoverageToggles() {
        const root = document.querySelector('.coverage-toggle');
        if (!root) return;
        const scopeButtons = Array.from(root.querySelectorAll('[data-toggle-scope]'));
        const viewButtons = Array.from(root.querySelectorAll('[data-toggle-view]'));
        const defaults = { scope: root.dataset.scope || 'all', view: root.dataset.view || 'absolute' };

        const show = () => {
            setPressed(scopeButtons, 'toggleScope', state.get('scope') || defaults.scope);
            setPressed(viewButtons, 'toggleView', state.get('view') || defaults.view);
        };
        scopeButtons.forEach((btn) => btn.addEventListener('click', () => {
            setSetting('scope', btn.dataset.toggleScope, defaults.scope);
        }));
        viewButtons.forEach((btn) => btn.addEventListener('click', () => {
            setSetting('view', btn.dataset.toggleView, defaults.view);
        }));
        state.subscribe((changed) => {
            if (changed.includes('scope') || changed.includes('view')) show();
        });
        show();
    }

    function wireThresholdSlider() {
        const slider = document.getElementById('aop-aop-threshold-slider');
        const label = document.getElementById('aop-aop-threshold-value');
        if (!slider) return;
        const defaultValue = slider.getAttribute('value') || slider.value;
        const show = () => {
            slider.value = state.get('min_jaccard') || defaultValue;
            if (label) label.textContent = slider.value;
        };
        // Live label while dragging; the plot reloads on release.
        slider.addEventListener('input', () => { if (label) label.textContent = slider.value; });
        slider.addEventListener('change', () => setSetting('min_jaccard', slider.value, defaultValue));
        state.subscribe((changed) => { if (changed.includes('min_jaccard')) show(); });
        show();
    }

    function wireSelect(id, key) {
        const select = document.getElementById(id);
        if (!select) return;
        const defaultValue = (Array.from(select.options).find((o) => o.defaultSelected) || select.options[0] || {}).value;
        const show = () => {
            const wanted = state.get(key);
            if (wanted && Array.from(select.options).some((o) => o.value === wanted)) select.value = wanted;
            else select.value = defaultValue;
        };
        select.addEventListener('change', () => setSetting(key, select.value, defaultValue));
        state.subscribe((changed) => { if (changed.includes(key)) show(); });
        show();
    }

    function init() {
        wireCoverageToggles();
        wireThresholdSlider();
        wireSelect('ontology-branch-select', 'branch');
        wireSelect('domain-lens-select', 'domain');
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
