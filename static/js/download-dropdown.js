/**
 * Download Dropdowns for AOP-Wiki RDF Dashboard
 *
 * Opens the per-plot and page-level download menus on click rather than hover.
 * Hover-only menus leave their links inside a display:none container, so they
 * never enter the tab order and cannot be reached by keyboard; they are also
 * unreliable on touch, where :hover latching varies by browser.
 *
 * Click delegation is used so menus rendered after page load still work.
 */

(function () {
    'use strict';

    var OPEN_CLASS = 'is-open';

    function menuFor(toggle) {
        var wrapper = toggle.closest('.download-dropdown');
        return wrapper ? wrapper.querySelector('.dropdown-menu') : null;
    }

    function setState(toggle, open) {
        var menu = menuFor(toggle);
        if (!menu) return;
        menu.classList.toggle(OPEN_CLASS, open);
        toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    }

    function closeAll(except) {
        document.querySelectorAll('.download-dropdown .dropdown-menu.' + OPEN_CLASS)
            .forEach(function (menu) {
                var toggle = menu.parentElement.querySelector('.dropdown-toggle');
                if (toggle && toggle !== except) setState(toggle, false);
            });
    }

    /** Describe each toggle to assistive tech. Safe to call repeatedly. */
    function annotate(root) {
        (root || document).querySelectorAll('.download-dropdown .dropdown-toggle')
            .forEach(function (toggle) {
                if (!toggle.hasAttribute('type')) toggle.setAttribute('type', 'button');
                toggle.setAttribute('aria-haspopup', 'true');
                if (!toggle.hasAttribute('aria-expanded')) {
                    toggle.setAttribute('aria-expanded', 'false');
                }
            });
    }

    document.addEventListener('click', function (event) {
        var toggle = event.target.closest('.download-dropdown .dropdown-toggle');

        if (toggle) {
            event.preventDefault();
            var menu = menuFor(toggle);
            var willOpen = !!menu && !menu.classList.contains(OPEN_CLASS);
            closeAll(toggle);
            setState(toggle, willOpen);
            return;
        }

        // A click on a menu item starts a download; close the menu behind it.
        // Any other click outside dismisses whatever is open.
        closeAll(null);
    });

    document.addEventListener('keydown', function (event) {
        if (event.key !== 'Escape') return;
        var wrapper = event.target.closest('.download-dropdown');
        closeAll(null);
        if (wrapper) {
            var toggle = wrapper.querySelector('.dropdown-toggle');
            if (toggle) toggle.focus();
        }
    });

    // Tabbing out of an open dropdown dismisses it.
    document.addEventListener('focusin', function (event) {
        document.querySelectorAll('.download-dropdown').forEach(function (wrapper) {
            if (wrapper.contains(event.target)) return;
            var toggle = wrapper.querySelector('.dropdown-toggle');
            if (toggle) setState(toggle, false);
        });
    });

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () { annotate(); });
    } else {
        annotate();
    }

    window.annotateDownloadDropdowns = annotate;
})();
