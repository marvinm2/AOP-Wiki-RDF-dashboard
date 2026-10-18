/**
 * Page state kept in the URL (#181).
 *
 * The selected version, the trends range and every plot setting (coverage
 * scope/view, overlap threshold and tab, ontology branch, domain, and the
 * lifeline's entity type and status filter, the beeswarm's cohort and picked
 * AOP, the property heatmap's tab) live in the query string, so a view can be bookmarked, shared or
 * reloaded as it is. A value of
 * null means "the default": it is left out of the URL and the server applies
 * its own default.
 *
 * Every control writes here with DashState.set(); everything that depends on
 * the state (the plot runtime, banners, download links) subscribes. Query
 * parameters this module does not own are left untouched.
 *
 * The parsing and serialising functions are pure and exported for node --test.
 */
(function (root) {
    'use strict';

    /** The keys this module owns, in the order they appear in the URL. */
    var KEYS = ['version', 'start', 'end', 'scope', 'view', 'min_jaccard', 'branch', 'domain', 'entity', 'status', 'overlap', 'cohort', 'aop', 'props'];

    /** {key: value or null} for every owned key, from a query string. */
    function parse(search) {
        var params = new URLSearchParams(search || '');
        var state = {};
        KEYS.forEach(function (k) {
            var v = params.get(k);
            state[k] = v === null || v === '' ? null : v;
        });
        return state;
    }

    /**
     * The query string for `state`, keeping parameters this module does not
     * own from `search` (in their original order, ahead of the owned ones).
     */
    function serialize(state, search) {
        var params = new URLSearchParams(search || '');
        KEYS.forEach(function (k) { params.delete(k); });
        KEYS.forEach(function (k) {
            var v = state[k];
            if (v !== null && v !== undefined && v !== '') params.set(k, String(v));
        });
        var qs = params.toString();
        return qs ? '?' + qs : '';
    }

    /** The keys whose value differs between two states. */
    function changedKeys(before, after) {
        return KEYS.filter(function (k) { return (before[k] || null) !== (after[k] || null); });
    }

    /**
     * A state store bound to a location/history pair (window's by default; the
     * tests pass fakes).
     */
    function createState(loc, hist) {
        var state = parse(loc.search);
        var listeners = [];

        return {
            /** One value, or a copy of the whole state with no argument. */
            get: function (key) {
                if (key === undefined) return Object.assign({}, state);
                return state[key] === undefined ? null : state[key];
            },
            /**
             * Merge `patch` (unknown keys are ignored), rewrite the URL without
             * a navigation, and notify subscribers of the keys that changed.
             */
            set: function (patch) {
                var next = Object.assign({}, state);
                Object.keys(patch).forEach(function (k) {
                    if (KEYS.indexOf(k) === -1) return;
                    var v = patch[k];
                    next[k] = v === undefined || v === '' ? null : (v === null ? null : String(v));
                });
                var changed = changedKeys(state, next);
                if (!changed.length) return changed;
                state = next;
                hist.replaceState(hist.state, '', loc.pathname + serialize(state, loc.search) + (loc.hash || ''));
                listeners.slice().forEach(function (fn) { fn(changed, Object.assign({}, state)); });
                return changed;
            },
            /** Call fn(changedKeys, state) after every change. Returns an unsubscribe function. */
            subscribe: function (fn) {
                listeners.push(fn);
                return function () { listeners = listeners.filter(function (f) { return f !== fn; }); };
            }
        };
    }

    var api = { KEYS: KEYS, parse: parse, serialize: serialize, changedKeys: changedKeys, createState: createState };

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
    } else {
        root.DashState = Object.assign(createState(root.location, root.history), api);
    }
})(typeof window !== 'undefined' ? window : this);
