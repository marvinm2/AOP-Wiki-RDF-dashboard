/**
 * Accessibility helpers for the plots (#181).
 *
 * - Labelled figures: every drawn plot is a focusable figure named by its
 *   card heading and described by the card's description.
 * - Keyboard: on a focused figure, Left/Right step through the categories it
 *   shows (snapshots on /trends, within any zoom or range), Home/End jump
 *   to the ends and Escape clears the
 *   hover. The values at the current step are read out through a live
 *   region. On /trends the linked crosshair follows (trend-links.js).
 * - Data-table fallback: "Show Raw Data" loads the plot's rows from
 *   /api/v2/data with the same version and settings as the figure, and
 *   reloads when those change. Replaces raw-data-tables.js.
 * - Reduced motion: motionMs() is 0 when the user asks for reduced motion
 *   (the network page's zoom animations use it; CSS handles the rest).
 *
 * categoriesOf(), visibleCategories(), valuesAt(), describeStep() and
 * tableHtml() are pure and
 * exported for node --test.
 */
(function (root) {
    'use strict';

    var TABLE_ROWS = 100;
    var MAX_SPOKEN = 8;

    // ------------------------------------------------------------ pure parts

    function toArray(v) {
        if (v == null) return [];
        return Array.isArray(v) ? v : Array.prototype.slice.call(v);
    }

    /** Category and value arrays of a (full) trace, whatever its orientation. */
    function axesOf(trace) {
        if (trace.type === 'pie') return { cats: toArray(trace.labels), vals: toArray(trace.values) };
        if (trace.orientation === 'h') return { cats: toArray(trace.y), vals: toArray(trace.x) };
        return { cats: toArray(trace.x), vals: toArray(trace.y) };
    }

    function shown(trace) {
        return trace.visible !== false && trace.visible !== 'legendonly';
    }

    /** The distinct categories across the shown traces, in first-seen order. */
    function categoriesOf(traces) {
        var seen = new Set();
        var out = [];
        traces.forEach(function (t) {
            if (!shown(t)) return;
            axesOf(t).cats.forEach(function (c) {
                var key = String(c);
                if (!seen.has(key)) { seen.add(key); out.push(key); }
            });
        });
        return out;
    }

    /**
     * The categories inside a zoomed or range-clamped axis ({type, range,
     * autorange} from _fullLayout), so the keyboard never reads values the
     * figure does not show. An autoranged axis keeps them all.
     */
    function visibleCategories(cats, axis) {
        if (!axis || axis.autorange !== false || !Array.isArray(axis.range)) return cats;
        var lo = Math.min.apply(null, axis.range.map(numeric));
        var hi = Math.max.apply(null, axis.range.map(numeric));
        if (axis.type === 'category') {
            return cats.filter(function (c, i) { return i >= lo && i <= hi; });
        }
        return cats.filter(function (c) {
            var v = axis.type === 'date' ? Date.parse(c) : Number(c);
            return isNaN(v) || (v >= lo && v <= hi);
        });

        function numeric(r) {
            return axis.type === 'date' && typeof r === 'string' ? Date.parse(r.replace(' ', 'T') + (/[zZ]|[+-]\d\d:?\d\d$/.test(r) ? '' : 'Z')) : Number(r);
        }
    }

    /** [{curveNumber, pointNumber, name, value}] of every shown trace at `category`. */
    function valuesAt(traces, category) {
        var out = [];
        traces.forEach(function (t, curveNumber) {
            if (!shown(t)) return;
            var a = axesOf(t);
            var pointNumber = a.cats.map(String).indexOf(String(category));
            if (pointNumber === -1) return;
            out.push({ curveNumber: curveNumber, pointNumber: pointNumber, name: t.name, value: a.vals[pointNumber] });
        });
        return out;
    }

    function fmt(v) {
        if (typeof v !== 'number') return v == null ? 'no value' : String(v);
        return Number.isInteger(v) ? String(v) : String(Math.round(v * 100) / 100);
    }

    /** The sentence read out for one step, e.g. "2026-01-01: AOPs 512, KEs 1490." */
    function describeStep(category, values) {
        if (!values.length) return category + ': no values.';
        var spoken = values.slice(0, MAX_SPOKEN).map(function (p) {
            return (p.name ? p.name + ' ' : '') + fmt(p.value);
        });
        var more = values.length > MAX_SPOKEN ? ', and ' + (values.length - MAX_SPOKEN) + ' more' : '';
        return category + ': ' + spoken.join(', ') + more + '.';
    }

    function escapeHtml(text) {
        return String(text).replace(/[&<>"']/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    /** HTML table for a /api/v2/data payload: the first `limit` rows, with a caption. */
    function tableHtml(payload, caption, limit) {
        var columns = payload.meta.columns.map(function (c) { return c.name; });
        var total = payload.meta.rows;
        var n = Math.min(total, limit);
        var html = '<div class="data-table-wrapper"><table class="raw-data-table">';
        if (caption) html += '<caption class="sr-only">' + escapeHtml(caption) + '</caption>';
        html += '<thead><tr>';
        columns.forEach(function (c) { html += '<th scope="col">' + escapeHtml(c) + '</th>'; });
        html += '</tr></thead><tbody>';
        for (var i = 0; i < n; i++) {
            html += '<tr>';
            columns.forEach(function (c) {
                var v = payload.data[c][i];
                html += '<td>' + escapeHtml(v == null ? '' : v) + '</td>';
            });
            html += '</tr>';
        }
        html += '</tbody></table></div>';
        if (total > n) {
            html += '<div class="data-table-truncated">Showing first ' + n + ' of ' + total
                + ' rows. Download CSV for full data.</div>';
        }
        return html;
    }

    function reducedMotion() {
        try {
            return !!(root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches);
        } catch (e) { return false; }
    }

    /** `ms`, or 0 when the user prefers reduced motion. */
    function motionMs(ms) {
        return reducedMotion() ? 0 : ms;
    }

    var api = {
        categoriesOf: categoriesOf, visibleCategories: visibleCategories, valuesAt: valuesAt, describeStep: describeStep,
        tableHtml: tableHtml, reducedMotion: reducedMotion, motionMs: motionMs,
    };

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
        return;
    }
    root.DashA11y = api;

    // ------------------------------------------------------------- the page

    var announcer = null;
    function announce(text) {
        if (!announcer) {
            announcer = document.createElement('div');
            announcer.id = 'plot-announcer';
            announcer.className = 'sr-only';
            announcer.setAttribute('aria-live', 'polite');
            document.body.appendChild(announcer);
        }
        announcer.textContent = text;
    }

    var uid = 0;
    function idFor(el, prefix) {
        if (!el.id) el.id = prefix + '-' + (++uid);
        return el.id;
    }

    function cardOf(container) {
        return container.closest('.plot-box') || container.closest('.plot-container');
    }

    function titleOf(container) {
        var card = cardOf(container);
        var h = card && card.querySelector('h2, h3, h4');
        return h ? h.textContent.trim() : container.dataset.plotName;
    }

    /** Name, describe and make focusable a drawn figure. */
    function label(container) {
        var card = cardOf(container);
        container.setAttribute('role', 'figure');
        container.setAttribute('tabindex', '0');
        container.setAttribute('aria-label', titleOf(container)
            + '. Use the left and right arrow keys to read the values.');
        var desc = card && card.querySelector('.plot-description');
        if (desc) container.setAttribute('aria-describedby', idFor(desc, 'plot-desc'));
    }

    var steps = new WeakMap();  // container -> current category index

    function onKey(event) {
        var container = event.currentTarget;
        var div = container.querySelector('.js-plotly-plot');
        if (!div || !div._fullData || !root.Plotly) return;
        var traces = div._fullData;
        var horizontal = traces.some(function (t) { return t.orientation === 'h'; });
        var axis = div._fullLayout && div._fullLayout[horizontal ? 'yaxis' : 'xaxis'];
        var cats = visibleCategories(categoriesOf(traces), axis);
        if (!cats.length) return;
        var i = steps.has(container) ? steps.get(container) : -1;
        switch (event.key) {
            case 'ArrowRight': i = Math.min(cats.length - 1, i + 1); break;
            case 'ArrowLeft': i = i === -1 ? cats.length - 1 : Math.max(0, i - 1); break;
            case 'Home': i = 0; break;
            case 'End': i = cats.length - 1; break;
            case 'Escape':
                steps.delete(container);
                root.Plotly.Fx.unhover(div);
                return;
            default: return;
        }
        event.preventDefault();
        steps.set(container, i);
        var values = valuesAt(traces, cats[i]);
        var hoverable = values.filter(function (p) { return traces[p.curveNumber].type !== 'pie'; });
        if (hoverable.length) {
            root.Plotly.Fx.hover(div, hoverable.map(function (p) {
                return { curveNumber: p.curveNumber, pointNumber: p.pointNumber };
            }));
        }
        announce(describeStep(cats[i], values));
    }

    document.addEventListener('plot:rendered', function (event) {
        var container = event.target;
        if (!container || !container.dataset || !container.dataset.plotName) return;
        if (container.dataset.a11y === 'self') return;  // the view labels itself (lifeline.js)
        label(container);
        steps.delete(container);
        if (!container.dataset.a11yKeys) {
            container.dataset.a11yKeys = '1';
            container.addEventListener('keydown', onKey);
        }
    });

    // ---------------------------------------------------- data-table fallback

    function tableUrl(plotName) {
        var query = root.plotLoader ? root.plotLoader.query(plotName) : '';
        return '/api/v2/data/' + plotName + query;
    }

    function containerFor(plotName) {
        return document.querySelector('[data-table-for="' + plotName + '"]');
    }

    function setOpen(container, open) {
        var content = container.querySelector('.data-table-content');
        var button = container.querySelector('.data-table-toggle');
        content.style.display = open ? 'block' : 'none';
        button.textContent = open ? 'Hide Raw Data' : 'Show Raw Data';
        button.setAttribute('aria-expanded', open ? 'true' : 'false');
    }

    async function fill(container, plotName) {
        var content = container.querySelector('.data-table-content');
        var url = tableUrl(plotName);
        content.dataset.url = url;
        content.innerHTML = '<div class="data-table-loading">Loading data...</div>';
        try {
            var response = await fetch(url);
            var payload = await response.json();
            if (content.dataset.url !== url) return;  // the selection moved on
            if (!response.ok || !payload.meta) {
                content.innerHTML = '<div class="data-table-empty">Data not available for this plot.</div>';
                content.dataset.url = '';
                return;
            }
            content.innerHTML = tableHtml(payload, 'Data behind: ' + titleOf(container), TABLE_ROWS);
        } catch (err) {
            if (content.dataset.url !== url) return;
            content.innerHTML = '<div class="data-table-empty">Failed to load data.</div>';
            content.dataset.url = '';
        }
    }

    /** The "Show Raw Data" button (templates/macros/data_table.html). */
    root.toggleDataTable = function (plotName) {
        var container = containerFor(plotName);
        if (!container) return;
        var content = container.querySelector('.data-table-content');
        var open = content.style.display === 'none' || content.style.display === '';
        setOpen(container, open);
        if (open && content.dataset.url !== tableUrl(plotName)) fill(container, plotName);
    };

    // An open table follows the state; a closed one refetches when reopened.
    if (root.DashState && root.PlotRuntime) {
        root.DashState.subscribe(function (changed) {
            document.querySelectorAll('[data-table-for]').forEach(function (container) {
                var plotName = container.dataset.tableFor;
                var deps = root.PlotRuntime.dependencies(plotName);
                if (!deps.some(function (k) { return changed.indexOf(k) !== -1; })) return;
                var content = container.querySelector('.data-table-content');
                if (content.style.display === 'block') fill(container, plotName);
            });
        });
    }
})(typeof window !== 'undefined' ? window : this);
