/**
 * Client-side figure export (#181).
 *
 * Every per-plot download menu gets "PNG (as shown)" and "SVG (as shown)"
 * once its figure is drawn. These save the figure as it is on the page, with
 * the current zoom, range, delta view and settings, using Plotly in the
 * browser. The server's PNG/SVG links stay as they are (a fixed-size render
 * of the full figure; whether to keep Kaleido is decided in #191).
 *
 * A D3 view that draws one SVG with its colours as attributes marks it
 * svg[data-export="svg"] (e.g. the organ-system UpSet, #184); that SVG is
 * saved as it is, or drawn onto a canvas for PNG.
 *
 * The file name says what the figure shows: the plot, then its version or
 * snapshot range, then its settings, e.g.
 *   latest_ontology_coverage_2026-01-01_scope-ke.png
 *   aop_entity_counts_absolute_2020-01-01_to_2026-01-01.svg
 *
 * filename() and plotOfDownload() are pure and exported for node --test.
 */
(function (root) {
    'use strict';

    var FORMATS = ['png', 'svg'];

    function safe(s) {
        return String(s).replace(/[^A-Za-z0-9._-]+/g, '-');
    }

    /**
     * File name (no extension) for `plotName` under `state`.
     * opts: {versioned, params: [setting keys], ranged: the figure is clamped
     * to the state's start/end}.
     */
    function filename(plotName, state, opts) {
        opts = opts || {};
        var parts = [plotName];
        if (opts.versioned && state.version) parts.push(state.version);
        if (!opts.versioned && opts.ranged && (state.start || state.end)) {
            parts.push((state.start || 'first') + '_to_' + (state.end || 'last'));
        }
        (opts.params || []).forEach(function (k) {
            if (state[k] !== null && state[k] !== undefined && state[k] !== '') parts.push(k + '-' + state[k]);
        });
        return parts.map(safe).join('_');
    }

    /**
     * The plot a per-plot download link exports, or null for other links:
     * /download/<plot>, /download/trend/<plot>, /download/latest/<suffix>
     * (= latest_<suffix>).
     */
    function plotOfDownload(href) {
        var path = String(href).split(/[?#]/)[0];
        var m = path.match(/^\/download\/(?:(trend|latest)\/)?([A-Za-z0-9_]+)$/);
        if (!m || m[2] === 'bulk' || (!m[1] && m[2] === 'latest') || (!m[1] && m[2] === 'trend')) return null;
        return m[1] === 'latest' ? 'latest_' + m[2] : m[2];
    }

    // ---------------------------------------------------------------- page

    function figureOf(container) {
        return container.querySelector('.js-plotly-plot');
    }

    /** The SVG of a drawn D3 view, or one a view builds on request (exportSvg, e.g. the overlap graph). */
    function svgOf(container) {
        if (typeof container.exportSvg === 'function') return container.exportSvg();
        return container.querySelector('svg[data-export="svg"]');
    }

    /** The container a menu item exports: the one on screen when a plot has two (tabs, #185). */
    function containerFor(name) {
        var all = Array.prototype.slice.call(document.querySelectorAll('.lazy-plot[data-plot-name="' + name + '"]'));
        return all.filter(function (el) { return el.offsetParent !== null; })[0] || all[0] || null;
    }

    /** Standalone SVG text of a drawn D3 view, without its pointer hit areas. */
    function svgText(svg) {
        var copy = svg.cloneNode(true);
        copy.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
        copy.removeAttribute('aria-hidden');
        Array.prototype.forEach.call(copy.querySelectorAll('.upset-hits, title'), function (n) { n.remove(); });
        var bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bg.setAttribute('width', '100%');
        bg.setAttribute('height', '100%');
        bg.setAttribute('fill', '#FFFFFF');
        copy.insertBefore(bg, copy.firstChild);
        return new XMLSerializer().serializeToString(copy);
    }

    function save(href, file) {
        var a = document.createElement('a');
        a.href = href;
        a.download = file;
        document.body.appendChild(a);
        a.click();
        a.remove();
    }

    function downloadSvg(svg, format, file) {
        var text = svgText(svg);
        var url = URL.createObjectURL(new Blob([text], { type: 'image/svg+xml' }));
        if (format === 'svg') {
            save(url, file + '.svg');
            setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
            return Promise.resolve(true);
        }
        return new Promise(function (resolve, reject) {
            var img = new Image();
            img.onload = function () {
                var scale = 2;
                var canvas = document.createElement('canvas');
                canvas.width = svg.width.baseVal.value * scale;
                canvas.height = svg.height.baseVal.value * scale;
                var ctx = canvas.getContext('2d');
                ctx.scale(scale, scale);
                ctx.drawImage(img, 0, 0);
                URL.revokeObjectURL(url);
                save(canvas.toDataURL('image/png'), file + '.png');
                resolve(true);
            };
            img.onerror = function (e) { URL.revokeObjectURL(url); reject(e); };
            img.src = url;
        });
    }

    /** The download menu in the container's card whose links export this plot. */
    function menuFor(container) {
        var box = container.closest('.plot-box') || container.parentElement;
        if (!box) return null;
        var name = container.dataset.plotName;
        var menus = Array.prototype.slice.call(box.querySelectorAll('.download-dropdown .dropdown-menu'));
        for (var i = 0; i < menus.length; i++) {
            var links = menus[i].querySelectorAll('a[href^="/download/"]');
            for (var j = 0; j < links.length; j++) {
                if (plotOfDownload(links[j].getAttribute('href')) === name) return menus[i];
            }
        }
        return null;
    }

    function addItems(container) {
        if (!figureOf(container) && typeof container.exportSvg !== 'function' && !svgOf(container)) return;  // nothing to save as drawn (e.g. the canvas lifeline)
        var menu = menuFor(container);
        var name = container.dataset.plotName;
        if (!menu || menu.querySelector('[data-export-format]')) return;
        FORMATS.forEach(function (format) {
            var item = document.createElement('button');
            item.type = 'button';
            item.className = 'dropdown-item dropdown-item--client';
            item.dataset.exportFormat = format;
            item.dataset.exportFor = name;
            item.title = 'The figure as drawn on this page, with its current zoom, range and settings';
            item.textContent = format.toUpperCase() + ' (as shown)';
            menu.appendChild(item);
        });
    }

    function stateNow() {
        return root.DashState ? root.DashState.get() : {};
    }

    /** Save the drawn figure of `container` as `format`. Resolves once handed to the browser. */
    function download(container, format) {
        var div = figureOf(container);
        var name = container.dataset.plotName;
        var Runtime = root.PlotRuntime;
        var svg = div ? null : svgOf(container);
        if (svg) {
            return downloadSvg(svg, format, filename(name, stateNow(), {
                versioned: Runtime ? Runtime.isVersioned(name) : false,
                params: Runtime ? Runtime.paramsOf(name) : [],
            }));
        }
        // A view drawn with another Plotly bundle (the full one, status-journey.js) keeps it on the div.
        var plotly = (div && div._dashPlotly) || root.Plotly;
        if (!div || !plotly) return Promise.resolve(false);
        var xaxis = (div.layout && div.layout.xaxis) || {};
        var file = filename(name, stateNow(), {
            versioned: Runtime ? Runtime.isVersioned(name) : false,
            params: Runtime ? Runtime.paramsOf(name) : [],
            ranged: xaxis.autorange === false,
        });
        return plotly.downloadImage(div, {
            format: format,
            filename: file,
            width: Math.max(div.clientWidth, 600),
            height: Math.max(div.clientHeight, 400),
            scale: format === 'png' ? 2 : 1,
        }).then(function () { return true; });
    }

    if (typeof document !== 'undefined') {
        document.addEventListener('plot:rendered', function (event) {
            if (event.target && event.target.dataset) addItems(event.target);
        });
        document.addEventListener('click', function (event) {
            var item = event.target && event.target.closest && event.target.closest('[data-export-format]');
            if (!item) return;
            var container = containerFor(item.dataset.exportFor);
            if (!container) return;
            download(container, item.dataset.exportFormat).catch(function (e) {
                console.error('Export of ' + item.dataset.exportFor + ' failed:', e);
            });
        });
    }

    var api = { filename: filename, plotOfDownload: plotOfDownload, download: download, menuFor: menuFor };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.DashExport = api;
})(typeof window !== 'undefined' ? window : this);
