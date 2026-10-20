/**
 * OECD status journey on /trends (#187).
 *
 * The journey is a server Plotly sankey, and sankey is not in the cartesian
 * bundle the pages load. Renderer `plotly-full` loads the full bundle (same
 * version, vendored with SRI) the first time the card comes into view and
 * draws the figure with it. The full bundle is captured through a one-shot
 * AMD `define`, so `window.Plotly` stays the cartesian one every other plot
 * was drawn with; the drawn div keeps its own library as `div._dashPlotly`
 * (viz/export.js uses it for the "as shown" downloads). The div is not a
 * `.plotly-graph-div`, so the /trends range selector leaves it alone; it has
 * no x axis to clamp.
 *
 * Keyboard: Left/Right move between the yearly steps, Up/Down between the
 * bands of a step, Home/End to the first and last step, Escape clears. The
 * focused band is drawn darker and read out in the live region. The band's
 * own SVG path is restyled: a Plotly restyle would fade the whole sankey in
 * again (about 500 ms), which reduced motion rules out.
 *
 * The pure helpers are exported for node --test.
 */
(function (root) {
    'use strict';

    const PLOT = 'oecd_status_journey';
    const FOCUS_ALPHA = 0.85;

    /**
     * One entry per band, ordered by step (the column it leaves), then by
     * the stages it joins: {i, step, stepLabel, text}. Node customdata is
     * [snapshot, AOPs]; link customdata the band's sentence.
     */
    function bandsOf(trace) {
        const link = trace.link || {};
        const node = trace.node || {};
        const snapshot = (n) => ((node.customdata || [])[n] || [])[0];
        const steps = Array.from(new Set(Array.from(link.source || [], snapshot))).sort();
        const bands = Array.from(link.source || [], (s, i) => ({
            i,
            step: steps.indexOf(snapshot(s)),
            stepLabel: `${snapshot(s)} to ${snapshot(link.target[i])}`,
            text: (link.customdata || [])[i] || '',
            order: [s, link.target[i]],
        }));
        bands.sort((a, b) => a.step - b.step || a.order[0] - b.order[0] || a.order[1] - b.order[1]);
        return bands;
    }

    /** The band index (into `bands`) a key moves to; undefined for other keys. */
    function journeyStep(key, current, bands) {
        if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(key)) return undefined;
        if (!bands.length) return undefined;
        const firstOf = (step) => bands.findIndex((b) => b.step === step);
        const last = bands[bands.length - 1].step;
        if (current == null) return key === 'ArrowLeft' || key === 'End' ? firstOf(last) : 0;
        const step = bands[current].step;
        if (key === 'ArrowUp') return bands[current - 1] && bands[current - 1].step === step ? current - 1 : current;
        if (key === 'ArrowDown') return bands[current + 1] && bands[current + 1].step === step ? current + 1 : current;
        if (key === 'Home') return firstOf(0);
        if (key === 'End') return firstOf(last);
        const to = key === 'ArrowLeft' ? Math.max(0, step - 1) : Math.min(last, step + 1);
        return to === step ? current : firstOf(to);
    }

    /** The sentence read out for a band. */
    function describeBand(bands, k) {
        const band = bands[k];
        const inStep = bands.filter((b) => b.step === band.step);
        return `${band.stepLabel}, band ${inStep.indexOf(band) + 1} of ${inStep.length}: ${band.text}.`;
    }

    /**
     * Fill opacity per drawn band path ({pointNumber, alpha} from its d3
     * datum): FOCUS_ALPHA for link `i`, its own alpha for the rest.
     */
    function focusOpacities(paths, i) {
        return paths.map((p) => String(p.pointNumber === i ? FOCUS_ALPHA : p.alpha));
    }

    /**
     * Load a UMD bundle without letting it set a global: a one-shot AMD
     * define takes the factory and puts the previous `define` back.
     */
    function captureUmd(win, insert) {
        return new Promise((resolve, reject) => {
            const had = Object.prototype.hasOwnProperty.call(win, 'define');
            const previous = win.define;
            let lib = null;
            const restore = () => {
                if (win.define !== capture) return;
                if (had) win.define = previous; else delete win.define;
            };
            function capture(deps, factory) {
                restore();
                lib = (typeof deps === 'function' ? deps : factory)();
            }
            capture.amd = {};
            win.define = capture;
            insert(() => { restore(); if (lib) resolve(lib); else reject(new Error('no library')); },
                () => { restore(); reject(new Error('script did not load')); });
        });
    }

    const api = { PLOT, bandsOf, journeyStep, describeBand, focusOpacities, captureUmd };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
        return;
    }
    root.DashStatusJourney = api;

    // ------------------------------------------------------------- the page

    let fullLoad = null;
    function loadFullPlotly(element) {
        if (!fullLoad) {
            fullLoad = captureUmd(root, (onload, onerror) => {
                const s = document.createElement('script');
                s.src = element.dataset.libSrc;
                if (element.dataset.libIntegrity) {
                    s.integrity = element.dataset.libIntegrity;
                    s.crossOrigin = 'anonymous';
                }
                s.onload = onload;
                s.onerror = onerror;
                document.head.appendChild(s);
            }).catch((e) => { fullLoad = null; throw e; });
        }
        return fullLoad;
    }

    let setup = null;
    function plotlySetup() {
        if (!setup) {
            setup = fetch('/api/v2/plotly').then((r) => {
                if (!r.ok) throw new Error(`/api/v2/plotly: ${r.status}`);
                return r.json();
            }).catch((e) => { setup = null; throw e; });
        }
        return setup;
    }

    const views = new WeakMap();  // container -> {div, lib, bands, base, focus}

    async function mount(element, payload) {
        let lib;
        try {
            lib = await loadFullPlotly(element);
        } catch (e) {
            element.innerHTML = '<p class="plot-error">The journey needs a script that did not load. '
                + 'The CSV and "Show Raw Data" hold the same flows.</p>';
            return;
        }
        const { template, config } = await plotlySetup();
        const figure = payload.figure;
        const layout = Object.assign({}, figure.layout, { template });
        let view = views.get(element);
        if (!view || !view.div.isConnected) {
            element.innerHTML = '';
            const div = document.createElement('div');
            div.className = 'plotly-full-graph';
            element.appendChild(div);
            await lib.newPlot(div, figure.data, layout, config);
            div._dashPlotly = lib;
            view = { div, lib };
            views.set(element, view);
            // A redraw (resize) makes new paths: keep the focus on them.
            div.on('plotly_afterplot', () => paint(view));
        } else {
            await lib.react(view.div, figure.data, layout, config);
        }
        // _fullData: the figure may send its arrays base64-packed.
        const trace = view.div._fullData[0];
        view.bands = bandsOf(trace);
        view.focus = null;
        wire(element);
    }

    function destroy(element) {
        const view = views.get(element);
        if (!view) return;
        try { view.lib.purge(view.div); } catch (e) { /* gone */ }
        views.delete(element);
    }

    function announce(element, text) {
        const live = element.closest('.plot-box').querySelector('.journey-live');
        if (live) live.textContent = text;
    }

    function paint(view) {
        const paths = Array.from(view.div.querySelectorAll('.sankey-link'));
        const data = paths.map((p) => ({
            pointNumber: p.__data__ ? p.__data__.pointNumber : -1,
            alpha: p.__data__ && p.__data__.tinyColorAlpha != null ? p.__data__.tinyColorAlpha : 1,
        }));
        const want = focusOpacities(data, view.focus == null ? -1 : view.bands[view.focus].i);
        paths.forEach((p, j) => { p.style.fillOpacity = want[j]; });
    }

    function show(element, view, k) {
        view.focus = k;
        paint(view);
    }

    function onKey(event) {
        const element = event.currentTarget;
        const view = views.get(element);
        if (!view) return;
        if (event.key === 'Escape') {
            if (view.focus == null) return;
            show(element, view, null);
            announce(element, 'Band cleared.');
            return;
        }
        const next = journeyStep(event.key, view.focus, view.bands);
        if (next === undefined) return;
        event.preventDefault();
        show(element, view, next);
        announce(element, describeBand(view.bands, next));
    }

    function wire(element) {
        element.setAttribute('role', 'figure');
        element.tabIndex = 0;
        element.setAttribute('aria-label', 'Sankey of AOPs moving between OECD stages, one column per year. '
            + 'Use left and right to move between years, up and down between the bands of a year, '
            + 'or "Show Raw Data" below for the table.');
        if (!element.dataset.journeyKeys) {
            element.dataset.journeyKeys = '1';
            element.addEventListener('keydown', onKey);
        }
    }

    const renderer = {
        endpoint(plotName, qs) { return `/api/v2/figure/${plotName}${qs}`; },
        shared: true,
        mount,
        update: mount,
        destroy,
    };
    if (root.PlotRuntime) root.PlotRuntime.registerRenderer('plotly-full', renderer);
})(typeof window !== 'undefined' ? window : globalThis);
