/**
 * KE reuse rank race (#187). Renderer `plotly-race` for the plot runtime.
 *
 * - Data: /api/v2/data/ke_reuse_race, one row per snapshot for every KE that
 *   is in the top 15 of at least one snapshot (ke_race_rows in
 *   plots/trends_plots.py): its AOP count and rank.
 * - The bars show the snapshot at the end of the /trends range (the version
 *   scrubber's right handle, which Play moves forward), like the beeswarm.
 *   Each bar carries its KE IRI as its Plotly id, so when the snapshot
 *   changes the bars grow and slide to their new rank; a KE that leaves the
 *   top 15 slides out below the chart and one that enters slides in from
 *   there. With prefers-reduced-motion the bars jump instead. Every bar and
 *   label keeps its text and a non-zero length in every frame: Plotly's
 *   transitions (redraw off) neither add text that was empty nor interpolate
 *   from an empty bar.
 * - The x axis is fixed across the snapshots, so the bars grow as the KEs
 *   are reused in more AOPs.
 * - Picking a KE (click its bar, or Up/Down on the focused chart) marks it in
 *   magenta in every snapshot and names its rank and AOPs; Enter opens its
 *   identifiers.org IRI (which resolves to AOP-Wiki), Escape clears.
 *
 * The pure helpers are exported for node --test.
 */
(function (root) {
    'use strict';

    const TOP = 15;
    // Where a KE outside the top waits: far enough below the chart to be off
    // the canvas, labels included, so it can carry its label and slide in.
    const OUT = TOP + 4;
    // Plotly draws a zero-length bar as an empty path, and animating from one
    // makes the bar fly in sideways; a KE with no AOPs gets a sliver instead.
    const MIN_BAR = 0.5;
    const TRANSITION_MS = 600;
    const LABEL_WIDTH = 42;

    // ------------------------------------------------------------ pure parts

    /** The snapshots in the data, oldest first. */
    function versionsOf(data) {
        return Array.from(new Set(data.Version || [])).sort();
    }

    /** The snapshot to show for the range end `end` (null = the newest). */
    function shownVersion(versions, end) {
        if (!versions.length) return null;
        if (!end) return versions[versions.length - 1];
        const earlier = versions.filter((v) => v <= end);
        return earlier.length ? earlier[earlier.length - 1] : versions[0];
    }

    function label(id, title) {
        let t = String(title || '').split(/\s+/).filter(Boolean).join(' ');
        if (t.length > LABEL_WIDTH) t = `${t.slice(0, LABEL_WIDTH - 1).trimEnd()}…`;
        return t ? `${id} · ${t}` : id;
    }

    /**
     * The bars of `version`: one per KE in the data, in KE order so ids keep
     * their slot, at its rank (OUT when outside the top or not ranked).
     */
    function barsAt(data, version) {
        const out = [];
        for (let i = 0; i < data.Version.length; i++) {
            if (data.Version[i] !== version) continue;
            const rank = data.Rank[i];
            const shown = rank !== null && rank !== undefined && rank <= TOP;
            out.push({
                ke: data.KE[i], id: data['KE ID'][i], title: String(data.Title[i] || '').split(/\s+/).filter(Boolean).join(' '),
                aops: data.AOPs[i], rank: shown ? rank : null, y: shown ? rank : OUT,
            });
        }
        return out;
    }

    /** The KE at the next rank up or down from `current` (a KE IRI or null). */
    function stepPick(bars, current, key) {
        const ranked = bars.filter((b) => b.rank !== null).sort((a, b) => a.rank - b.rank);
        if (!ranked.length) return undefined;
        if (key === 'Escape') return null;
        const at = ranked.findIndex((b) => b.ke === current);
        if (key === 'Home') return ranked[0].ke;
        if (key === 'End') return ranked[ranked.length - 1].ke;
        if (key === 'ArrowDown') return ranked[at < 0 ? 0 : Math.min(ranked.length - 1, at + 1)].ke;
        if (key === 'ArrowUp') return ranked[at < 0 ? ranked.length - 1 : Math.max(0, at - 1)].ke;
        return undefined;
    }

    /** The sentence for a picked KE in `version`, with its rank in the snapshot before. */
    function describePick(data, versions, version, ke) {
        const bars = barsAt(data, version);
        const bar = bars.find((b) => b.ke === ke);
        if (!bar) return '';
        const name = `${bar.id}${bar.title ? `, ${bar.title}` : ''}`;
        const before = versions[versions.indexOf(version) - 1];
        const prev = before ? barsAt(data, before).find((b) => b.ke === ke) : null;
        let change = '';
        if (prev) {
            if (prev.rank === null && bar.rank !== null) change = `, new in the top ${TOP}`;
            else if (prev.rank !== null && bar.rank === null) change = `, out of the top ${TOP} (was ${prev.rank})`;
            else if (prev.rank !== null && prev.rank !== bar.rank) change = `, was ${prev.rank} in ${before}`;
        }
        const where = bar.rank === null ? `not in the top ${TOP}` : `rank ${bar.rank} of ${TOP}`;
        return `${name}: ${where} in ${version}, in ${bar.aops} AOP${bar.aops === 1 ? '' : 's'}${change}.`;
    }

    const api = { TOP, OUT, MIN_BAR, versionsOf, shownVersion, label, barsAt, stepPick, describePick };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
        return;
    }
    root.DashRace = api;

    // ------------------------------------------------------------- the page

    const views = new WeakMap();
    const reducedMotion = () => !!(root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches);
    const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    let tokens = null;
    function getTokens() {
        if (!tokens) {
            tokens = fetch('/static/data/tokens.json').then((r) => r.json())
                .catch(() => ({ color: { primary: '#29235C', magenta: '#E6007E', blue: '#307BBF' } }));
        }
        return tokens;
    }

    async function mount(element, payload) {
        destroy(element);
        const Plotly = root.Plotly;
        if (!Plotly) {
            element.innerHTML = '<p class="race-note">The race needs Plotly, which did not load. '
                + 'Use "Show Raw Data" below or the CSV download.</p>';
            return;
        }
        const data = payload.data;
        const versions = versionsOf(data);
        const colour = (await getTokens()).color;
        const state = root.DashState;
        const maxAops = Math.max(1, ...data.AOPs);

        element.dataset.a11y = 'self';
        element.setAttribute('role', 'figure');
        element.innerHTML = `
            <p class="race-caption"></p>
            <div class="race-chart" tabindex="0" role="group" aria-roledescription="bar chart race">
                <div class="race-plot"></div>
            </div>
            <p class="race-readout"></p>
            <p class="sr-only race-live" aria-live="polite"></p>`;
        const q = (s) => element.querySelector(s);
        const chart = q('.race-chart');
        const div = q('.race-plot');
        let picked = null;

        const version = () => shownVersion(versions, state ? state.get('end') : null);

        function traces() {
            const bars = barsAt(data, version());
            const colours = bars.map((b) => (b.ke === picked ? colour.magenta : (colour.blue || colour.primary)));
            return [{
                type: 'bar', orientation: 'h', ids: bars.map((b) => b.ke),
                x: bars.map((b) => Math.max(b.aops, MIN_BAR)), y: bars.map((b) => b.y),
                text: bars.map((b) => String(b.aops)), textposition: 'outside',
                marker: { color: colours },
                customdata: bars.map((b) => [b.ke, b.id, b.title, b.rank]),
                hovertemplate: '<b>%{customdata[1]}</b> %{customdata[2]}<br>Rank %{customdata[3]}: %{text} AOPs<extra></extra>',
            }, {
                // The KE names, left of the axis, sliding with their bars.
                type: 'scatter', mode: 'text', ids: bars.map((b) => b.ke), hoverinfo: 'skip',
                x: bars.map(() => 0), y: bars.map((b) => b.y), cliponaxis: false,
                text: bars.map((b) => label(b.id, b.title)),
                textposition: 'middle left',
                textfont: { color: bars.map((b) => (b.ke === picked ? colour.magenta : colour.primary)), size: 12 },
            }];
        }

        function layout() {
            return {
                height: 600, margin: { l: 330, r: 40, t: 20, b: 60 }, showlegend: false,
                // Overlay: the KEs waiting at OUT share a y, and Plotly would
                // stack them end to end, so they would slide in shifted.
                hovermode: 'closest', bargap: 0.25, barmode: 'overlay',
                xaxis: { title: { text: 'Number of AOPs' }, range: [0, maxAops * 1.12], fixedrange: true },
                yaxis: { range: [TOP + 0.6, 0.4], autorange: false, visible: false, fixedrange: true },
                annotations: [{
                    text: version(), x: 1, y: 0.02, xref: 'paper', yref: 'paper', xanchor: 'right', yanchor: 'bottom',
                    showarrow: false, font: { size: 28, color: colour.primary }, opacity: 0.35,
                }],
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            };
        }

        const config = { displaylogo: false, displayModeBar: false, responsive: true };

        async function draw(animate) {
            if (!div.classList.contains('js-plotly-plot')) {
                await Plotly.newPlot(div, traces(), layout(), config);
                div.on('plotly_click', (ev) => {
                    const p = ev.points && ev.points[0];
                    if (p && p.customdata) pick(p.customdata[0]);
                });
            } else if (animate && !reducedMotion()) {
                // The snapshot name is not animated; set it before the bars move.
                await Plotly.relayout(div, { 'annotations[0].text': version() });
                await Plotly.animate(div, { data: traces(), traces: [0, 1] }, {
                    transition: { duration: TRANSITION_MS, easing: 'cubic-in-out' },
                    frame: { duration: TRANSITION_MS, redraw: false },
                    mode: 'immediate',
                });
            } else {
                await Plotly.react(div, traces(), layout(), config);
            }
            describe();
        }

        function describe() {
            const v = version();
            q('.race-caption').textContent = `The ${TOP} Key Events in the most AOPs in the ${v} snapshot.`
                + (state && state.get('end') ? '' : ' Move the end of the range on the timeline, or press Play, to watch the ranks change.');
            const name = `Bar chart race of the ${TOP} most reused Key Events, ${v} snapshot`;
            element.setAttribute('aria-label', name);
            chart.setAttribute('aria-label', `${name}. Up and down arrows pick a Key Event, Enter opens it, Escape clears the pick.`);
            const readout = q('.race-readout');
            if (!picked) {
                readout.textContent = 'Click a bar or use the arrow keys on the chart to follow one Key Event through the snapshots.';
            } else {
                readout.innerHTML = `<a href="${esc(picked)}" target="_blank" rel="noopener">${esc(describePick(data, versions, v, picked))}</a>`;
            }
        }

        function pick(ke) {
            picked = ke;
            draw(false).then(() => {
                q('.race-live').textContent = ke ? describePick(data, versions, version(), ke) : 'Pick cleared.';
            });
        }

        chart.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                if (picked) { event.preventDefault(); root.open(picked, '_blank', 'noopener'); }
                return;
            }
            const next = stepPick(barsAt(data, version()), picked, event.key);
            if (next === undefined) return;
            event.preventDefault();
            pick(next);
        });

        const unsubscribe = state ? state.subscribe((changed) => {
            if (changed.includes('end')) {
                draw(true).then(() => {
                    if (picked) q('.race-live').textContent = describePick(data, versions, version(), picked);
                });
            }
        }) : () => {};

        await draw(false);
        views.set(element, {
            destroy() {
                unsubscribe();
                if (div.classList.contains('js-plotly-plot')) Plotly.purge(div);
            },
        });
    }

    function destroy(element) {
        const view = views.get(element);
        if (view) view.destroy();
        views.delete(element);
    }

    const renderer = {
        endpoint(plotName, qs) { return `/api/v2/data/${plotName}${qs}`; },
        mount,
        update: mount,
        destroy,
    };
    if (root.PlotRuntime) root.PlotRuntime.registerRenderer('plotly-race', renderer);
    root.DashRace.renderer = renderer;
})(typeof window !== 'undefined' ? window : this);
