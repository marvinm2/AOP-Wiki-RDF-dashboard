/**
 * The version scrubber (#182): every snapshot on one time axis.
 *
 * - /snapshot (data-mode="single"): one handle picks the version and replaces
 *   the version <select>. Click the axis or drag the handle; it snaps to the
 *   nearest snapshot.
 * - /trends (data-mode="range"): a brush picks the snapshot range and replaces
 *   the two range <select>s. It snaps to snapshots; a click without a drag
 *   resets to all snapshots.
 * - Both write to the page state (static/js/state.js), so the URL, the plots,
 *   the downloads and the methodology queries follow; and both follow the
 *   state when something else changes it (Reset, back/forward, the selects).
 * - Play steps through the snapshots: the version on /snapshot, the end of the
 *   range on /trends. Any other input pauses it.
 * - Keyboard: each handle is a slider (arrows step one snapshot, Page Up/Down
 *   a year, Home/End the ends). No animated transitions anywhere, so
 *   prefers-reduced-motion needs nothing extra.
 * - Shaded periods mark changes in the data itself (window.TIMELINE_EVENTS,
 *   from plots/registry.py; methodology note "version_timeline").
 *
 * D3 is used for the scale, axis, brush and drag only. The pure helpers
 * (nearestIndex, stepIndex, normaliseRange, playNext) are exported for
 * node --test.
 */
(function (root) {
    'use strict';

    // ------------------------------------------------------------ pure parts

    /** Index of the time in `times` (ascending ms) closest to `t`. */
    function nearestIndex(times, t) {
        let best = 0;
        for (let i = 1; i < times.length; i++) {
            if (Math.abs(times[i] - t) < Math.abs(times[best] - t)) best = i;
        }
        return best;
    }

    /** The index a key moves `i` to among `n` snapshots, or null for other keys. */
    function stepIndex(i, key, n) {
        const year = 4;  // quarterly snapshots
        const to = {
            ArrowLeft: i - 1, ArrowDown: i - 1, ArrowRight: i + 1, ArrowUp: i + 1,
            PageDown: i - year, PageUp: i + year, Home: 0, End: n - 1,
        }[key];
        if (to === undefined) return null;
        return Math.max(0, Math.min(n - 1, to));
    }

    /**
     * The state patch for a range of version indices: {start, end}, or nulls
     * for the whole range (the default, left out of the URL).
     */
    function normaliseRange(i0, i1, versions) {
        const lo = Math.min(i0, i1);
        const hi = Math.max(i0, i1);
        if (lo === 0 && hi === versions.length - 1) return { start: null, end: null };
        return { start: versions[lo], end: versions[hi] };
    }

    /**
     * The next step of Play, or null when it is done. Single: the next
     * version, restarting from the first when at the last. Range: the end
     * moves forward from the start; at the last snapshot it starts over.
     * `pos` is {i} (single) or {i0, i1} (range).
     */
    function playNext(mode, pos, n, first) {
        if (mode === 'single') {
            if (first) return { i: pos.i >= n - 1 ? 0 : pos.i + 1 };
            return pos.i >= n - 1 ? null : { i: pos.i + 1 };
        }
        if (first) {
            const start = pos.i1 >= n - 1 && pos.i0 >= n - 2 ? 0 : pos.i0;
            return { i0: start, i1: pos.i1 >= n - 1 ? start : pos.i1 + 1 };
        }
        return pos.i1 >= n - 1 ? null : { i0: pos.i0, i1: pos.i1 + 1 };
    }

    const api = { nearestIndex, stepIndex, normaliseRange, playNext };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
        return;
    }
    root.DashScrubber = api;

    // ------------------------------------------------------------- the view

    const PLAY_MS = 2000;
    const KEY_COMMIT_MS = 250;
    const HEIGHT = 58;
    const TRACK_Y = 22;
    const MARGIN = { left: 14, right: 14 };

    const parse = (v) => Date.parse(`${v}T00:00:00Z`);

    function mount(el, versions, events, state) {
        const d3 = root.d3;
        const mode = el.dataset.mode === 'range' ? 'range' : 'single';
        const n = versions.length;
        const times = versions.map(parse);
        const latest = versions[n - 1];

        el.innerHTML = '';
        el.classList.add('version-scrubber--ready');
        const bar = d3.select(el).append('div').attr('class', 'scrubber-bar');
        const play = bar.append('button')
            .attr('type', 'button').attr('class', 'scrubber-play')
            .attr('aria-pressed', 'false')
            .attr('aria-label', mode === 'single' ? 'Play through the snapshots' : 'Play the range forward')
            .text('Play');
        // Not an image: it holds the sliders. Its decoration is hidden from
        // assistive technology; the legend and the sliders carry the content.
        const svg = bar.append('svg').attr('class', 'scrubber-svg').attr('height', HEIGHT)
            .attr('role', 'presentation');
        const readout = bar.append('span').attr('class', 'scrubber-readout').attr('aria-hidden', 'true');

        const legend = d3.select(el).append('ul').attr('class', 'scrubber-legend');
        events.forEach((ev) => {
            const li = legend.append('li');
            li.append('span').attr('class', `scrubber-swatch scrubber-band--${ev.id}`).attr('aria-hidden', 'true');
            li.append('span').text(`${ev.label} (${ev.start} to ${ev.end})`);
            li.append('span').attr('class', 'sr-only').text(`: ${ev.detail}`);
        });

        const gBands = svg.append('g').attr('aria-hidden', 'true');
        const gAxis = svg.append('g').attr('class', 'scrubber-axis').attr('aria-hidden', 'true');
        const gMarks = svg.append('g').attr('class', 'scrubber-marks').attr('aria-hidden', 'true');
        const gBrush = svg.append('g').attr('class', 'scrubber-brush');
        const gHandles = svg.append('g');

        const x = d3.scaleUtc();
        let width = 0;

        // ---- position, from the state
        function position() {
            if (mode === 'single') {
                const v = state.get('version') || latest;
                const i = versions.indexOf(v);
                return { i: i === -1 ? n - 1 : i };
            }
            const s = versions.indexOf(state.get('start'));
            const e = versions.indexOf(state.get('end'));
            return { i0: s === -1 ? 0 : s, i1: e === -1 ? n - 1 : e };
        }
        let shown = position();  // what is drawn (can run ahead of the state while dragging)

        function commit(pos) {
            if (mode === 'single') {
                state.set({ version: versions[pos.i] === latest ? null : versions[pos.i] });
            } else {
                state.set(normaliseRange(pos.i0, pos.i1, versions));
            }
        }

        // ---- handles (sliders)
        const handleSpecs = mode === 'single'
            ? [{ key: 'i', label: 'Snapshot version' }]
            : [{ key: 'i0', label: 'Range start' }, { key: 'i1', label: 'Range end' }];
        const handles = gHandles.selectAll('g').data(handleSpecs).join('g')
            .attr('class', 'scrubber-handle')
            .attr('role', 'slider')
            .attr('tabindex', 0)
            .attr('aria-label', (d) => d.label)
            .attr('aria-valuemin', 0)
            .attr('aria-valuemax', n - 1);
        handles.append('line').attr('y1', TRACK_Y - 12).attr('y2', TRACK_Y + 12);
        handles.append('circle').attr('cy', TRACK_Y).attr('r', mode === 'single' ? 7 : 6);

        function valueText(i) {
            return versions[i] === latest ? `${versions[i]} (current)` : versions[i];
        }

        function drawPosition() {
            handles
                .attr('transform', (d) => `translate(${x(times[shown[d.key]])},0)`)
                .attr('aria-valuenow', (d) => shown[d.key])
                .attr('aria-valuetext', (d) => valueText(shown[d.key]));
            if (mode === 'single') {
                gMarks.selectAll('circle').classed('is-selected', (d, i) => i === shown.i);
                readout.text(valueText(shown.i));
            } else {
                gMarks.selectAll('circle').classed('is-selected', (d, i) => i >= shown.i0 && i <= shown.i1);
                readout.text(shown.i0 === 0 && shown.i1 === n - 1
                    ? 'All snapshots' : `${versions[shown.i0]} → ${versions[shown.i1]}`);
                // "All snapshots" is drawn as no selection, so a drag anywhere
                // starts a new range instead of moving one that fills the axis.
                const full = shown.i0 === 0 && shown.i1 === n - 1;
                brushing = true;
                gBrush.call(brush.move, full ? null : [x(times[shown.i0]) - 4, x(times[shown.i1]) + 4]);
                brushing = false;
            }
        }

        // ---- keyboard
        let keyTimer = null;
        handles.on('keydown', (event, d) => {
            if (event.key === ' ' || event.key === 'Enter') return;
            const to = stepIndex(shown[d.key], event.key, n);
            if (to === null) return;
            event.preventDefault();
            stop();
            shown = Object.assign({}, shown, { [d.key]: to });
            if (mode === 'range' && shown.i0 > shown.i1) shown = { i0: to, i1: to };
            drawPosition();
            clearTimeout(keyTimer);
            keyTimer = setTimeout(() => commit(shown), KEY_COMMIT_MS);
        });

        // ---- pointer: single mode clicks/drags, range mode brushes
        const brush = d3.brushX();
        let brushing = false;
        if (mode === 'single') {
            const overlay = svg.insert('rect', () => gHandles.node()).attr('class', 'scrubber-overlay')
                .attr('y', 0).attr('height', HEIGHT - 14);
            overlay.on('click', (event) => {
                stop();
                shown = { i: nearestIndex(times, x.invert(d3.pointer(event)[0]).getTime()) };
                drawPosition();
                commit(shown);
            });
        } else {
            brush.on('end', (event) => {
                if (brushing || !event.sourceEvent) return;
                stop();
                if (!event.selection) {
                    shown = { i0: 0, i1: n - 1 };
                } else {
                    const [a, b] = event.selection.map((px) => nearestIndex(times, x.invert(px).getTime()));
                    shown = { i0: a, i1: b };
                }
                drawPosition();
                commit(shown);
            });
        }

        // Dragging a handle moves its snapshot; the state follows on release.
        handles.call(d3.drag()
            .on('start', () => stop())
            .on('drag', (event, d) => {
                const to = nearestIndex(times, x.invert(event.x).getTime());
                shown = Object.assign({}, shown, { [d.key]: to });
                if (mode === 'range' && shown.i0 > shown.i1) shown = { i0: to, i1: to };
                drawPosition();
            })
            .on('end', () => commit(shown)));

        // ---- play
        let timer = null;
        function stop() {
            if (!timer) return;
            clearInterval(timer);
            timer = null;
            play.attr('aria-pressed', 'false').text('Play');
        }
        function tick(first) {
            const next = playNext(mode, shown, n, first);
            if (!next) { stop(); return; }
            shown = next;
            drawPosition();
            commit(shown);
        }
        play.on('click', () => {
            if (timer) { stop(); return; }
            play.attr('aria-pressed', 'true').text('Pause');
            tick(true);
            timer = setInterval(() => tick(false), PLAY_MS);
        });

        // ---- layout
        function layout() {
            const w = svg.node().getBoundingClientRect().width;
            if (!w || w === width) return;
            width = w;
            const pad = 45 * 24 * 3600 * 1000;
            x.domain([times[0] - pad, times[n - 1] + pad]).range([MARGIN.left, width - MARGIN.right]);

            // Clipped to the axis: a period can start before the first snapshot.
            const [x0, x1] = [MARGIN.left, width - MARGIN.right];
            const clip = (t) => Math.max(x0, Math.min(x1, x(t)));
            const shownEvents = events.filter((ev) => x(parse(ev.end)) > x0 && x(parse(ev.start)) < x1);
            gBands.selectAll('rect').data(shownEvents).join('rect')
                .attr('class', (ev) => `scrubber-band scrubber-band--${ev.id}`)
                .attr('x', (ev) => clip(parse(ev.start)))
                .attr('width', (ev) => Math.max(2, clip(parse(ev.end)) - clip(parse(ev.start))))
                .attr('y', TRACK_Y - 14).attr('height', 28)
                .selectAll('title').data((ev) => [ev]).join('title').text((ev) => `${ev.label}: ${ev.detail}`);

            gAxis.attr('transform', `translate(0,${TRACK_Y + 14})`)
                .call(d3.axisBottom(x).ticks(d3.utcYear.every(width < 520 ? 2 : 1)).tickFormat(d3.utcFormat('%Y')).tickSizeOuter(0));

            gMarks.selectAll('line.scrubber-track').data([0]).join('line').attr('class', 'scrubber-track')
                .attr('x1', MARGIN.left).attr('x2', width - MARGIN.right).attr('y1', TRACK_Y).attr('y2', TRACK_Y);
            gMarks.selectAll('circle').data(times).join('circle')
                .attr('cx', (t) => x(t)).attr('cy', TRACK_Y).attr('r', 3)
                .selectAll('title').data((t, i) => [versions[i]]).join('title').text((v) => v);

            svg.select('.scrubber-overlay').attr('x', MARGIN.left).attr('width', width - MARGIN.left - MARGIN.right);
            if (mode === 'range') {
                brush.extent([[MARGIN.left, TRACK_Y - 14], [width - MARGIN.right, TRACK_Y + 14]]);
                gBrush.call(brush);
            }
            drawPosition();
        }
        layout();
        if (root.ResizeObserver) new ResizeObserver(layout).observe(svg.node());

        // On /trends the scrubber sticks with the section nav; section jumps
        // must clear the whole block, whose height depends on wrapping.
        const sticky = el.closest('.sticky-controls');
        if (sticky && root.ResizeObserver) {
            new ResizeObserver(() => document.documentElement.style.setProperty(
                '--sticky-controls-height', `${Math.ceil(sticky.getBoundingClientRect().height) + 10}px`,
            )).observe(sticky);
        }

        // ---- follow the state when something else changes it
        const keys = mode === 'single' ? ['version'] : ['start', 'end'];
        state.subscribe((changed) => {
            if (!keys.some((k) => changed.includes(k))) return;
            const pos = position();
            if (JSON.stringify(pos) === JSON.stringify(shown)) return;
            if (!timer) clearTimeout(keyTimer);
            shown = pos;
            drawPosition();
        });

        // The <select>s it replaces are hidden by CSS from here on.
        document.body.classList.add('has-scrubber');
        return { stop };
    }

    async function init() {
        const el = document.getElementById('version-scrubber');
        if (!el || !root.d3 || !root.DashState) return;  // the <select>s stay as they are
        let versions = [];
        try {
            const data = await (await fetch('/api/versions')).json();
            versions = (data.versions || []).map((v) => v.version).sort();
        } catch (e) {
            console.error('Version scrubber: /api/versions failed', e);
            return;
        }
        if (versions.length < 2) return;
        root.versionScrubber = mount(el, versions, root.TIMELINE_EVENTS || [], root.DashState);
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})(typeof window !== 'undefined' ? window : this);
