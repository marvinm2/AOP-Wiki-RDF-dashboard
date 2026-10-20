/**
 * KER curation calendar on /trends (#187).
 *
 * The calendar is a server Plotly figure: one heatmap strip per year, weeks
 * across and weekdays down, a cell per day (_render_ker_calendar in
 * plots/trends_plots.py). This module adds the keyboard path a calendar
 * needs, which viz/a11y.js (one axis only) cannot give: on the focused chart
 * Up/Down move a day, Left/Right a week, PageUp/PageDown a year, Home/End to
 * the first and last day shown, Escape clears. The day is hovered on the
 * chart and read out in the live region.
 *
 * Cells carry "Tue 2023-10-17: 33" as customdata; days outside a year or
 * after the newest snapshot are null.
 *
 * The pure helpers are exported for node --test.
 */
(function (root) {
    'use strict';

    const PLOT = 'ker_curation_calendar';
    const WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

    /** Every drawn day, oldest first: {date, count, curve, row, col, subplot}. */
    function daysOf(traces) {
        const days = [];
        traces.forEach((t, curve) => {
            const cd = t.customdata || [];
            const subplot = (t.xaxis || 'x') + (t.yaxis || 'y');
            cd.forEach((row, r) => Array.from(row || []).forEach((cell, c) => {
                const m = /(\d{4}-\d{2}-\d{2}): (\d+)/.exec(cell || '');
                if (m) days.push({ date: m[1], count: Number(m[2]), curve, row: r, col: c, subplot });
            }));
        });
        return days.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
    }

    function shift(date, n) {
        const d = new Date(`${date}T00:00:00Z`);
        d.setUTCDate(d.getUTCDate() + n);
        return d.toISOString().slice(0, 10);
    }

    /**
     * The index (into `days`) a key moves to from `current` (an index or
     * null); undefined for other keys. A move past a gap (blank days) lands
     * on the nearest drawn day in that direction.
     */
    function calendarStep(key, current, days) {
        if (!days.length) return undefined;
        if (key === 'Home') return 0;
        if (key === 'End') return days.length - 1;
        const by = { ArrowUp: -1, ArrowDown: 1, ArrowLeft: -7, ArrowRight: 7 }[key];
        const year = key === 'PageUp' ? -1 : key === 'PageDown' ? 1 : 0;
        if (by === undefined && !year) return undefined;
        if (current === null || current === undefined) return days.length - 1;
        const from = days[current].date;
        const want = year
            ? `${Number(from.slice(0, 4)) + year}${from.slice(4)}`
            : shift(from, by);
        const forward = want > from;
        if (forward) {
            const i = days.findIndex((d) => d.date >= want);
            return i < 0 ? current : i;
        }
        let i = -1;
        days.forEach((d, j) => { if (d.date <= want) i = j; });
        return i < 0 ? current : i;
    }

    /** The sentence read out for a day. */
    function describeDay(day) {
        const weekday = WEEKDAYS[(new Date(`${day.date}T00:00:00Z`).getUTCDay() + 6) % 7];
        const n = day.count;
        return `${weekday} ${day.date}: ${n === 0 ? 'no' : n} KER${n === 1 ? '' : 's'} created.`;
    }

    const api = { PLOT, daysOf, calendarStep, describeDay };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
        return;
    }
    root.DashCalendar = api;

    // ------------------------------------------------------------- the page

    const steps = new WeakMap();   // container -> {days, at}

    function onKey(event) {
        const container = event.currentTarget;
        const div = container.querySelector('.js-plotly-plot');
        if (!div || !div._fullData || !root.Plotly) return;
        const live = container.closest('.plot-box').querySelector('.calendar-live');
        let view = steps.get(container);
        if (!view) { view = { days: daysOf(div._fullData), at: null }; steps.set(container, view); }
        if (event.key === 'Escape') {
            if (view.at === null) return;
            view.at = null;
            root.Plotly.Fx.unhover(div);
            if (live) live.textContent = 'Day cleared.';
            return;
        }
        const next = calendarStep(event.key, view.at, view.days);
        if (next === undefined) return;
        event.preventDefault();
        view.at = next;
        const day = view.days[next];
        root.Plotly.Fx.hover(div, [{ curveNumber: day.curve, pointNumber: [day.row, day.col] }], day.subplot);
        if (live) live.textContent = describeDay(day);
    }

    document.addEventListener('plot:rendered', (event) => {
        const container = event.target;
        if (!container || !container.dataset || container.dataset.plotName !== PLOT) return;
        steps.delete(container);
        container.tabIndex = 0;
        container.setAttribute('role', 'figure');
        container.setAttribute('aria-label', 'Calendar of KER creation, one strip per year, a cell per day. '
            + 'Up and down move a day, left and right a week, Page Up and Page Down a year, '
            + 'or use "Show Raw Data" below for the table.');
        if (!container.dataset.calendarKeys) {
            container.dataset.calendarKeys = '1';
            container.addEventListener('keydown', onKey);
        }
    });
})(typeof window !== 'undefined' ? window : globalThis);
