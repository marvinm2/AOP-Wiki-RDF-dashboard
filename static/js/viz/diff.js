/**
 * Version Diff Explorer on /diff (#188).
 *
 * Version B is the page's `version` (version-selector.js) and version A its
 * `against`: this module fills the "From (A)" selector with the versions older
 * than B, the first option meaning "the version before B" (null in the state).
 *
 * Renderer `diff-worklist` draws the changed entities from /api/v2/data as a
 * table: filters for the entity type and the kind of change, a text filter on
 * the name and ID, and a link per entity. It is plain HTML, so every part is
 * reachable with the keyboard and read by screen readers as a table; the
 * count of matching rows is announced as the filters change.
 *
 * The pure helpers are exported for node --test.
 */
(function (root) {
    'use strict';

    const SHOWN = 300;   // rows drawn before "Show all"

    /** Rows of a /api/v2/data payload as objects. */
    function rowsOf(payload) {
        const names = payload.meta.columns.map((c) => c.name);
        const n = payload.meta.rows;
        const rows = [];
        for (let i = 0; i < n; i++) {
            const row = {};
            names.forEach((c) => { row[c] = payload.data[c][i]; });
            rows.push(row);
        }
        return rows;
    }

    /** Rows matching the filters: type and change exactly ('' = any), text in the name or ID. */
    function filterRows(rows, filters) {
        const text = (filters.text || '').trim().toLowerCase();
        return rows.filter((r) => (!filters.type || r.Type === filters.type)
            && (!filters.change || r.Change === filters.change)
            && (!text || String(r.Name).toLowerCase().includes(text) || String(r.ID).toLowerCase() === text));
    }

    /** The distinct values of a column, in first-seen order (the server sorts the rows). */
    function distinct(rows, column) {
        return Array.from(new Set(rows.map((r) => r[column]).filter((v) => v)));
    }

    /** Versions A may be for B: those older than B, newest first. */
    function againstOptions(versions, b) {
        return versions.filter((v) => v < b).sort().reverse();
    }

    function escapeHtml(v) {
        return String(v == null ? '' : v).replace(/[&<>"']/g, (c) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
        }[c]));
    }

    function stageCell(r) {
        if (!r['OECD Stage A'] && !r['OECD Stage B']) return '';
        if (r['OECD Stage A'] === r['OECD Stage B']) return escapeHtml(r['OECD Stage A']);
        return `${escapeHtml(r['OECD Stage A'] || '—')} → ${escapeHtml(r['OECD Stage B'] || '—')}`;
    }

    function tableHtml(rows, limit) {
        let html = '<div class="data-table-wrapper"><table class="raw-data-table diff-worklist-table">'
            + '<caption class="sr-only">Changed entities</caption><thead><tr>'
            + ['Type', 'ID', 'Name', 'Change', 'Fields changed', 'OECD stage'].map((c) => `<th scope="col">${c}</th>`).join('')
            + '</tr></thead><tbody>';
        rows.slice(0, limit).forEach((r) => {
            html += `<tr><td>${escapeHtml(r.Type)}</td>`
                + `<td><a href="${escapeHtml(r.Link)}" target="_blank" rel="noopener">${escapeHtml(r.ID)}</a></td>`
                + `<td>${escapeHtml(r.Name)}</td><td>${escapeHtml(r.Change)}</td>`
                + `<td>${escapeHtml(r['Fields Changed'])}</td><td>${stageCell(r)}</td></tr>`;
        });
        return html + '</tbody></table></div>';
    }

    function select(label, values, any) {
        const options = [`<option value="">${any}</option>`]
            .concat(values.map((v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`));
        return `<label>${label} <select>${options.join('')}</select></label>`;
    }

    const views = new WeakMap();   // container -> {rows, limit}

    function draw(element) {
        const view = views.get(element);
        const [type, change] = element.querySelectorAll('.diff-filters select');
        const text = element.querySelector('.diff-filters input');
        const matched = filterRows(view.rows, { type: type.value, change: change.value, text: text.value });
        const body = element.querySelector('.diff-worklist-body');
        body.innerHTML = matched.length ? tableHtml(matched, view.limit) : '<p class="diff-empty">No entity matches the filters.</p>';
        const more = element.querySelector('.diff-show-all');
        more.hidden = matched.length <= view.limit;
        more.textContent = `Show all ${matched.length} rows`;
        const shown = Math.min(matched.length, view.limit);
        element.querySelector('.diff-count').textContent = matched.length === view.rows.length
            ? `${view.rows.length} changed entities${shown < matched.length ? `, first ${shown} shown` : ''}.`
            : `${matched.length} of ${view.rows.length} changed entities match${shown < matched.length ? `, first ${shown} shown` : ''}.`;
    }

    function mount(element, payload) {
        const rows = rowsOf(payload);
        views.set(element, { rows, limit: SHOWN });
        element.innerHTML = '<div class="diff-filters" role="group" aria-label="Filter the changed entities">'
            + select('Type', distinct(rows, 'Type'), 'All types')
            + select('Change', distinct(rows, 'Change'), 'All changes')
            + '<label>Name or ID <input type="search" placeholder="e.g. liver"></label>'
            + '</div>'
            + '<p class="diff-count" aria-live="polite"></p>'
            + '<div class="diff-worklist-body"></div>'
            + '<button type="button" class="data-table-toggle diff-show-all" hidden></button>';
        element.querySelectorAll('.diff-filters select, .diff-filters input').forEach((control) => {
            control.addEventListener(control.tagName === 'INPUT' ? 'input' : 'change', () => {
                views.get(element).limit = SHOWN;
                draw(element);
            });
        });
        element.querySelector('.diff-show-all').addEventListener('click', () => {
            views.get(element).limit = Infinity;
            draw(element);
        });
        draw(element);
    }

    const renderer = {
        endpoint(plotName, qs) { return `/api/v2/data/${plotName}${qs}`; },
        mount,
        update: mount,
        destroy(element) { views.delete(element); element.innerHTML = ''; },
    };

    /** The "From (A)" selector, kept to the versions older than B. */
    async function initAgainst() {
        const state = root.DashState;
        const selector = document.getElementById('diff-against');
        if (!state || !selector) return;
        let versions = [];
        try {
            versions = ((await (await fetch('/api/versions')).json()).versions || []).map((v) => v.version);
        } catch (e) {
            console.error('Version list unavailable:', e);
        }
        if (!versions.length) return;
        const latest = versions.slice().sort().pop();
        function fill() {
            const b = state.get('version') || latest;
            const options = againstOptions(versions, b);
            const wanted = state.get('against');
            if (wanted && !options.includes(wanted)) {
                state.set({ against: null });   // A must be older than B
                return;                          // the subscription fills it again
            }
            selector.innerHTML = '';
            selector.add(new Option(options.length ? `The version before B (${options[0]})` : 'No older version', ''));
            options.forEach((v) => selector.add(new Option(v, v)));
            selector.value = wanted || '';
        }
        selector.addEventListener('change', () => state.set({ against: selector.value || null }));
        state.subscribe((changed) => {
            if (changed.includes('version') || changed.includes('against')) fill();
        });
        fill();
    }

    const api = { rowsOf, filterRows, distinct, againstOptions, tableHtml };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    if (root.PlotRuntime) root.PlotRuntime.registerRenderer('diff-worklist', renderer);
    if (typeof document !== 'undefined') {
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initAgainst);
        else initAgainst();
    }
})(typeof window !== 'undefined' ? window : globalThis);
