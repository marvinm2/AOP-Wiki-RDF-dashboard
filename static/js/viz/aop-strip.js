/**
 * AOP composition history on /aop (#190). Renderer `d3-aop-strip`.
 *
 * - Data: /api/v2/data/latest_aop_composition?aop=<n>, the AOP row and one
 *   row per Key Event, each with a "Roles" string (one character per version,
 *   oldest first: M, K or A for the role in the AOP, "-" absent; "x" for the
 *   AOP row). See plot_latest_aop_composition in plots/aop_history.py.
 * - Drawing: an SVG strip, a row per Key Event and a column per version from
 *   the AOP's first, cells in the MIE/KE/AO role colours of the network graph
 *   (static/data/tokens.json); a Key Event that joined the AOP is marked ▸,
 *   one that left it ×. The top row is the AOP's OECD stage. No transitions.
 * - Keyboard: the strip is one tab stop; the arrow keys, Home/End and Page
 *   Up/Down move a cursor over the cells and each cell is read out; Enter
 *   opens the Key Event's AOP-Wiki page (each label is also a link for the
 *   mouse). "Show Raw Data" lists the rows.
 * - The AOP picker (#aop-picker) takes a number or a title from /api/aops and
 *   sets ?aop=.
 *
 * The pure helpers are exported for node --test.
 */
(function (root) {
    'use strict';

    const ABSENT = '-';
    const ROLES = { M: 'MIE', K: 'KE', A: 'AO' };
    const ROLE_WORDS = { MIE: 'molecular initiating event', KE: 'key event', AO: 'adverse outcome' };
    const KE_IRI = 'https://identifiers.org/aop.events/';

    // ------------------------------------------------------------ pure parts

    /** The payload as row objects, AOP row first. */
    function rowsOf(payload) {
        const d = payload.data;
        const out = [];
        for (let i = 0; i < payload.meta.rows; i++) {
            const r = {};
            Object.keys(d).forEach((k) => { r[k] = d[k][i]; });
            out.push(r);
        }
        return out;
    }

    /** Index of the AOP's first version (its first column). */
    function startOf(aopCodes) {
        const i = String(aopCodes || '').search(/[^-]/);
        return i < 0 ? 0 : i;
    }

    /** {role: 'MIE'|'KE'|'AO'|null, added, removed} for a roles string at version i. */
    function cellState(codes, i, start) {
        const now = codes[i];
        const before = i > start ? codes[i - 1] : null;
        const role = now && now !== ABSENT ? ROLES[now] : null;
        return {
            role,
            added: Boolean(role) && before !== null && before === ABSENT,
            removed: !role && before !== null && before !== ABSENT,
        };
    }

    /** The sentence read out for one cell. */
    function describeCell(row, version, state) {
        const name = `${row.ID}${row.Title ? ` ${row.Title}` : ''}`;
        if (state.role) {
            return `${name}, ${version}: ${ROLE_WORDS[state.role]}${state.added ? ', joined the AOP in this version' : ''}.`;
        }
        return `${name}, ${version}: not in the AOP${state.removed ? ', left it in this version' : ''}.`;
    }

    /** Counts for the summary line: Key Events, and how often one joined or left after the AOP's first version. */
    function summary(rows, versions) {
        const aop = rows[0];
        const kes = rows.slice(1);
        const start = startOf(aop.Roles);
        let added = 0;
        let removed = 0;
        kes.forEach((r) => {
            for (let i = start + 1; i < r.Roles.length; i++) {
                const s = cellState(r.Roles, i, start);
                if (s.added) added++;
                if (s.removed) removed++;
            }
        });
        const last = aop.Roles.length - 1;
        const now = kes.filter((r) => r.Roles[last] !== ABSENT).length;
        const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
        const shown = versions[last];
        return `${aop.ID}${aop.Title ? `: ${aop.Title}` : ''}. ${plural(kes.length, 'Key Event')} in `
            + `${plural(aop.Roles.length - start, 'version')} from ${versions[start]} to ${shown}; `
            + `${now} in ${shown}. After ${versions[start]}: ${added} joined, ${removed} left.`;
    }

    /** The AOP number a picker entry names: "3", "AOP 3", "3 · Title", or a title matching one AOP. */
    function parseAop(text, index) {
        const t = String(text || '').trim();
        const m = t.match(/^(?:aop\s*)?(\d{1,6})(?:\s*·.*)?$/i);
        if (m) return Number(m[1]);
        if (!t || !index) return null;
        const lower = t.toLowerCase();
        const exact = index.filter((a) => a.title.toLowerCase() === lower);
        if (exact.length === 1) return exact[0].number;
        const partial = index.filter((a) => a.title.toLowerCase().includes(lower));
        return partial.length === 1 ? partial[0].number : null;
    }

    function optionText(a) { return `${a.number} · ${a.title}`; }

    function escapeHtml(text) {
        return String(text == null ? '' : text).replace(/[&<>"']/g, (c) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
        }[c]));
    }

    const api = { rowsOf, startOf, cellState, describeCell, summary, parseAop, optionText };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
        return;
    }
    root.DashAopStrip = api;

    // ------------------------------------------------------------- the page

    const LABEL_W = 320;
    const LABEL_W_NARROW = 150;   // a phone: the cells scroll, the labels stay short
    const ROW_H = 20;
    const HEAD_H = 54;
    const STAGE_H = 16;
    let tokens = null;
    let versionsPromise = null;
    const views = new WeakMap();

    function getTokens() {
        if (!tokens) {
            tokens = fetch('/static/data/tokens.json').then((r) => r.json())
                .catch((e) => { tokens = null; throw e; });
        }
        return tokens;
    }

    function versionList() {
        if (!versionsPromise) {
            versionsPromise = fetch('/api/versions').then((r) => r.json())
                .then((d) => (d.versions || []).map((v) => v.version).sort())
                .catch((e) => { versionsPromise = null; throw e; });
        }
        return versionsPromise;
    }

    function cardOf(element) { return element.closest('.plot-box'); }

    function setSummary(element, text) {
        const p = cardOf(element).querySelector('.aop-summary');
        if (p) p.textContent = text;
    }

    function trim(text, n) { return text.length > n ? `${text.slice(0, n - 1)}…` : text; }

    async function mount(element, payload) {
        destroy(element);
        const d3 = root.d3;
        if (!d3) {
            element.innerHTML = '<p class="plot-error">The strip needs a script that did not load. '
                + 'Use "Show Raw Data" below or the CSV download.</p>';
            return;
        }
        const rows = rowsOf(payload);
        if (!rows.length) {
            element.innerHTML = `<p class="aop-empty">${root.DashState && root.DashState.get('aop')
                ? 'No history for this AOP up to the selected version.' : 'Pick an AOP above to see its Key Events in every version.'}</p>`;
            setSummary(element, '');
            return;
        }
        const [all, tk] = await Promise.all([versionList(), getTokens()]);
        const aop = rows[0];
        const kes = rows.slice(1);
        const versions = all.slice(0, aop.Roles.length);
        const start = startOf(aop.Roles);
        const cols = versions.slice(start);
        const stages = String(aop.Stages || '').split('|');
        const roleColour = tk.color.network_roles;
        const stageColour = tk.color.oecd_status;
        const narrow = element.clientWidth < 600;
        const labelW = narrow ? LABEL_W_NARROW : LABEL_W;
        const cellW = Math.max(12, Math.min(26, Math.floor((element.clientWidth - labelW - 16) / cols.length)));
        const width = labelW + cellW * cols.length + 8;
        const height = HEAD_H + STAGE_H + 6 + ROW_H * kes.length + 4;
        const x = (j) => labelW + (j - start) * cellW;
        const y = (r) => HEAD_H + STAGE_H + 6 + r * ROW_H;

        element.innerHTML = '';
        element.setAttribute('role', 'figure');
        element.setAttribute('aria-label', `Key Events of ${aop.ID} per version`);
        const scroll = d3.select(element).append('div').attr('class', 'aop-strip-scroll');
        const svg = scroll.append('svg')
            .attr('class', 'aop-strip')
            .attr('width', width).attr('height', height)
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('tabindex', 0)
            .attr('role', 'group')
            .attr('aria-roledescription', 'strip')
            .attr('aria-label', `${kes.length} Key Events by ${cols.length} versions. Use the arrow keys to read the cells.`);
        const live = d3.select(element).append('p').attr('class', 'sr-only aop-live').attr('aria-live', 'polite');

        // Version labels: the year at its first column, every column ticked.
        const head = svg.append('g').attr('aria-hidden', 'true');
        cols.forEach((v, k) => {
            const j = start + k;
            const year = v.slice(0, 4);
            if (k === 0 || year !== cols[k - 1].slice(0, 4)) {
                head.append('text').attr('class', 'aop-year').attr('x', x(j) + 1).attr('y', HEAD_H - 18).text(year);
                head.append('line').attr('class', 'aop-year-rule')
                    .attr('x1', x(j)).attr('x2', x(j)).attr('y1', HEAD_H - 14).attr('y2', height);
            }
            head.append('text').attr('class', 'aop-month').attr('x', x(j) + cellW / 2).attr('y', HEAD_H - 4)
                .attr('text-anchor', 'middle').text(cellW >= 18 ? v.slice(5, 7) : '');
        });

        // The AOP's OECD stage per version.
        const stageRow = svg.append('g').attr('aria-hidden', 'true');
        stageRow.append('text').attr('class', 'aop-row-label aop-row-label--stage')
            .attr('x', 4).attr('y', HEAD_H + STAGE_H - 4).text('OECD stage');
        cols.forEach((v, k) => {
            const j = start + k;
            if (aop.Roles[j] === ABSENT) return;
            stageRow.append('rect').attr('x', x(j) + 1).attr('y', HEAD_H).attr('width', cellW - 2).attr('height', STAGE_H)
                .attr('fill', stageColour[stages[j]] || '#BFBFBF')
                .append('title').text(`${aop.ID}, ${v}: ${stages[j] || 'not in AOP-Wiki'}`);
        });

        // One row per Key Event.
        const body = svg.append('g');
        kes.forEach((r, i) => {
            const g = body.append('g').attr('class', 'aop-row');
            if (i % 2 === 0) {
                g.append('rect').attr('class', 'aop-band').attr('x', 0).attr('y', y(i))
                    .attr('width', width).attr('height', ROW_H).attr('aria-hidden', 'true');
            }
            const number = r.ID.replace(/^KE\s*/, '');
            const link = g.append('a').attr('href', `${KE_IRI}${number}`).attr('target', '_blank').attr('rel', 'noopener')
                .attr('tabindex', -1);
            link.append('text').attr('class', `aop-row-label aop-row-label--${r.Role.toLowerCase()}`)
                .attr('x', 4).attr('y', y(i) + ROW_H - 6)
                .text(`${r.ID} ${trim(r.Title || '', narrow ? 14 : 40)}`)
                .append('title').text(`${r.ID}: ${r.Title || 'no title'} (newest role: ${ROLE_WORDS[r.Role]})`);
            const cells = g.append('g').attr('aria-hidden', 'true');
            for (let j = start; j < r.Roles.length; j++) {
                const s = cellState(r.Roles, j, start);
                if (s.role) {
                    cells.append('rect').attr('class', 'aop-cell').attr('x', x(j) + 1).attr('y', y(i) + 2)
                        .attr('width', cellW - 2).attr('height', ROW_H - 4).attr('fill', roleColour[s.role])
                        .append('title').text(describeCell(r, versions[j], s));
                    if (s.added) {
                        cells.append('text').attr('class', 'aop-mark aop-mark--added')
                            .attr('x', x(j) + cellW / 2).attr('y', y(i) + ROW_H - 6).attr('text-anchor', 'middle').text('▸');
                    }
                } else if (s.removed) {
                    cells.append('text').attr('class', 'aop-mark aop-mark--removed')
                        .attr('x', x(j) + cellW / 2).attr('y', y(i) + ROW_H - 6).attr('text-anchor', 'middle').text('×')
                        .append('title').text(describeCell(r, versions[j], s));
                }
            }
        });

        // The keyboard cursor.
        const cursor = svg.append('rect').attr('class', 'aop-cursor').attr('width', cellW).attr('height', ROW_H)
            .attr('aria-hidden', 'true').style('display', 'none');
        const view = { row: 0, col: aop.Roles.length - 1 };
        views.set(element, view);
        function show() {
            cursor.style('display', null).attr('x', x(view.col)).attr('y', y(view.row));
            const r = kes[view.row];
            live.text(describeCell(r, versions[view.col], cellState(r.Roles, view.col, start)));
        }
        svg.on('keydown', (event) => {
            if (!kes.length) return;
            const last = aop.Roles.length - 1;
            switch (event.key) {
                case 'ArrowRight': view.col = Math.min(last, view.col + 1); break;
                case 'ArrowLeft': view.col = Math.max(start, view.col - 1); break;
                case 'ArrowDown': view.row = Math.min(kes.length - 1, view.row + 1); break;
                case 'ArrowUp': view.row = Math.max(0, view.row - 1); break;
                case 'Home': view.col = start; break;
                case 'End': view.col = last; break;
                case 'PageDown': view.row = Math.min(kes.length - 1, view.row + 10); break;
                case 'PageUp': view.row = Math.max(0, view.row - 10); break;
                case 'Enter': {
                    const number = kes[view.row].ID.replace(/^KE\s*/, '');
                    root.open(`${KE_IRI}${number}`, '_blank', 'noopener');
                    return;
                }
                case 'Escape': cursor.style('display', 'none'); live.text(''); return;
                default: return;
            }
            event.preventDefault();
            show();
        });
        svg.on('focus', () => { if (kes.length) show(); });
        svg.on('blur', () => cursor.style('display', 'none'));

        setSummary(element, summary(rows, versions));
    }

    function destroy(element) {
        views.delete(element);
        element.innerHTML = '';
    }

    const renderer = {
        endpoint(plotName, qs) { return `/api/v2/data/${plotName}${qs}`; },
        mount,
        update: mount,
        destroy,
    };
    if (root.PlotRuntime) root.PlotRuntime.registerRenderer('d3-aop-strip', renderer);

    // ------------------------------------------------------------- the picker

    async function initPicker() {
        const form = document.getElementById('aop-picker');
        const state = root.DashState;
        if (!form || !state) return;
        const input = form.querySelector('#aop-input');
        const help = form.querySelector('#aop-picker-help');
        let index = [];
        const fill = () => {
            const n = state.get('aop');
            const a = index.find((x) => String(x.number) === String(n));
            input.value = n ? (a ? optionText(a) : String(n)) : '';
        };
        try {
            index = (await fetch('/api/aops').then((r) => r.json())).aops || [];
            const list = document.getElementById('aop-options');
            list.innerHTML = index.map((a) => `<option value="${escapeHtml(optionText(a))}"></option>`).join('');
        } catch (e) {
            index = [];
        }
        fill();
        form.addEventListener('submit', (event) => {
            event.preventDefault();
            const n = parseAop(input.value, index);
            if (n === null) {
                help.classList.remove('sr-only');
                help.classList.add('aop-picker-error');
                help.setAttribute('role', 'alert');
                help.textContent = 'No single AOP matches; type its number, or pick it from the list.';
                return;
            }
            help.className = 'sr-only';
            help.removeAttribute('role');
            help.textContent = 'Type an AOP number, or part of its title and pick it from the list.';
            state.set({ aop: String(n) });
        });
        state.subscribe((changed) => { if (changed.includes('aop')) fill(); });
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initPicker);
    else initPicker();
})(typeof window !== 'undefined' ? window : this);
