/**
 * Figure helpers for the plotly-json and delta renderers (#181).
 *
 * - decodeArray(): Plotly JSON from the server encodes numeric arrays as
 *   {dtype, bdata} (base64). Plotly.js draws those as they are, but anything
 *   that computes on the values needs plain numbers.
 * - deriveDelta(): the quarter-over-quarter view of a trend figure. Every
 *   *_delta trend plot is its absolute plot with y replaced by y[i] - y[i-1]
 *   (0 for the first snapshot) and relabelled; three are drawn as bars. The
 *   per-plot labels come from DELTA_VIEWS in plots/registry.py.
 *
 * Pure functions, exported for node --test.
 */
(function (root) {
    'use strict';

    const DTYPES = {
        f8: Float64Array, f4: Float32Array, i4: Int32Array, i2: Int16Array, i1: Int8Array,
        u4: Uint32Array, u2: Uint16Array, u1: Uint8Array,
    };

    function base64ToBytes(b64) {
        if (typeof atob === 'function') {
            const bin = atob(b64);
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
            return bytes;
        }
        return new Uint8Array(Buffer.from(b64, 'base64'));  // node --test
    }

    /** A plain array from a JSON array or a {dtype, bdata} typed-array spec. */
    function decodeArray(value) {
        if (value == null || Array.isArray(value)) return value;
        if (typeof value === 'object' && value.bdata && DTYPES[value.dtype]) {
            const bytes = base64ToBytes(value.bdata);
            const Typed = DTYPES[value.dtype];
            return Array.from(new Typed(bytes.buffer, bytes.byteOffset, bytes.byteLength / Typed.BYTES_PER_ELEMENT));
        }
        return value;
    }

    /** Successive differences, 0 for the first point, null across gaps. */
    function differences(values) {
        return values.map((v, i) => {
            if (i === 0) return 0;
            const prev = values[i - 1];
            return v == null || prev == null ? null : v - prev;
        });
    }

    /** Replace the label in front of "=%{y}" in a hovertemplate. */
    function relabelHover(template, label) {
        if (typeof template !== 'string') return template;
        return template.replace(/(^|<br>)([^<>=]*)=%\{y\}/, (m, pre) => `${pre}${label}=%{y}`);
    }

    /**
     * The delta figure for `fig` under `spec` ({y_title, hover_label?, bar?,
     * name_replace?: [from, to]}). `fig` is not modified.
     */
    function deriveDelta(fig, spec) {
        const out = JSON.parse(JSON.stringify(fig));
        const hoverLabel = spec.hover_label || spec.y_title;
        const rename = (s) => (spec.name_replace && typeof s === 'string'
            ? s.split(spec.name_replace[0]).join(spec.name_replace[1]) : s);
        out.data = (out.data || []).map((trace) => {
            const t = Object.assign({}, trace);
            t.x = decodeArray(t.x);
            t.y = differences(decodeArray(t.y) || []);
            if (t.name !== undefined) t.name = rename(t.name);
            if (t.legendgroup !== undefined) t.legendgroup = rename(t.legendgroup);
            t.hovertemplate = relabelHover(rename(t.hovertemplate), hoverLabel);
            if (spec.bar) {
                const color = (t.line && t.line.color) || (t.marker && t.marker.color);
                t.type = 'bar';
                ['mode', 'line', 'stackgroup', 'fill', 'fillcolor', 'fillpattern'].forEach((k) => delete t[k]);
                t.marker = Object.assign({}, t.marker || {}, color ? { color } : {});
                delete t.marker.symbol;
            }
            return t;
        });
        out.layout = Object.assign({}, out.layout);
        out.layout.yaxis = Object.assign({}, out.layout.yaxis, {
            title: Object.assign({}, (out.layout.yaxis || {}).title, { text: spec.y_title }),
        });
        if (spec.bar) out.layout.barmode = 'relative';
        return out;
    }

    const api = { decodeArray, differences, relabelHover, deriveDelta };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.DashFigure = api;
})(typeof window !== 'undefined' ? window : this);
