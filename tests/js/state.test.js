// Unit tests for static/js/state.js (#181). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const S = require('../../static/js/state.js');

function fakeWindow(search) {
    const loc = { pathname: '/snapshot', search, hash: '' };
    const hist = {
        state: null,
        calls: [],
        replaceState(st, title, url) {
            this.calls.push(url);
            const q = url.indexOf('?');
            loc.search = q === -1 ? '' : url.slice(q).replace(/#.*$/, '');
        },
    };
    return { loc, hist };
}

test('parse reads owned keys and treats empty as default', () => {
    const st = S.parse('?version=2024-01-01&scope=&min_jaccard=0.4&other=1');
    assert.equal(st.version, '2024-01-01');
    assert.equal(st.scope, null);
    assert.equal(st.min_jaccard, '0.4');
    assert.equal(st.other, undefined);
});

test('serialize drops defaults and keeps foreign parameters first', () => {
    const qs = S.serialize({ version: '2024-01-01', scope: null, domain: 'Liver' }, '?utm=x&version=old');
    assert.equal(qs, '?utm=x&version=2024-01-01&domain=Liver');
    assert.equal(S.serialize({ version: null }, ''), '');
});

test('set rewrites the URL and notifies only on change', () => {
    const { loc, hist } = fakeWindow('?start=2020-01-01');
    const state = S.createState(loc, hist);
    const seen = [];
    state.subscribe((changed, snapshot) => seen.push([changed, snapshot.version]));

    assert.deepEqual(state.set({ version: '2024-01-01', bogus: 'x' }), ['version']);
    assert.equal(loc.search, '?version=2024-01-01&start=2020-01-01');  // owned keys in KEYS order
    assert.deepEqual(state.set({ version: '2024-01-01' }), []);  // no-op
    assert.deepEqual(state.set({ version: null, scope: 'ao' }), ['version', 'scope']);
    assert.equal(loc.search, '?start=2020-01-01&scope=ao');
    assert.deepEqual(seen, [[['version'], '2024-01-01'], [['version', 'scope'], null]]);
    assert.equal(hist.calls.length, 2);
});

test('values are stored as strings, empty means default', () => {
    const { loc, hist } = fakeWindow('');
    const state = S.createState(loc, hist);
    state.set({ min_jaccard: 0.4 });
    assert.equal(state.get('min_jaccard'), '0.4');
    state.set({ min_jaccard: '' });
    assert.equal(state.get('min_jaccard'), null);
    assert.equal(loc.search, '');
});

test('unsubscribe stops notifications', () => {
    const { loc, hist } = fakeWindow('');
    const state = S.createState(loc, hist);
    let n = 0;
    const off = state.subscribe(() => n++);
    state.set({ view: 'percentage' });
    off();
    state.set({ view: null });
    assert.equal(n, 1);
});
