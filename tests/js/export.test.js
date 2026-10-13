// Unit tests for static/js/viz/export.js (#181). Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert/strict');
const E = require('../../static/js/viz/export.js');

test('file names carry the version for snapshot plots and the settings', () => {
    const state = { version: '2026-01-01', scope: 'ke', view: null, start: '2020-01-01', end: null };
    assert.equal(E.filename('latest_ontology_coverage', state, { versioned: true, params: ['scope', 'view'] }),
        'latest_ontology_coverage_2026-01-01_scope-ke');
    // The latest version is the default: no version in the state, none in the name.
    assert.equal(E.filename('latest_entity_counts', {}, { versioned: true }), 'latest_entity_counts');
});

test('trend file names carry the snapshot range only when the figure is clamped to it', () => {
    const state = { start: '2020-01-01', end: '2026-01-01', version: '2025-01-01' };
    assert.equal(E.filename('aop_entity_counts_absolute', state, { ranged: true }),
        'aop_entity_counts_absolute_2020-01-01_to_2026-01-01');
    assert.equal(E.filename('aop_entity_counts_absolute', { end: '2024-01-01' }, { ranged: true }),
        'aop_entity_counts_absolute_first_to_2024-01-01');
    assert.equal(E.filename('aops_created_over_time', state, { ranged: false }), 'aops_created_over_time');
});

test('unsafe characters in settings do not reach the file name', () => {
    assert.equal(E.filename('p', { branch: 'GO:0008150 / x' }, { params: ['branch'] }), 'p_branch-GO-0008150-x');
});

test('per-plot download links name their plot; page-level ones do not', () => {
    assert.equal(E.plotOfDownload('/download/latest_entity_counts?format=png'), 'latest_entity_counts');
    assert.equal(E.plotOfDownload('/download/latest/aop_aop_overlap?format=csv&version=2026-01-01'), 'latest_aop_aop_overlap');
    assert.equal(E.plotOfDownload('/download/trend/aop_property_presence_absolute?format=svg'), 'aop_property_presence_absolute');
    assert.equal(E.plotOfDownload('/download/bulk?category=trends-all&formats=csv'), null);
    assert.equal(E.plotOfDownload('/download/network/metrics'), null);
    assert.equal(E.plotOfDownload('/about'), null);
});
