// The ontology coverage map for one small branch, as
// plots/latest_plots.py:_render_ontology_coverage_map draws it from the
// fixture in tests/test_ontology_coverage_map.py (shared by the node and
// browser tests).

// [term, id, status, subtree size, level, parent id, start %, width %]
const BLOCKS = [
    ['cardiovascular system', 'UBERON:0004535', 'Contains annotated terms', 200, 0, '', 0, 100],
    ['heart', 'UBERON:0000948', 'Contains annotated terms', 100, 1, '', 0, 50],
    ['blood vessel', 'UBERON:0001981', 'Coverage hole', 60, 1, '', 50, 30],
    ['cardiac chamber', 'UBERON:0004151', 'Contains annotated terms', 40, 2, 'UBERON:0000948', 0, 20],
    ['cardiac valve', 'UBERON:0000946', 'Coverage hole', 30, 2, 'UBERON:0000948', 20, 15],
    ['lymph vessel', 'UBERON:0001473', 'Coverage hole', 20, 2, 'UBERON:0000948', 35, 10],
    ['cardiac ventricle', 'UBERON:0002082', 'Annotated', 10, 3, 'UBERON:0004151', 0, 5],
    ['mitral valve', 'UBERON:0002135', 'Inside a hole', 5, 3, 'UBERON:0000946', 20, 2.5],
];
const STATUSES = [['Annotated', '#29235C'], ['Contains annotated terms', '#307BBF'],
    ['Coverage hole', '#E6007E'], ['Inside a hole', '#93D5F6']];
const LEVELS = ['Branch', 'Level 1', 'Level 2', 'Level 3'];

function coverageMapTraces() {
    const label = Object.fromEntries(BLOCKS.map((b) => [b[1], b[0]]));
    return STATUSES.map(([status, colour]) => {
        const part = BLOCKS.filter((b) => b[2] === status);
        return {
            type: 'bar', orientation: 'h', name: `${status} (${part.length})`,
            y: part.map((b) => LEVELS[b[4]]), base: part.map((b) => b[6]), x: part.map((b) => b[7]),
            text: part.map((b) => b[0]), textposition: 'inside', insidetextanchor: 'middle',
            marker: { color: colour, line: { color: '#FFFFFF', width: 1 } },
            customdata: part.map((b) => [b[0], b[1], b[2], b[3], b[5] ? label[b[5]] : 'the branch root', '', b[5], b[4]]),
            hovertemplate: '<b>%{customdata[0]}</b> (%{customdata[1]})<br>%{customdata[2]}<extra></extra>',
        };
    });
}

function coverageMapLayout() {
    return {
        title: { text: 'Ontology coverage map — Cardiovascular' }, barmode: 'overlay', bargap: 0.06, height: 540,
        xaxis: { range: [0, 100], autorange: false, visible: false },
        yaxis: { categoryorder: 'array', categoryarray: LEVELS, autorange: 'reversed', fixedrange: true },
        uniformtext: { minsize: 10, mode: 'hide' },
    };
}

/** The figure as /api/plot HTML, the way the snapshot page receives it. */
function coverageMapHtml() {
    return '<div id="coverage-map" class="plotly-graph-div" style="height:540px;width:100%"></div>'
        + `<script>Plotly.newPlot("coverage-map", ${JSON.stringify(coverageMapTraces())}, ${JSON.stringify(coverageMapLayout())}, {displayModeBar:false});</script>`;
}

module.exports = { coverageMapTraces, coverageMapLayout, coverageMapHtml };
