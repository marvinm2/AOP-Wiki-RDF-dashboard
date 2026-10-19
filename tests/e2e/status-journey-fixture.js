// A small OECD status journey sankey, shared by the unit and browser tests:
// three anchors, five stages, the 2024-04-01 wipe between the last two.
const NODES = [
    ['2023-01-01', 'Under Development', 3], ['2023-01-01', 'Included in OECD Work Plan', 1],
    ['2024-01-01', 'Under Development', 4],
    ['2025-01-01', 'No Status', 2], ['2025-01-01', 'Under Development', 1], ['2025-01-01', 'Under Review', 1],
];
// [source, target, AOPs, colour]
const LINKS = [
    [0, 2, 3, 'rgba(0, 159, 227, 0.12)'], [1, 2, 1, 'rgba(147, 213, 246, 0.4)'],
    [2, 3, 2, 'rgba(0, 159, 227, 0.4)'], [2, 4, 1, 'rgba(0, 159, 227, 0.12)'], [2, 5, 1, 'rgba(0, 159, 227, 0.4)'],
];

function text([s, t, n]) {
    const [a, from] = NODES[s];
    const [b, to] = NODES[t];
    return `${n} AOP${n === 1 ? '' : 's'} ${from === to ? 'stayed' : 'moved'}: ${from} (${a}) to ${to} (${b})`;
}

function journeyFigure() {
    return {
        data: [{
            type: 'sankey', arrangement: 'fixed',
            node: {
                label: NODES.map((n) => n[1]), customdata: NODES.map((n) => [n[0], n[2]]),
                x: [0.001, 0.001, 0.5, 0.999, 0.999, 0.999], y: [0.3, 0.8, 0.5, 0.2, 0.55, 0.85],
                pad: 12, thickness: 14,
            },
            link: {
                source: LINKS.map((l) => l[0]), target: LINKS.map((l) => l[1]), value: LINKS.map((l) => l[2]),
                color: LINKS.map((l) => l[3]), customdata: LINKS.map(text),
            },
        }],
        layout: { title: { text: 'journey' }, height: 420 },
    };
}

module.exports = { journeyFigure };
