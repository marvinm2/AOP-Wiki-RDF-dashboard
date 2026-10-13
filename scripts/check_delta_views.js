#!/usr/bin/env node
// Check that the browser-derived delta views (static/js/viz/figure.js +
// DELTA_VIEWS in plots/registry.py) draw what the server's *_delta figures
// draw: same y values, chart type, trace names, hover text, axis title and bar
// colours (#181). Needs a running dashboard; node 18+.
//
//   node scripts/check_delta_views.js "$(python -c 'import json; from plots import registry; print(json.dumps(registry.DELTA_VIEWS))')" [base URL]
const F = require('../static/js/viz/figure.js');
const views = JSON.parse(process.argv[2]);
const B = (process.argv[3] || 'https://aopwiki-dashboard.vhp4safety.nl') + '/api/v2/figure/';
(async () => {
  let bad = 0;
  for (const [id, spec] of Object.entries(views)) {
    const abs = (await (await fetch(B + spec.of)).json()).figure;
    const srv = (await (await fetch(B + id)).json()).figure;
    const mine = F.deriveDelta(abs, spec);
    const probs = [];
    if ((srv.layout.yaxis.title || {}).text !== mine.layout.yaxis.title.text) probs.push('ytitle');
    if ((srv.layout.barmode || null) !== (mine.layout.barmode || null)) probs.push('barmode');
    srv.data.forEach((t, i) => {
      const m = mine.data[i];
      const ys = F.decodeArray(t.y), ym = m.y;
      if (ys.length !== ym.length || ys.some((v, j) => Math.abs(v - ym[j]) > 1e-9)) probs.push(`y[${i}]`);
      if ((t.type || 'scatter') !== (m.type || 'scatter')) probs.push(`type[${i}]`);
      if (t.name !== m.name) probs.push(`name[${i}] ${t.name} vs ${m.name}`);
      if (t.hovertemplate !== m.hovertemplate) probs.push(`hover[${i}] ${t.hovertemplate} | ${m.hovertemplate}`);
      const cs = t.marker && t.marker.color, cm = m.marker && m.marker.color;
      if (m.type === 'bar' && cs !== cm) probs.push(`color[${i}] ${cs} vs ${cm}`);
    });
    if (probs.length) bad++;
    console.log(id.padEnd(42), probs.length ? probs.slice(0, 4).join('; ') : 'OK');
  }
  console.log(bad ? `${bad} differ` : 'all match');
  process.exitCode = bad ? 1 : 0;
})();
