// Diverging horizontal bar chart of feature strength (signed Wald z from the
// refit): enriched features extend right, depleted extend left. Colour encodes
// sign, but so does direction, so identity never rests on colour alone. Colours
// live in CSS custom properties (--pos / --neg / ink tokens) set by the page, so
// light and dark themes are handled without touching this code.
//
// Design follows the dataviz method: thin bars, rounded data-end only, a 2px gap
// between rows, recessive zero axis, direct labels (feature name + value), and a
// native per-bar hover tooltip.

const ROW_H = 26, BAR_H = 12, LABEL_W = 168, VALUE_W = 52, PAD = 10, R = 3;

// rect-with-only-the-data-end rounded, as an SVG path
function bar(x0, x1, y, h, r) {
  const dir = x1 >= x0 ? 1 : -1;
  r = Math.min(r, Math.abs(x1 - x0));
  const xe = x1 - dir * r;
  return `M${x0},${y} L${xe},${y} Q${x1},${y} ${x1},${y + r}` +
    ` L${x1},${y + h - r} Q${x1},${y + h} ${xe},${y + h} L${x0},${y + h} Z`;
}

const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

// items: [{ feature, value }] already sorted by the caller (strongest first)
export function renderBars(container, items) {
  if (!items.length) { container.innerHTML = `<p class="empty">No features selected at this penalty.</p>`; return; }
  const width = Math.max(container.clientWidth || 640, 420);
  const plotW = width - LABEL_W - VALUE_W - PAD;
  const zeroX = LABEL_W + plotW / 2;
  const maxAbs = Math.max(...items.map((d) => Math.abs(d.value))) || 1;
  const scale = (v) => (v / maxAbs) * (plotW / 2 - PAD);
  const height = items.length * ROW_H + 8;

  const rows = items.map((d, i) => {
    const y = i * ROW_H + 4, cy = y + ROW_H / 2;
    const end = zeroX + scale(d.value);
    const cls = d.value >= 0 ? "pos" : "neg";
    const vx = d.value >= 0 ? end + 6 : end - 6;
    const anchor = d.value >= 0 ? "start" : "end";
    const head = d.raw && d.raw !== d.feature ? `${d.feature} [${d.raw}]` : d.feature;
    const tip = `${head}: ${d.value >= 0 ? "+" : ""}${d.value.toFixed(2)} (z)${d.desc ? `\n${d.desc}` : ""}`;
    return `<g class="bar-row">
      <title>${esc(tip)}</title>
      <text class="feat" x="${LABEL_W - 12}" y="${cy}" text-anchor="end" dominant-baseline="central">${esc(d.feature)}</text>
      <path class="${cls}" d="${bar(zeroX, end, cy - BAR_H / 2, BAR_H, R)}"/>
      <text class="val" x="${vx}" y="${cy}" text-anchor="${anchor}" dominant-baseline="central">${d.value >= 0 ? "+" : ""}${d.value.toFixed(2)}</text>
    </g>`;
  }).join("");

  container.style.minHeight = `${container.offsetHeight}px`;

  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" role="img"
      aria-label="Feature strength, signed Wald z">
    <line class="axis" x1="${zeroX}" x2="${zeroX}" y1="2" y2="${height - 6}"/>
    ${rows}
  </svg>`;
}
