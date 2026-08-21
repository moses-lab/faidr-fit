// Diverging horizontal bar chart of feature strength (signed Wald z from the
// refit): enriched features extend right, depleted extend left. Colour encodes
// sign, but so does direction, so identity never rests on colour alone. Colours
// live in CSS custom properties (--pos / --neg / ink tokens) set by the page, so
// light and dark themes are handled without touching this code.
//
// Design follows the dataviz method: thin bars, rounded data-end only, a 2px gap
// between rows, recessive zero axis, direct labels (feature name + value), and a
// native per-bar hover tooltip.
//
// The SVG is always rendered at a fixed pixel size that matches its own
// viewBox exactly, so the browser never rescales it (a rescaled SVG whose
// height doesn't match its rescaled width gets letterboxed, growing with row
// count). The label column and value gutters shrink smoothly with the
// container down to LABEL_MIN/GUTTER_MIN, so narrow phones fit without
// needing to scroll; HARD_MIN is just a failsafe against a zero-width render.

const ROW_H = 26, BAR_H = 12, PAD = 8, R = 3, HARD_MIN = 240;
const LABEL_FULL = 168, LABEL_MIN = 96, W_COMPACT = 300, W_FULL = 420;
const GUTTER_FULL = 56, GUTTER_MIN = 44;
const FEAT_FONT = "12px system-ui, -apple-system, sans-serif";

// Linear interpolation between a compact and a full value, clamped, keyed off
// the available width so layout degrades gracefully rather than at a cliff.
function scaleDim(full, min, width) {
  if (width >= W_FULL) return full;
  if (width <= W_COMPACT) return min;
  const t = (width - W_COMPACT) / (W_FULL - W_COMPACT);
  return Math.round(min + t * (full - min));
}

// rect-with-only-the-data-end rounded, as an SVG path
function bar(x0, x1, y, h, r) {
  const dir = x1 >= x0 ? 1 : -1;
  r = Math.min(r, Math.abs(x1 - x0));
  const xe = x1 - dir * r;
  return `M${x0},${y} L${xe},${y} Q${x1},${y} ${x1},${y + r}` +
    ` L${x1},${y + h - r} Q${x1},${y + h} ${xe},${y + h} L${x0},${y + h} Z`;
}

const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

// Truncates text to fit maxWidth, appending an ellipsis, so truncation is
// always a visible "…" rather than a silent clip at the SVG edge.
let measureCtx = null;
function truncate(text, maxWidth, font) {
  if (!measureCtx) measureCtx = document.createElement("canvas").getContext("2d");
  measureCtx.font = font;
  if (measureCtx.measureText(text).width <= maxWidth) return text;
  let lo = 0, hi = text.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (measureCtx.measureText(text.slice(0, mid) + "…").width <= maxWidth) lo = mid;
    else hi = mid - 1;
  }
  return text.slice(0, lo) + "…";
}

// items: [{ feature, value }] already sorted by the caller (strongest first)
export function renderBars(container, items) {
  if (!items.length) { container.innerHTML = `<p class="empty">No features selected.</p>`; return; }

  // Use the real container width (HARD_MIN is only a failsafe against a
  // zero-width render before layout settles). No artificial floor beyond
  // that, so narrow phones aren't forced into horizontal scroll.
  const width = Math.max(container.clientWidth || 640, HARD_MIN);
  const LABEL_W = scaleDim(LABEL_FULL, LABEL_MIN, width);
  const GUTTER = scaleDim(GUTTER_FULL, GUTTER_MIN, width);

  // The bar area sits between the label column and a value-label gutter on
  // BOTH sides, so a long negative value's number has its own reserved space
  // and never reaches back into the labels.
  const plotL = LABEL_W + GUTTER, plotR = width - GUTTER;
  const plotW = plotR - plotL;
  const zeroX = plotL + plotW / 2;
  const maxAbs = Math.max(...items.map((d) => Math.abs(d.value))) || 1;
  const scale = (v) => (v / maxAbs) * (plotW / 2 - PAD);
  const height = items.length * ROW_H + 8;
  const labelMaxW = LABEL_W - 16;

  const rows = items.map((d, i) => {
    const y = i * ROW_H + 4, cy = y + ROW_H / 2;
    const end = zeroX + scale(d.value);
    const cls = d.value >= 0 ? "pos" : "neg";
    const vx = d.value >= 0 ? end + 6 : end - 6;
    const anchor = d.value >= 0 ? "start" : "end";
    const head = d.raw && d.raw !== d.feature ? `${d.feature} [${d.raw}]` : d.feature;
    const tip = `${head}: ${d.value >= 0 ? "+" : ""}${d.value.toFixed(2)} (z)${d.desc ? `\n${d.desc}` : ""}`;
    const label = truncate(d.feature, labelMaxW, FEAT_FONT);
    return `<g class="bar-row">
      <title>${esc(tip)}</title>
      <text class="feat" x="${LABEL_W - 12}" y="${cy}" text-anchor="end" dominant-baseline="central">${esc(label)}</text>
      <path class="${cls}" d="${bar(zeroX, end, cy - BAR_H / 2, BAR_H, R)}"/>
      <text class="val" x="${vx}" y="${cy}" text-anchor="${anchor}" dominant-baseline="central">${d.value >= 0 ? "+" : ""}${d.value.toFixed(2)}</text>
    </g>`;
  }).join("");

  container.style.minHeight = `${container.offsetHeight}px`;

  container.innerHTML = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img"
      aria-label="Feature strength, signed Wald z">
    <line class="axis" x1="${zeroX}" x2="${zeroX}" y1="2" y2="${height - 6}"/>
    ${rows}
  </svg>`;
}
