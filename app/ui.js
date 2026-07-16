// The app shell: turns the two inputs (a positive protein/IDR set, and a feature-
// count slider) into a fit and renders the feature-strength chart. The lasso path
// (up to DFMAX features) is fit once per label change; sliding just looks up a
// lambda already sitting on that path, so it's cheap array lookups, not refits.
import { fitLassoLogistic, fitLogistic } from "../core/src/index.js";
import { renderBars } from "./chart.js";

const DFMAX = 30;  // widest feature count the path is fit out to

// Match a pasted list against the proteome. Each line is trimmed; we match an id
// exactly, else as a prefix (so "Q5VUJ6" catches "Q5VUJ6_IDR_1_54"). Returns a
// 0/1 label vector plus how many lines matched something.
function buildLabels(text, ids) {
  const wanted = text.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  const y = new Float64Array(ids.length);
  let matched = 0;
  const exact = new Set(wanted);
  for (let i = 0; i < ids.length; i++) {
    if (exact.has(ids[i]) || wanted.some((w) => ids[i].startsWith(w))) { y[i] = 1; matched++; }
  }
  return { y, matched, nWanted: wanted.length };
}

// df has a value at every lambda step, with duplicates (several lambdas can
// share a feature count) and gaps (some counts, e.g. 16, may never occur on
// this path). Keep the first (most-penalized) step at each distinct count,
// sorted ascending -- that's the discrete stop list the slider walks over.
function distinctCounts(df) {
  const firstStep = new Map();
  for (let i = 0; i < df.length; i++) if (!firstStep.has(df[i])) firstStep.set(df[i], i);
  const counts = [...firstStep.keys()].sort((a, b) => a - b);
  return { counts, firstStep };
}

export function startApp(proteome, go, root = document) {
  const { X, ids, labels, descs } = proteome;
  const el = (id) => root.getElementById(id);
  const positives = el("positives"), slider = el("features"), chart = el("chart");
  const status = el("status"), lamOut = el("lambda-value"), countOut = el("feature-count");
  const goSelect = el("go-select"), goRow = el("go-row");
  const decBtn = el("feat-dec"), incBtn = el("feat-inc");

  // State from the most recent path fit. counts/firstStep let the slider (a
  // plain index 0..counts.length-1) resolve to a real step on the path.
  let y = null, lambdaPath = [], coefficients = [], counts = [], firstStep = new Map();

  function setSliderEnabled(enabled) {
    slider.disabled = !enabled;
    if (decBtn) decBtn.disabled = !enabled;
    if (incBtn) incBtn.disabled = !enabled;
  }

  function renderAt(idx) {
    if (!counts.length) return;
    idx = Math.max(0, Math.min(counts.length - 1, Math.round(idx)));
    slider.value = String(idx);

    const featureCount = counts[idx];
    const pathIdx = firstStep.get(featureCount);
    const lambda = lambdaPath[pathIdx];
    const beta = coefficients[pathIdx].beta;
    lamOut.textContent = lambda.toExponential(2);

    const sel = [];
    for (let j = 0; j < X.p; j++) if (beta[j] !== 0) sel.push(j);
    countOut.textContent = String(sel.length);

    if (!sel.length) { renderBars(chart, []); return; }
    // unpenalised refit on the selected features for signed significance (Wald z)
    const selCols = sel.map((j) => X.cols[j]);
    const { waldZ } = fitLogistic(selCols, y);
    const items = sel
      .map((j, k) => ({ feature: labels[j], raw: X.features[j], desc: descs[j], value: waldZ[k] }))
      .sort((a, b) => b.value - a.value);                    // display enriched → depleted
    renderBars(chart, items);
  }

  // A hand-typed target count (Enter/blur on the readout) snaps to whichever
  // available step is closest -- the path just doesn't have every count.
  function commitFeatureCount() {
    if (!counts.length) return;
    const v = parseInt(countOut.textContent, 10);
    if (!Number.isFinite(v)) { renderAt(+slider.value); return; }
    let best = 0, bestDist = Infinity;
    for (let i = 0; i < counts.length; i++) {
      const d = Math.abs(counts[i] - v);
      if (d < bestDist) { bestDist = d; best = i; }
    }
    renderAt(best);
  }

  // Single entry point: validate a 0/1 label vector, fit the whole path once
  // (up to DFMAX features), and render a default step.
  function applyLabels(labelVector, statusText) {
    let pos = 0; for (let i = 0; i < labelVector.length; i++) pos += labelVector[i];
    const neg = ids.length - pos;
    status.textContent = statusText;
    if (pos < 2 || neg < 2) {
      y = null; counts = []; firstStep = new Map();
      renderBars(chart, []); countOut.textContent = "0"; lamOut.textContent = "—";
      slider.min = "0"; slider.max = "0"; slider.value = "0";
      setSliderEnabled(false);
      return;
    }
    y = labelVector;
    const path = fitLassoLogistic(X.cols, y, { dfmax: DFMAX, nlambda: 150 });
    lambdaPath = path.lambdaPath;
    coefficients = path.coefficients;
    ({ counts, firstStep } = distinctCounts(path.df));

    slider.min = "0";
    slider.max = String(counts.length - 1);
    slider.step = "1";
    setSliderEnabled(true);
    renderAt(Math.floor((counts.length - 1) / 2)); // start roughly mid-range
  }

  // Paste box: matches ids/prefixes; typing here clears any GO selection.
  function fromTextarea() {
    if (goSelect) goSelect.value = "";
    const { y: lab, matched, nWanted } = buildLabels(positives.value, ids);
    const neg = ids.length - matched;
    const st = !nWanted
      ? "Choose a benchmark GO set above, or paste a positive set here."
      : (matched < 2 || neg < 2)
        ? `${matched} IDRs matched — need at least 2 positives.`
        : `${matched} IDRs matched from ${nWanted} lines · ${matched} positive / ${neg} background`;
    applyLabels(lab, st);
  }

  // GO dropdown: build the label vector straight from the term's positive rows.
  function fromGo() {
    const t = go.terms.find((t) => t.go === goSelect.value);
    if (!t) return;
    positives.value = "";                      // a GO pick overrides the paste box
    const yv = new Float64Array(ids.length);
    for (let k = t.start; k < t.start + t.len; k++) yv[go.members[k]] = 1;
    applyLabels(yv, `${t.go} ${t.label} · ${t.proteins} proteins / ${t.idrs} IDRs`);
  }

  positives.addEventListener("input", debounce(fromTextarea, 300));
  slider.addEventListener("input", () => renderAt(+slider.value));
  if (decBtn) decBtn.addEventListener("click", () => renderAt(+slider.value - 1));
  if (incBtn) incBtn.addEventListener("click", () => renderAt(+slider.value + 1));
  // Enter commits and drops focus; Escape restores the current value; blur commits.
  countOut.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); countOut.blur(); }
    else if (e.key === "Escape") { e.preventDefault(); renderAt(+slider.value); countOut.blur(); }
  });
  countOut.addEventListener("blur", commitFeatureCount);
  window.addEventListener("resize", debounce(() => renderAt(+slider.value), 150));

  if (go && go.terms.length) {
    for (const t of go.terms) {
      const opt = document.createElement("option");
      opt.value = t.go;
      opt.textContent = `${t.go} ${t.label} — ${t.proteins} proteins / ${t.idrs} IDRs`;
      goSelect.appendChild(opt);
    }
    goSelect.addEventListener("change", fromGo);
    // default to nucleolus so the app is immediately demoing (matches the report)
    goSelect.value = go.terms.some((t) => t.go === "GO:0005730") ? "GO:0005730" : go.terms[0].go;
    fromGo();
  } else {
    if (goRow) goRow.style.display = "none";
    fromTextarea();
  }
}

function debounce(fn, ms) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
