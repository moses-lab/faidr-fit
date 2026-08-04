// The app shell: turns the two inputs (a positive protein/IDR set, and a feature-
// count slider) into a fit and renders the feature-strength chart. The lasso path
// (up to DFMAX features) is fit once per label change; sliding just looks up a
// lambda already sitting on that path, so it's cheap array lookups, not refits.
//
// Predictions (the "Predict" panel below) are a second read of that same path
// fit: predictLogistic scores every IDR in the proteome from the penalised
// coefficients at the current lambda. This is deliberately NOT the unpenalised
// refit used for the bar chart -- that refit exists only to rank/sign features
// for display, its coefficients are never the scoring model. Because scores
// depend on slider position, predictions and the 1% FPR threshold are
// recomputed on every renderAt call, not just when the training set changes.
import { fitLassoLogistic, fitLogistic, predictLogistic } from "../core/src/index.js";
import { Clusterize } from "./clusterize.js";
import { renderBars } from "./chart.js";

const DFMAX = 30;  // widest feature count the path is fit out to
const TARGET_FPR = 0.01; // threshold is calibrated to this rate against training negatives

// Shared by the training-set textarea and the predict-query box: a line matches
// an id exactly, else as a prefix (so "Q5VUJ6" catches "Q5VUJ6_IDR_1_54").
function matchIds(text, ids) {
  const wanted = text.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  const exact = new Set(wanted);
  const idx = [];
  for (let i = 0; i < ids.length; i++) {
    if (exact.has(ids[i]) || wanted.some((w) => ids[i].startsWith(w))) idx.push(i);
  }
  return { idx, nWanted: wanted.length };
}

// Match a pasted list against the proteome, as a 0/1 label vector for training.
function buildLabels(text, ids) {
  const { idx, nWanted } = matchIds(text, ids);
  const y = new Float64Array(ids.length);
  for (const i of idx) y[i] = 1;
  return { y, matched: idx.length, nWanted };
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

// Apply sigmoid to an array of linear predictors.
function etaToProb(etaArr) {
  const p = new Float64Array(etaArr.length);
  for (let i = 0; i < etaArr.length; i++) p[i] = 1 / (1 + Math.exp(-etaArr[i]));
  return p;
}

// The probability past which only ~1% of the training negatives score higher --
// i.e. the cutoff giving a 1% false positive rate against this benchmark's own
// negative set. Sort negatives descending and take the value ~1% of the way in.
function negativeThreshold(p, yVec) {
  const negs = [];
  for (let i = 0; i < yVec.length; i++) if (yVec[i] === 0) negs.push(p[i]);
  if (!negs.length) return 1;
  negs.sort((a, b) => b - a);
  const k = Math.min(negs.length - 1, Math.max(0, Math.round(negs.length * TARGET_FPR) - 1));
  return negs[k];
}

export function startApp(proteome, go, root = document) {
  const { X, ids, labels, descs } = proteome;
  const el = (id) => root.getElementById(id);
  const positives = el("positives"), slider = el("features"), chart = el("chart");
  const status = el("status"), countOut = el("feature-count");
  const goSelect = el("go-select"), goRow = el("go-row");
  const decBtn = el("feat-dec"), incBtn = el("feat-inc");
  const predictQuery = el("predict-query"), downloadBtn = el("download-tsv");
  const predictClusterize = new Clusterize({
    scrollId: "predict-scroll",
    contentId: "predict-tbody",
    rows: [],
    show_no_data_row: false,
  });

  // State from the most recent path fit. counts/firstStep let the slider (a
  // plain index 0..counts.length-1) resolve to a real step on the path.
  // currentProb/currentThreshold are the prediction-side state, recomputed on
  // every renderAt call; predIds/predText belong to the predict-query box and
  // are independent of the training set, so a lookup survives switching
  // benchmarks (only the scores against it change).
  let y = null, path = null, lambdaPath = [], coefficients = [], counts = [], firstStep = new Map();
  let currentProb = null, currentThreshold = null;

  // Full-proteome index list, reused whenever the predict-query box is empty
  // so the table defaults to showing every IDR rather than nothing.
  const allIdx = new Array(ids.length);
  for (let i = 0; i < ids.length; i++) allIdx[i] = i;

  let predIds = allIdx, predText = "";

  function setSliderEnabled(enabled) {
    slider.disabled = !enabled;
    if (decBtn) decBtn.disabled = !enabled;
    if (incBtn) incBtn.disabled = !enabled;
  }

  function renderComputeTime(ms) {
    const out = el("compute-time");
    if (!out) return;
    if (!Number.isFinite(ms)) { out.textContent = ""; return; }
    out.textContent = `· computed in ${ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`}`;
  }

  function renderThresholdReadout() {
    const out = el("predict-threshold");
    if (!out) return;
    if (currentThreshold == null) { out.textContent = ""; return; }
    let neg = 0;
    for (let i = 0; i < y.length; i++) if (y[i] === 0) neg++;
    out.textContent = `Signal threshold: p >= ${currentThreshold.toFixed(3)} ` +
      `(${(TARGET_FPR * 100).toFixed(0)}% FPR against ${neg.toLocaleString()} training negatives)`;
  }

  function renderPredictTable() {
    const tbody = el("predict-tbody"), predStatus = el("predict-status");
    if (!tbody || !predStatus) return;
    if (!currentProb) {
      predStatus.textContent = "Choose a benchmark or paste a training set above to enable predictions.";
      if (downloadBtn) downloadBtn.disabled = true;
      predictClusterize.update([]);
      return;
    }
    if (downloadBtn) downloadBtn.disabled = false;

    if (!predIds.length) {
      predStatus.textContent = "No matching IDRs in the proteome.";
      predictClusterize.update([]);
      return;
    }

    const rows = predIds
      .map((i) => ({ id: ids[i], p: currentProb[i] }))
      .sort((a, b) => b.p - a.p); // most confident first

    let signal = 0;
    for (const r of rows) if (r.p >= currentThreshold) signal++;
    const noSignal = rows.length - signal;
    const pct = ((signal / rows.length) * 100).toFixed(1);
    const scope = predText.trim() ? "matching" : "total";
    predStatus.textContent =
      `${rows.length.toLocaleString()} ${scope} IDR${rows.length === 1 ? "" : "s"} — ` +
      `${signal.toLocaleString()} signal / ${noSignal.toLocaleString()} no signal (${pct}% signal)`;

    const htmlRows = rows.map((r) => {
      const pass = r.p >= currentThreshold;
      return `<tr><td>${r.id}</td><td>${r.p.toFixed(3)}</td>` +
        `<td class="call ${pass ? "yes" : "no"}">${pass ? "Signal" : "No signal"}</td></tr>`;
    });
    predictClusterize.update(htmlRows);
  }

  // Exports whatever's currently in the predict table: the full proteome
  // when the query box is empty, or just the matched rows when it's
  // filtered. Model state is recorded as a header comment so a file
  // downloaded now isn't silently ambiguous with one from another slider
  // position later.
  function downloadPredictions() {
    if (!currentProb || !path) return;
    const featureCount = counts[+slider.value];
    const pathIdx = firstStep.get(featureCount);
    const lambda = lambdaPath[pathIdx];
    const lines = [
      `# FAIDR predictions -- feature_count=${featureCount} lambda=${lambda.toExponential(4)} ` +
        `threshold=${currentThreshold.toFixed(6)} target_fpr=${TARGET_FPR}`,
      "id\tprobability\tabove_threshold",
    ];
    for (const i of predIds) {
      lines.push(`${ids[i]}\t${currentProb[i].toFixed(6)}\t${currentProb[i] >= currentThreshold ? 1 : 0}`);
    }
    const blob = new Blob([lines.join("\n")], { type: "text/tab-separated-values" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `faidr-predictions-${featureCount}features.tsv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function renderAt(idx) {
    if (!counts.length) return;
    idx = Math.max(0, Math.min(counts.length - 1, Math.round(idx)));
    slider.value = String(idx);

    const featureCount = counts[idx];
    const pathIdx = firstStep.get(featureCount);
    const beta = coefficients[pathIdx].beta;

    const sel = [];
    for (let j = 0; j < X.p; j++) if (beta[j] !== 0) sel.push(j);
    countOut.textContent = String(sel.length);

    if (sel.length) {
      // unpenalised refit on the selected features for signed significance (Wald z)
      const selCols = sel.map((j) => X.cols[j]);
      const { waldZ } = fitLogistic(selCols, y);
      const items = sel
        .map((j, k) => ({ feature: labels[j], raw: X.features[j], desc: descs[j], value: waldZ[k] }))
        .sort((a, b) => b.value - a.value);                    // display enriched → depleted
      renderBars(chart, items);
    } else {
      renderBars(chart, []);
    }

    // Prediction scores come from the penalised path fit itself, at this same
    // lambda step -- never from the unpenalised refit above.
    const { eta } = predictLogistic(path, X.cols, pathIdx);
    currentProb = etaToProb(eta);
    currentThreshold = negativeThreshold(currentProb, y);
    renderThresholdReadout();
    renderPredictTable();
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
    const t0 = performance.now();
    let pos = 0; for (let i = 0; i < labelVector.length; i++) pos += labelVector[i];
    const neg = ids.length - pos;
    status.textContent = statusText;
    if (pos < 2 || neg < 2) {
      y = null; path = null; counts = []; firstStep = new Map();
      currentProb = null; currentThreshold = null;
      renderBars(chart, []); countOut.textContent = "0";
      slider.min = "0"; slider.max = "0"; slider.value = "0";
      setSliderEnabled(false);
      renderComputeTime(null);
      renderThresholdReadout();
      renderPredictTable();
      return;
    }
    y = labelVector;
    path = fitLassoLogistic(X.cols, y, { dfmax: DFMAX, nlambda: 150 });
    lambdaPath = path.lambdaPath;
    coefficients = path.coefficients;
    ({ counts, firstStep } = distinctCounts(path.df));

    slider.min = "0";
    slider.max = String(counts.length - 1);
    slider.step = "1";
    setSliderEnabled(true);
    renderAt(Math.floor((counts.length - 1) / 2)); // start roughly mid-range
    renderComputeTime(performance.now() - t0);
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

  // GO dropdown: build the label vector from the term's positive rows, and
  // write those same ids into the paste box so the user can see and tweak
  // them. Setting .value here does not fire the textarea's "input" event, so
  // goSelect stays on this term until the user actually edits the text --
  // at which point fromTextarea takes over and re-matches against their edit.
  function fromGo() {
    const t = go.terms.find((t) => t.go === goSelect.value);
    if (!t) return;
    const idList = [];
    const yv = new Float64Array(ids.length);
    for (let k = t.start; k < t.start + t.len; k++) {
      const i = go.members[k];
      yv[i] = 1;
      idList.push(ids[i]);
    }
    positives.value = idList.join("\n");
    applyLabels(yv, `${t.go} ${t.label} · ${t.proteins} proteins / ${t.idrs} IDRs`);
    fromPredictQuery();
  }

  // Predict-query box: independent of the training set, so it survives
  // switching benchmarks. Only the match list is recomputed here; the scores
  // shown against it come from currentProb, kept fresh by renderAt.
  function fromPredictQuery() {
    predText = predictQuery.value;
    predIds = predText.trim() ? matchIds(predText, ids).idx : allIdx;
    renderPredictTable();
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
  if (predictQuery) predictQuery.addEventListener("input", debounce(fromPredictQuery, 300));
  if (downloadBtn) downloadBtn.addEventListener("click", downloadPredictions);

  if (go && go.terms.length) {
    for (const t of go.terms) {
      const opt = document.createElement("option");
      opt.value = t.go;
      opt.textContent = `${t.go} ${t.label} — ${t.proteins} proteins / ${t.idrs} IDRs`;
      goSelect.appendChild(opt);
    }
    goSelect.addEventListener("change", fromGo);
    if (positives.value.trim()) {
      fromTextarea();
      fromPredictQuery();
    } else {
      // default to nucleolus so the app is immediately demoing (matches the report)
      goSelect.value = go.terms.some((t) => t.go === "GO:0005730") ? "GO:0005730" : go.terms[0].go;
      fromGo();
    }
  } else {
    if (goRow) goRow.style.display = "none";
    fromTextarea();
    fromPredictQuery();
  }
}

function debounce(fn, ms) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
