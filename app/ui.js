// The app shell: turns the two inputs (a positive protein/IDR set, and a feature-
// count slider) into a fit and renders the feature-strength chart. The lasso path
// (up to DFMAX features) is fit once per label change; sliding just looks up a
// lambda already sitting on that path, so it's cheap array lookups, not refits.
//
// Predictions are a second read of that same path fit: predictLogistic scores
// every IDR in the proteome from the penalised coefficients at the current
// lambda. This is deliberately NOT the unpenalised refit used for the bar
// chart -- that refit exists only to rank/sign features for display, its
// coefficients are never the scoring model. Because scores depend on slider
// position, predictions and the 1% FPR threshold are recomputed on every
// renderAt call, not just when the training set changes.
import { fitLassoLogistic, fitLogistic, predictLogistic } from "../core/src/index.js";
import { renderBars } from "./chart.js";

const DFMAX = 30;  // widest feature count the path is fit out to
const TARGET_FPR = 0.01; // threshold is calibrated to this rate against training negatives

// A line matches an id exactly, else as a prefix (so "Q5VUJ6" catches "Q5VUJ6_IDR_1_54").
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

function formatMs(ms) {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

// Apparent (in-sample) recall: fraction of the label-1 rows scoring at or
// above the threshold. This is an optimistic upper bound, not a
// generalisation estimate -- the model was fit on this same y.
function apparentRecall(prob, yVec, threshold) {
  let pos = 0, hit = 0;
  for (let i = 0; i < yVec.length; i++) {
    if (yVec[i] === 1) {
      pos++;
      if (prob[i] >= threshold) hit++;
    }
  }
  return pos ? hit / pos : 0;
}

// Apparent precision (PPV) at the threshold, plus enrichment over the base
// rate -- how many times better than flagging the same number of IDRs at
// random from the full set. At low prevalence (most GO terms), a high
// specificity can still come with modest precision because the negative
// pool is so much larger than the positive one.
function apparentPrecision(prob, yVec, threshold) {
  let pos = 0, predPos = 0, truePos = 0;
  for (let i = 0; i < yVec.length; i++) {
    if (yVec[i] === 1) pos++;
    if (prob[i] >= threshold) {
      predPos++;
      if (yVec[i] === 1) truePos++;
    }
  }
  const precision = predPos ? truePos / predPos : 0;
  const baseRate = yVec.length ? pos / yVec.length : 0;
  const enrichment = baseRate ? precision / baseRate : 0;
  return { precision, enrichment };
}

export function startApp(proteome, go, root = document) {
  const { X, ids, labels, descs } = proteome;
  const el = (id) => root.getElementById(id);
  const positives = el("positives"), slider = el("features"), chart = el("chart");
  const status = el("status"), countOut = el("feature-count");
  const goSelect = el("go-select"), goRow = el("go-row");
  const decBtn = el("feat-dec"), incBtn = el("feat-inc");
  const resultsBody = el("results-body");
  const downloadBtn = el("download-tsv");

  // State from the most recent path fit. counts/firstStep let the slider (a
  // plain index 0..counts.length-1) resolve to a real step on the path.
  // currentProb/currentThreshold are the prediction-side state, recomputed on
  // every renderAt call.
  let y = null, path = null, lambdaPath = [], coefficients = [], counts = [], firstStep = new Map();
  let currentProb = null, currentThreshold = null;

  function setSliderEnabled(enabled) {
    slider.disabled = !enabled;
    if (decBtn) decBtn.disabled = !enabled;
    if (incBtn) incBtn.disabled = !enabled;
  }

  // Grey out everything downstream of the inputs while a fit is running --
  // it's stale until the fit finishes, and this also blocks interaction
  // during the one frame we yield for "Computing…" to paint.
  function setComputingState(computing) {
    if (resultsBody) resultsBody.classList.toggle("computing", computing);
    if (chart) chart.classList.toggle("computing", computing);
  }

  function renderPredictSummary() {
    const summary = el("predict-status");
    if (!summary) return;
    if (!currentProb) {
      summary.textContent = "";
      if (downloadBtn) downloadBtn.disabled = true;
      return;
    }
    summary.textContent = `${ids.length.toLocaleString()} IDR predictions are available for download.`;
    if (downloadBtn) downloadBtn.disabled = false;
  }

  // Both are apparent (in-sample) metrics against the same y the model was
  // fit on -- recall and precision reported here are optimistic, see report
  // for discussion. Precision comes with a fold-enrichment over random
  // guessing at this positive set's base rate.
  function renderMetrics() {
    const recallOut = el("predict-recall");
    const precisionOut = el("predict-precision");
    if (!recallOut || !precisionOut) return;
    if (!currentProb || !y) {
      recallOut.textContent = "Recall: —";
      precisionOut.textContent = "Precision (÷Relevance): —";
      return;
    }
    const recall = apparentRecall(currentProb, y, currentThreshold);
    const { precision, enrichment } = apparentPrecision(currentProb, y, currentThreshold);
    recallOut.textContent = `Recall: ${(recall * 100).toFixed(1)}%`;
    precisionOut.textContent = `Precision (÷Relevance): ${(precision * 100).toFixed(1)}% (${enrichment.toFixed(1)}×)`;
  }

  // Exports predictions for the full proteome at the current slider position.
  // Model state is recorded as a header comment so a file downloaded now
  // isn't silently ambiguous with one from another slider position later.
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
    for (let i = 0; i < ids.length; i++) {
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
    renderPredictSummary();
    renderMetrics();
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
  // (up to DFMAX features), and render a default step. The fit is synchronous
  // and can take a moment, so it's deferred a frame -- just long enough for
  // "Computing…" to actually paint before it blocks the main thread.
  function applyLabels(labelVector, statusText) {
    let pos = 0; for (let i = 0; i < labelVector.length; i++) pos += labelVector[i];
    const neg = ids.length - pos;

    if (pos < 2 || neg < 2) {
      status.textContent = statusText;
      y = null; path = null; counts = []; firstStep = new Map();
      currentProb = null; currentThreshold = null;
      renderBars(chart, []); countOut.textContent = "0";
      slider.min = "0"; slider.max = "0"; slider.value = "0";
      setSliderEnabled(false);
      setComputingState(false);
      renderPredictSummary();
      renderMetrics();
      return;
    }

    status.textContent = "Computing…";
    setComputingState(true);
    requestAnimationFrame(() => setTimeout(() => {
      const t0 = performance.now();
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

      status.textContent = `${statusText} · computed in ${formatMs(performance.now() - t0)}`;
      setComputingState(false);
    }, 0));
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
    } else {
      // default to nucleolus so the app is immediately demoing (matches the report)
      goSelect.value = go.terms.some((t) => t.go === "GO:0005730") ? "GO:0005730" : go.terms[0].go;
      fromGo();
    }
  } else {
    if (goRow) goRow.style.display = "none";
    fromTextarea();
  }
}

function debounce(fn, ms) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
