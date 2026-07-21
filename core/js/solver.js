import { cholesky, cholInverse, cholSolve } from "./linalg.js";

// Numerical guards matching glmnet: clamp fitted probabilities away from 0/1 and
// floor the IRLS working weights, so near-separated sets stay finite.
const PMIN = 1e-5, PMAX = 1 - 1e-5, WMIN = 1e-5;

const soft = (a, t) => (a > t ? a - t : a < -t ? a + t : 0);

function sigmoid(x) {
  if (x >= 0) { const z = Math.exp(-x); return 1 / (1 + z); }
  const z = Math.exp(x); return z / (1 + z);
}

// Predict from fitted coefficients on any matrix with the same feature order
// used during fitting. `type: "link"` returns the linear predictor; the default
// returns response-scale probabilities via a stable sigmoid.
export function predictLogistic(X, beta0, beta, opts = {}) {
  const { type = "response" } = opts;
  const n = X.n, p = X.p;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let e = beta0;
    for (let j = 0; j < p; j++) e += X.cols[j][i] * beta[j];
    out[i] = type === "link" ? e : sigmoid(e);
  }
  return out;
}

// Standardize each column to mean 0, population sd 1 (glmnet's 1/n convention).
// Constant columns become all-zero and are skipped by the solver.
function standardize(cols, n) {
  const p = cols.length;
  const mean = new Float64Array(p), sd = new Float64Array(p), std = [];
  for (let j = 0; j < p; j++) {
    const c = cols[j];
    let m = 0; for (let i = 0; i < n; i++) m += c[i]; m /= n;
    let v = 0; for (let i = 0; i < n; i++) { const d = c[i] - m; v += d * d; } v /= n;
    const s = Math.sqrt(v);
    mean[j] = m; sd[j] = s;
    const sc = new Float64Array(n);
    if (s > 0) { const inv = 1 / s; for (let i = 0; i < n; i++) sc[i] = (c[i] - m) * inv; }
    std.push(sc);
  }
  return { std, mean, sd };
}

// Lasso-penalised logistic regression at a single lambda. Reproduces glmnet's
// algorithm: an outer IRLS quadratic approximation of the logistic loss, each
// solved by coordinate descent on the standardized design. Objective (α = 1):
//   (1/n) Σ [ -yᵢ ηᵢ + log(1+e^ηᵢ) ] + λ Σ|βⱼ|,  η = β0 + Xβ.
// Returns coefficients on the ORIGINAL feature scale.
export function fitLassoLogistic(X, y, lambda, opts = {}) {
  const { tol = 1e-7, maxOuter = 100, maxInner = 200, trace } = opts;
  const n = X.n, p = X.p;
  const { std, mean, sd } = standardize(X.cols, n);

  let ybar = 0; for (let i = 0; i < n; i++) ybar += y[i]; ybar /= n;
  let b0 = Math.log(ybar / (1 - ybar));
  const b = new Float64Array(p);                 // standardized-scale coefficients
  const eta = new Float64Array(n).fill(b0);

  const w = new Float64Array(n), r = new Float64Array(n), den = new Float64Array(p);

  for (let outer = 0; outer < maxOuter; outer++) {
    // IRLS: working weights w and residual r = z - eta = (y-p)/w around current eta.
    let wsum = 0;
    for (let i = 0; i < n; i++) {
      let pi = 1 / (1 + Math.exp(-eta[i]));
      if (pi < PMIN) pi = PMIN; else if (pi > PMAX) pi = PMAX;
      const wi = Math.max(pi * (1 - pi), WMIN);
      w[i] = wi; r[i] = (y[i] - pi) / wi; wsum += wi;
    }
    // Column denominators (1/n)Σ w x² are fixed within this IRLS step; precompute.
    for (let j = 0; j < p; j++) {
      if (sd[j] === 0) { den[j] = 0; continue; }
      const xj = std[j]; let d = 0;
      for (let i = 0; i < n; i++) { const wx = w[i] * xj[i]; d += wx * xj[i]; }
      den[j] = d;
    }
    // Coordinate descent on the penalized weighted least squares.
    for (let inner = 0; inner < maxInner; inner++) {
      let maxChange = 0;
      // intercept (unpenalized)
      let wr = 0; for (let i = 0; i < n; i++) wr += w[i] * r[i];
      const d0 = wr / wsum;
      if (d0 !== 0) { b0 += d0; for (let i = 0; i < n; i++) r[i] -= d0; }
      // features
      for (let j = 0; j < p; j++) {
        if (sd[j] === 0) continue;
        const xj = std[j];
        let num = 0; for (let i = 0; i < n; i++) num += w[i] * xj[i] * r[i];
        const a = (num + den[j] * b[j]) / n;
        const denN = den[j] / n;
        const bjNew = soft(a, lambda) / denN;
        const delta = bjNew - b[j];
        if (delta !== 0) {
          b[j] = bjNew;
          for (let i = 0; i < n; i++) r[i] -= xj[i] * delta;
          const ch = Math.abs(delta) * denN;
          if (ch > maxChange) maxChange = ch;
        }
      }
      if (maxChange < tol) break;
    }
    // Refresh eta from the updated coefficients; converge when it stops moving.
    let etaMax = 0;
    for (let i = 0; i < n; i++) {
      let e = b0;
      for (let j = 0; j < p; j++) if (b[j] !== 0) e += std[j][i] * b[j];
      const d = Math.abs(e - eta[i]); if (d > etaMax) etaMax = d;
      eta[i] = e;
    }
    // Optional instrumentation for the test suite: report the penalised objective's
    // two ingredients after each completed outer IRLS sweep. l1 is on the
    // standardized scale (what the penalty actually acts on); eta is the current
    // linear predictor, so the caller can form (1/n)Σ[-yη+log(1+e^η)] + λ·l1.
    if (trace) {
      let l1 = 0; for (let j = 0; j < p; j++) l1 += Math.abs(b[j]);
      trace(l1, Float64Array.from(eta));
    }
    if (etaMax < tol) break;
  }

  // Back-transform standardized coefficients to the original feature scale.
  const beta = new Float64Array(p);
  let beta0 = b0;
  for (let j = 0; j < p; j++) {
    if (b[j] !== 0 && sd[j] > 0) { const bo = b[j] / sd[j]; beta[j] = bo; beta0 -= bo * mean[j]; }
  }
  return { beta0, beta, features: X.features };
}

// Smallest λ that zeroes every coefficient: the largest null-model residual
// correlation on the standardized design, maxⱼ |(1/n) x̃ⱼᵀ(y−ȳ)|. Closed form, so
// the app can bound its penalty slider without a search. (There is no matching
// closed form for the lower end: use λ_min = ε·λ_max, ε≈1e-4 when n>p.)
// `argmax` is a free byproduct of the same scan: the coordinate whose correlation
// equals λ_max, i.e. the feature that enters first as λ drops below λ_max.
export function lambdaMax(X, y) {
  const n = X.n;
  let ybar = 0; for (let i = 0; i < n; i++) ybar += y[i]; ybar /= n;
  let lmax = 0, argmax = -1;
  for (let j = 0; j < X.p; j++) {
    const c = X.cols[j];
    let m = 0; for (let i = 0; i < n; i++) m += c[i]; m /= n;
    let v = 0, g = 0;
    for (let i = 0; i < n; i++) { const d = c[i] - m; v += d * d; g += d * (y[i] - ybar); }
    const sd = Math.sqrt(v / n);
    if (sd > 0) { const val = Math.abs(g / n / sd); if (val > lmax) { lmax = val; argmax = j; } }
  }
  return { lambdaMax: lmax, argmax };
}

// Unpenalised logistic regression on the given columns (original scale), by
// Newton-Raphson. Returns the intercept and, per feature, the coefficient and its
// Wald z = β / se, se from the diagonal of (XᵀWX)⁻¹ at convergence. This is the
// browser analogue of the paper's t-statistic.
export function refitLogistic(selCols, features, y, opts = {}) {
  const { tol = 1e-10, maxIter = 100 } = opts;
  const n = y.length, k = selCols.length, m = k + 1;   // + intercept
  const val = (j, i) => (j === 0 ? 1 : selCols[j - 1][i]);

  const beta = new Float64Array(m);
  let ybar = 0; for (let i = 0; i < n; i++) ybar += y[i]; ybar /= n;
  beta[0] = Math.log(ybar / (1 - ybar));

  let H = null;
  for (let it = 0; it < maxIter; it++) {
    const g = new Float64Array(m);
    H = Array.from({ length: m }, () => new Float64Array(m));
    for (let i = 0; i < n; i++) {
      let e = beta[0]; for (let j = 1; j < m; j++) e += selCols[j - 1][i] * beta[j];
      let pi = 1 / (1 + Math.exp(-e));
      if (pi < PMIN) pi = PMIN; else if (pi > PMAX) pi = PMAX;
      const wi = Math.max(pi * (1 - pi), WMIN), yi = y[i] - pi;
      for (let a = 0; a < m; a++) {
        const xa = val(a, i);
        g[a] += xa * yi;
        for (let bb = a; bb < m; bb++) H[a][bb] += xa * val(bb, i) * wi;
      }
    }
    for (let a = 0; a < m; a++) for (let bb = 0; bb < a; bb++) H[a][bb] = H[bb][a];
    const step = cholSolve(cholesky(H), g);
    let mx = 0;
    for (let a = 0; a < m; a++) { beta[a] += step[a]; const s = Math.abs(step[a]); if (s > mx) mx = s; }
    if (mx < tol) break;
  }

  const cov = cholInverse(cholesky(H));
  const refit = features.map((f, idx) => ({
    feature: f,
    coef: beta[idx + 1],
    waldZ: beta[idx + 1] / Math.sqrt(cov[idx + 1][idx + 1]),
  }));
  return { intercept: beta[0], refit, beta };
}
