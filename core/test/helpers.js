// Self-contained test utilities for the lasso-logistic kernel. No external deps,
// no glmnet/CVXPY oracle: correctness is judged by mathematical invariants only.
//
// The solver standardizes each column internally (mean 0, population sd 1) and
// applies the L1 penalty to the STANDARDIZED coefficients. So the objective it
// minimises, written on the original feature scale, is
//   f(β0, β) = (1/n) Σ [ -yᵢ ηᵢ + log(1+e^ηᵢ) ] + λ Σⱼ |βⱼ|·sdⱼ,
// with η = β0 + Xβ. The λ Σ|βⱼ|·sdⱼ term (not λ Σ|βⱼ|) is what makes the KKT and
// λ_max checks below match the implementation. Every helper here bakes in that
// standardized-penalty convention.

// ---- deterministic PRNG (mulberry32) -------------------------------------
export function rng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// standard normal via Box-Muller, driven by a uniform PRNG
export function gauss(rand) {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---- overflow-safe primitives --------------------------------------------
export function sigmoid(x) {
  // branch on sign so we never exponentiate a large positive number
  if (x >= 0) { const z = Math.exp(-x); return 1 / (1 + z); }
  const z = Math.exp(x); return z / (1 + z);
}
// log(1 + e^x), stable for |x| large
export function log1pexp(x) {
  if (x > 0) return x + Math.log1p(Math.exp(-x));
  return Math.log1p(Math.exp(x));
}

// ---- matrix container matching the solver's X = {n, p, cols, features} ----
// rows: array of length-n rows, each an array of p numbers.
export function matrixFromRows(rows, features) {
  const n = rows.length, p = n ? rows[0].length : 0;
  const cols = [];
  for (let j = 0; j < p; j++) {
    const c = new Float64Array(n);
    for (let i = 0; i < n; i++) c[i] = rows[i][j];
    cols.push(c);
  }
  return { n, p, cols, features: features || cols.map((_, j) => `f${j}`) };
}

// population mean/sd per column (the solver's 1/n convention)
export function colStats(X) {
  const { n, p, cols } = X;
  const mean = new Float64Array(p), sd = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    const c = cols[j];
    let m = 0; for (let i = 0; i < n; i++) m += c[i]; m /= n;
    let v = 0; for (let i = 0; i < n; i++) { const d = c[i] - m; v += d * d; } v /= n;
    mean[j] = m; sd[j] = Math.sqrt(v);
  }
  return { mean, sd };
}

// standardized columns (constant columns -> all zero, as the solver treats them)
export function standardizedCols(X) {
  const { n, p, cols } = X;
  const { mean, sd } = colStats(X);
  const std = [];
  for (let j = 0; j < p; j++) {
    const c = cols[j], s = new Float64Array(n);
    if (sd[j] > 0) { const inv = 1 / sd[j]; for (let i = 0; i < n; i++) s[i] = (c[i] - mean[j]) * inv; }
    std.push(s);
  }
  return { std, mean, sd };
}

export function eta(X, beta0, beta) {
  const { n, p, cols } = X;
  const e = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = beta0;
    for (let j = 0; j < p; j++) s += cols[j][i] * beta[j];
    e[i] = s;
  }
  return e;
}

// mean negative log-likelihood (1/n)Σ[-yη + log(1+e^η)]
export function meanNLL(etaArr, y) {
  const n = y.length; let s = 0;
  for (let i = 0; i < n; i++) s += -y[i] * etaArr[i] + log1pexp(etaArr[i]);
  return s / n;
}

// full penalised objective on the standardized-penalty scale
export function objective(X, y, beta0, beta, lambda) {
  const { sd } = colStats(X);
  let l1 = 0; for (let j = 0; j < X.p; j++) l1 += Math.abs(beta[j]) * sd[j];
  return meanNLL(eta(X, beta0, beta), y) + lambda * l1;
}

// ---- KKT stationarity of the penalised objective (Test 1) -----------------
// Gradient wrt the standardized coefficient bⱼ is -(1/n) x̃ⱼᵀ r, so at the optimum
//   active  (βⱼ≠0):  (1/n) x̃ⱼᵀ r  =  λ·sign(βⱼ)
//   inactive(βⱼ=0):  |(1/n) x̃ⱼᵀ r| ≤ λ
// with r = y - p̂. Intercept (unpenalised): mean(r) = 0. Returns the worst
// violation and where it occurred so failures point at the culprit coordinate.
export function kktViolation(X, y, beta0, beta, lambda, tiny = 1e-8) {
  const { n } = X;
  const { std } = standardizedCols(X);
  const e = eta(X, beta0, beta);
  const r = new Float64Array(n);
  let rmean = 0;
  for (let i = 0; i < n; i++) { r[i] = y[i] - sigmoid(e[i]); rmean += r[i]; }
  rmean /= n;

  let worst = { where: "intercept", viol: Math.abs(rmean), c: rmean };
  for (let j = 0; j < X.p; j++) {
    const xj = std[j];
    let c = 0; for (let i = 0; i < n; i++) c += xj[i] * r[i]; c /= n;
    let viol;
    if (Math.abs(beta[j]) >= tiny) {
      viol = Math.abs(c - lambda * Math.sign(beta[j]));
    } else {
      // boundary-safe: a (near-)zero coord satisfies KKT if it looks either
      // inactive (|c|≤λ) or exactly on the active boundary (|c|=λ); take the min.
      const inactive = Math.max(0, Math.abs(c) - lambda);
      const active = Math.abs(Math.abs(c) - lambda);
      viol = Math.min(inactive, active);
    }
    if (viol > worst.viol) worst = { where: j, viol, c };
  }
  return worst;
}

// λ_max: smallest λ that zeroes every coefficient. Gradient at the null model
// (β=0, β0=logit(ȳ), so p̂=ȳ) wrt the standardized coord j is (1/n) x̃ⱼᵀ(y-ȳ).
export function lambdaMax(X, y) {
  const { n } = X;
  const { std } = standardizedCols(X);
  let ybar = 0; for (let i = 0; i < n; i++) ybar += y[i]; ybar /= n;
  let lmax = 0, arg = -1;
  for (let j = 0; j < X.p; j++) {
    const xj = std[j];
    let c = 0; for (let i = 0; i < n; i++) c += xj[i] * (y[i] - ybar); c /= n;
    if (Math.abs(c) > lmax) { lmax = Math.abs(c); arg = j; }
  }
  return { lambdaMax: lmax, argmax: arg };
}

export function logit(p) { return Math.log(p / (1 - p)); }
export const mean = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
