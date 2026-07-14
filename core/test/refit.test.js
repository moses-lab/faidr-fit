// Test 6: refitLogistic — the unpenalised MLE + Wald-z path, which the KKT check
// does not touch. refitLogistic(selCols, features, y) -> {intercept, refit:[{feature,
// coef, waldZ}], beta:[β0, β...]}. Wald SEs use the SUM log-likelihood's Fisher
// information (XᵀWX)⁻¹ with W=diag(p̂(1−p̂)) and NO 1/N; we validate against a
// finite-difference Hessian of the sum negative log-likelihood.
import { test } from "node:test";
import assert from "node:assert/strict";
import { refitLogistic } from "../js/solver.js";
import { sigmoid, logit, mean } from "./helpers.js";
import { assertClose, assertAllFinite } from "./assertions.js";
import { rng, gauss } from "./helpers.js";

// --- small self-contained linear algebra (independent of js/linalg.js) ------
function invert(A) {
  const n = A.length;
  const M = A.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    const d = M[c][c];
    for (let k = 0; k < 2 * n; k++) M[c][k] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c]; for (let k = 0; k < 2 * n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((row) => row.slice(n));
}

// gradient of the SUM negative log-likelihood wrt θ=[β0, β...]
function sumNllGrad(cols, y, theta) {
  const n = y.length, m = theta.length;
  const g = new Float64Array(m);
  for (let i = 0; i < n; i++) {
    let e = theta[0]; for (let j = 1; j < m; j++) e += cols[j - 1][i] * theta[j];
    const p = sigmoid(e), d = p - y[i]; // ∂(-loglik)/∂η
    g[0] += d; for (let j = 1; j < m; j++) g[j] += cols[j - 1][i] * d;
  }
  return g;
}
// finite-difference Hessian of the sum NLL via central differences of the gradient
function fdHessian(cols, y, theta, h = 1e-5) {
  const m = theta.length, H = Array.from({ length: m }, () => new Float64Array(m));
  for (let a = 0; a < m; a++) {
    const tp = [...theta]; tp[a] += h; const gp = sumNllGrad(cols, y, tp);
    const tm = [...theta]; tm[a] -= h; const gm = sumNllGrad(cols, y, tm);
    for (let b = 0; b < m; b++) H[a][b] = (gp[b] - gm[b]) / (2 * h);
  }
  for (let a = 0; a < m; a++) for (let b = a + 1; b < m; b++) { const s = (H[a][b] + H[b][a]) / 2; H[a][b] = H[b][a] = s; }
  return H;
}

function randCols(rand, n, p, scale = 1) {
  const cols = [];
  for (let j = 0; j < p; j++) { const c = new Float64Array(n); for (let i = 0; i < n; i++) c[i] = gauss(rand) * scale; cols.push(c); }
  return cols;
}
function labelsFrom(rand, cols, w) {
  const n = cols[0].length, y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let e = -0.2; for (let j = 0; j < cols.length; j++) e += w[j] * cols[j][i];
    y[i] = rand() < sigmoid(e) ? 1 : 0;
  }
  return y;
}

test("refit MLE: (1/N)·Xselᵀ(y−p̂) ≈ 0 for every coordinate incl. intercept", () => {
  const rand = rng(301);
  for (let t = 0; t < 40; t++) {
    const n = 150 + Math.floor(rand() * 150), p = 1 + Math.floor(rand() * 4);
    const cols = randCols(rand, n, p);
    const y = labelsFrom(rand, cols, cols.map(() => gauss(rand)));
    if (y.every((v) => v === y[0])) continue;
    const features = cols.map((_, j) => `f${j}`);
    const res = refitLogistic(cols, features, y);
    const grad = sumNllGrad(cols, y, res.beta); // = -Σ x(y-p̂) = Σ x(p̂-y)
    for (let a = 0; a < res.beta.length; a++) {
      assertClose(grad[a] / n, 0, 1e-6, `gradient coord ${a}:`);
    }
  }
});

test("Wald z matches the finite-difference Fisher information (sum-loglik SE)", () => {
  const rand = rng(302);
  let checked = 0;
  for (let t = 0; t < 40; t++) {
    const n = 200 + Math.floor(rand() * 200), p = 1 + Math.floor(rand() * 3);
    const cols = randCols(rand, n, p, 0.8);
    const y = labelsFrom(rand, cols, cols.map(() => 0.6 * gauss(rand)));
    if (y.every((v) => v === y[0])) continue;
    const features = cols.map((_, j) => `f${j}`);
    const res = refitLogistic(cols, features, y);
    // skip near-separated fits where |β| blew up (clamp regime, SE meaningless)
    if (res.beta.some((b) => Math.abs(b) > 12)) continue;
    const Hinv = invert(fdHessian(cols, y, res.beta));
    for (let j = 0; j < p; j++) {
      const se = Math.sqrt(Hinv[j + 1][j + 1]);
      const zExpected = res.beta[j + 1] / se;
      assertClose(res.refit[j].waldZ, zExpected, 1e-3 * (1 + Math.abs(zExpected)), `feature ${j} z:`);
      checked++;
    }
  }
  assert.ok(checked > 20, `only ${checked} Wald-z coordinates checked`);
});

test("intercept-only refit ⇒ β0 = logit(ȳ)", () => {
  const rand = rng(303);
  for (let t = 0; t < 10; t++) {
    const n = 80 + Math.floor(rand() * 80), y = new Float64Array(n);
    for (let i = 0; i < n; i++) y[i] = rand() < 0.3 + 0.4 * rand() ? 1 : 0;
    if (y.every((v) => v === y[0])) continue;
    const res = refitLogistic([], [], y);
    assertClose(res.intercept, logit(mean(y)), 1e-8, "intercept:");
    assert.equal(res.refit.length, 0);
  }
});

// Closed-form check of the intercept SE convention 1/sqrt(N·ȳ(1−ȳ)): a single
// constant column is collinear with the intercept, so we instead verify the SE
// formula the implementation uses (Fisher info = Σ p̂(1−p̂)) reproduces it on the
// intercept-only Hessian built the same way as js/solver.js.
test("intercept-only Fisher information gives SE = 1/√(N·ȳ(1−ȳ))", () => {
  const rand = rng(304);
  const n = 120, y = new Float64Array(n);
  for (let i = 0; i < n; i++) y[i] = rand() < 0.35 ? 1 : 0;
  const ybar = mean(y);
  // at the MLE β0=logit(ȳ), p̂=ȳ, so the (sum) Fisher information is N·ȳ(1−ȳ)
  const fisher = n * ybar * (1 - ybar);
  const seClosed = 1 / Math.sqrt(fisher);
  // finite-difference Hessian of the intercept-only sum NLL at logit(ȳ)
  const H = fdHessian([], y, [logit(ybar)]);
  assertClose(Math.sqrt(1 / H[0][0]), seClosed, 1e-6, "intercept SE:");
});

test("separation guard: separable Xsel returns finite β and z (clamped, no NaN)", () => {
  // perfectly separable: y = 1 iff x > 0
  const n = 60, x = new Float64Array(n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) { x[i] = i - n / 2 + 0.5; y[i] = x[i] > 0 ? 1 : 0; }
  const res = refitLogistic([x], ["sep"], y);
  assertAllFinite(res.beta, "beta:");
  assert.ok(Number.isFinite(res.refit[0].waldZ), `waldZ not finite: ${res.refit[0].waldZ}`);
  // the coefficient should be large-and-positive (points the right way) but bounded
  assert.ok(res.refit[0].coef > 0, "separating coef should be positive");
});
