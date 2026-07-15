// Degenerate inputs and numerical-safety cases. Cheap checks, mostly routed
// through the KKT assertion; plus a brute-force 1-D grid-search oracle.
import { test } from "node:test";
import assert from "node:assert/strict";
import { fitLassoLogistic, lambdaMax } from "../js/solver.js";
import {
  sigmoid, log1pexp, matrixFromRows, objective, logit, mean,
} from "./helpers.js";
import { assertKKT, assertClose, assertAllFinite } from "./assertions.js";
import { rng, gauss } from "./helpers.js";

test("zero-variance column ⇒ its coefficient is exactly 0, no NaN, KKT holds", () => {
  const rand = rng(401);
  const n = 60, rows = [];
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = gauss(rand);
    rows.push([a, 5.0, gauss(rand)]); // col 1 is constant
    y[i] = a + 0.3 * gauss(rand) > 0 ? 1 : 0;
  }
  const X = matrixFromRows(rows);
  const { lambdaMax: lm } = lambdaMax(X, y);
  const lambda = lm * 0.1;
  const res = fitLassoLogistic(X, y, lambda);
  assertAllFinite(res.beta, "beta:");
  assert.equal(res.beta[1], 0, "constant column must get coefficient 0");
  assertKKT(X, y, res, lambda, 1e-5, "zero-variance:");
});

test("duplicated columns ⇒ KKT holds and tied coefficients share the load", () => {
  const rand = rng(402);
  const n = 80, rows = [];
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = gauss(rand), b = gauss(rand);
    rows.push([a, a, b]); // cols 0 and 1 identical
    y[i] = a - 0.5 * b + 0.3 * gauss(rand) > 0 ? 1 : 0;
  }
  const X = matrixFromRows(rows);
  const { lambdaMax: lm } = lambdaMax(X, y);
  for (const frac of [0.4, 0.1]) {
    const lambda = lm * frac;
    const res = fitLassoLogistic(X, y, lambda);
    // individual split between the tied columns is non-unique; only assert KKT
    // (which uses standardized columns, so the tied pair is handled jointly)
    assertKKT(X, y, res, lambda, 1e-5, `dup-col frac=${frac}:`);
    assertAllFinite(res.beta, "beta:");
  }
});

test("all-zero / all-one labels ⇒ β0 = ∓∞ (documented), feature coefs finite & 0", () => {
  const rand = rng(403);
  const n = 30, rows = [];
  for (let i = 0; i < n; i++) rows.push([gauss(rand), gauss(rand)]);
  const X = matrixFromRows(rows);
  const zero = fitLassoLogistic(X, new Float64Array(n), 0.05);
  assert.equal(zero.beta0, -Infinity, "all-zero labels ⇒ β0 = -∞");
  assertAllFinite(zero.beta, "all-zero features:");
  assert.ok([...zero.beta].every((v) => v === 0), "no feature should activate");
  const one = fitLassoLogistic(X, new Float64Array(n).fill(1), 0.05);
  assert.equal(one.beta0, Infinity, "all-one labels ⇒ β0 = +∞");
  assertAllFinite(one.beta, "all-one features:");
});

test("perfect separation with λ>0 ⇒ finite solution, KKT holds, no overflow", () => {
  const n = 40, rows = [], y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = i - n / 2 + 0.5;
    rows.push([x, Math.sin(i)]); // x perfectly separates y
    y[i] = x > 0 ? 1 : 0;
  }
  const X = matrixFromRows(rows);
  const { lambdaMax: lm } = lambdaMax(X, y);
  const lambda = lm * 0.2;
  const res = fitLassoLogistic(X, y, lambda);
  assertAllFinite(res.beta, "beta:");
  assert.ok(Number.isFinite(res.beta0), "β0 finite");
  assertKKT(X, y, res, lambda, 1e-5, "separable:");
});

test("tiny problems: N=1 and p=1 stay finite and KKT-valid (where defined)", () => {
  // p=1, small N, non-degenerate labels
  const X = matrixFromRows([[-1.5], [0.4], [1.1], [2.0], [-0.7]]);
  const y = new Float64Array([0, 0, 1, 1, 0]);
  const { lambdaMax: lm } = lambdaMax(X, y);
  for (const frac of [0.7, 0.2, 0.02]) {
    const res = fitLassoLogistic(X, y, lm * frac);
    assertAllFinite(res.beta, `p=1 frac=${frac}:`);
    assertKKT(X, y, res, lm * frac, 1e-5, `p=1 frac=${frac}:`);
  }
  // N=1: a single sample; solver must not crash or NaN
  const X1 = matrixFromRows([[0.5, -0.3]]);
  const r1 = fitLassoLogistic(X1, new Float64Array([1]), 0.1);
  assertAllFinite(r1.beta, "N=1:");
});

// Independent oracle for the 1-D case: brute-force minimise the true objective on
// a fine (β0, β) grid and confirm the solver reaches an objective no worse than
// the best grid point (it should find the global optimum of this convex problem).
test("1-D grid-search oracle: solver objective ≤ best grid objective", () => {
  const rand = rng(404);
  for (let t = 0; t < 8; t++) {
    const n = 25 + Math.floor(rand() * 25), rows = [], y = new Float64Array(n);
    const scale = Math.exp((rand() - 0.5) * 2);
    for (let i = 0; i < n; i++) { const x = gauss(rand) * scale; rows.push([x]); y[i] = x / scale + 0.4 * gauss(rand) > 0 ? 1 : 0; }
    if (y.every((v) => v === y[0])) continue;
    const X = matrixFromRows(rows);
    const { lambdaMax: lm } = lambdaMax(X, y);
    const lambda = lm * (0.05 + rand() * 0.5);
    const res = fitLassoLogistic(X, y, lambda);
    const solverObj = objective(X, y, res.beta0, res.beta, lambda);
    // grid centred generously around the solution
    let best = Infinity;
    for (let a = -6; a <= 6; a += 0.05) {
      for (let bb = -8 / scale; bb <= 8 / scale; bb += 0.05 / scale) {
        const o = objective(X, y, a, [bb], lambda);
        if (o < best) best = o;
      }
    }
    assert.ok(
      solverObj <= best + 1e-3,
      `solver objective ${solverObj} worse than grid best ${best} (λ=${lambda})`,
    );
  }
});

// ---- numerical safety -----------------------------------------------------
test("overflow-safe sigmoid / log1pexp at x = ±1000", () => {
  assert.equal(sigmoid(1000), 1);
  assert.equal(sigmoid(-1000), 0);
  assert.ok(Number.isFinite(sigmoid(1000)) && Number.isFinite(sigmoid(-1000)));
  assertClose(log1pexp(1000), 1000, 1e-9); // log(1+e^1000) ≈ 1000
  assertClose(log1pexp(-1000), 0, 1e-9);
  assert.ok(Number.isFinite(log1pexp(1000)));
});

test("extreme-scale design produces no NaN in the fit", () => {
  const rand = rng(405);
  const n = 50, rows = [], y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    rows.push([gauss(rand) * 1e6, gauss(rand) * 1e-6]); // wildly mismatched scales
    y[i] = gauss(rand) > 0 ? 1 : 0;
  }
  const X = matrixFromRows(rows);
  const { lambdaMax: lm } = lambdaMax(X, y);
  const res = fitLassoLogistic(X, y, lm * 0.1);
  assertAllFinite(res.beta, "extreme-scale beta:");
  assert.ok(Number.isFinite(res.beta0), "extreme-scale β0 finite");
});

test("separable IRLS keeps working weights finite (no NaN from p(1−p) floor)", () => {
  const n = 50, rows = [], y = new Float64Array(n);
  for (let i = 0; i < n; i++) { const x = i - n / 2 + 0.5; rows.push([x]); y[i] = x > 0 ? 1 : 0; }
  const X = matrixFromRows(rows);
  // very small λ pushes toward the (separating) MLE where p̂→0/1; must stay finite
  const res = fitLassoLogistic(X, y, 1e-4);
  assertAllFinite(res.beta, "separable small-λ beta:");
  assert.ok(Number.isFinite(res.beta0));
});
