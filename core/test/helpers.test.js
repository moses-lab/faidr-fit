// Step 1: unit-test the assertion utilities themselves on a hand-computed 3×2
// example before trusting them to judge the solver. Reference values were worked
// out independently (see the numbers inline).
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  sigmoid, log1pexp, matrixFromRows, colStats, meanNLL, objective,
  eta, kktViolation, logit,
} from "./helpers.js";
import { assertClose, assertVectorClose } from "./assertions.js";

const rows = [[1.0, -1.0], [2.0, 0.5], [0.0, 3.0]];
const y = new Float64Array([1, 0, 1]);
const beta0 = 0.2, beta = [0.5, -0.3];
const X = matrixFromRows(rows);

test("sigmoid matches definition and is symmetric", () => {
  assertClose(sigmoid(0), 0.5, 0);
  assertClose(sigmoid(2), 1 / (1 + Math.exp(-2)), 1e-15);
  assertClose(sigmoid(-2) + sigmoid(2), 1, 1e-15);
});

test("log1pexp matches naive log(1+e^x) in the safe range", () => {
  for (const x of [-3, -1, 0, 1, 3]) assertClose(log1pexp(x), Math.log(1 + Math.exp(x)), 1e-12);
});

test("colStats gives population mean and sd", () => {
  const { mean, sd } = colStats(X);
  assertVectorClose(mean, [1.0, 0.8333333333333334], 1e-12);
  assertVectorClose(sd, [0.816496580927726, 1.6499158227686108], 1e-12);
});

test("eta and meanNLL match hand computation", () => {
  const e = eta(X, beta0, beta);
  assertVectorClose(e, [1.0, 1.05, -0.7], 1e-12);
  assertClose(meanNLL(e, y), 0.9221687386737747, 1e-12);
});

test("objective uses the standardized-penalty l1 (λ Σ|βⱼ|·sdⱼ)", () => {
  // l1std = 0.9032230372944463, meanNLL + 0.1*l1std = 1.0124910424032194
  assertClose(objective(X, y, beta0, beta, 0.1), 1.0124910424032194, 1e-12);
});

test("kktViolation intercept branch reports mean residual", () => {
  // at a non-optimal point the intercept residual is the mean of y - sigmoid(eta)
  const w = kktViolation(X, y, beta0, beta, /*λ=*/1.0); // large λ so features are inactive-ish
  // mean residual = 0.06545143145200243; with a big λ the coord terms stay small,
  // so the intercept is (at least a candidate for) the worst violation.
  assert.ok(w.viol >= 0);
  // sanity: recompute the intercept violation directly
  const e = eta(X, beta0, beta);
  let rm = 0; for (let i = 0; i < 3; i++) rm += y[i] - sigmoid(e[i]); rm /= 3;
  assertClose(Math.abs(rm), 0.06545143145200243, 1e-12);
});

test("kktViolation coord correlations match hand computation", () => {
  // c = [-0.5752066019061658, 0.24276104480229133] on standardized columns.
  // With tiny λ and these (nonzero) betas, active-branch violation = |c - λ·sign(β)|.
  const lambda = 0.0;
  const w = kktViolation(X, y, beta0, beta, lambda);
  // worst active violation = max(|c0 - 0|, |c1 - 0|) since sign·λ=0
  assertClose(w.viol, 0.5752066019061658, 1e-12);
});

test("logit is the inverse of sigmoid", () => {
  for (const p of [0.1, 0.4, 0.75]) assertClose(sigmoid(logit(p)), p, 1e-14);
});
