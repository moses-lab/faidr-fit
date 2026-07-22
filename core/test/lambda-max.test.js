// Test 2: λ_max closed form. Exercises the soft-threshold boundary and intercept
// handling directly. At λ just above λ_max the whole solution collapses to the
// null model; just below, exactly the argmax coordinate switches on.
import { test } from "node:test";
import assert from "node:assert/strict";
import { fitLassoLogistic, lambdaMax } from "../js/solver.js";
import { logit, mean, matrixFromRows } from "./helpers.js";
import { assertClose } from "./assertions.js";
import { rng, makeNondegenerate } from "./datagen.js";
import { evaluatedInR } from "../test-support/r-oracle.js";

test("lambdaMax matches glmnet's own path-starting λ (fit$lambda[1])", () => {
  // R-ORACLE-TAG-START
  const env = {
    X: [[1.0, -1.0], [2.0, 0.5], [0.0, 3.0], [-1.0, 0.5]],
    y: [1, 0, 1, 0],  // Note: glmnet needs at least 2 of each class to avoid a degenerate fit
  };
  const r = String.raw`
fit <- glmnet(X, y, family = "binomial", alpha = 1)
list(lambdaMax = fit$lambda[1])
`;
  // R-ORACLE-TAG-END

  const X = matrixFromRows(env.X);
  const y = new Float64Array(env.y);
  const { lambdaMax: lm } = lambdaMax(X, y);
  const expected = evaluatedInR(r, env);
  assertClose(lm, expected.lambdaMax, 1e-12, "lambdaMax vs glmnet path start:");
});

test("λ ≥ λ_max ⇒ all coefficients exactly zero, β0 = logit(ȳ)", () => {
  const rand = rng(101);
  for (let t = 0; t < 40; t++) {
    const { X, y } = makeNondegenerate(rand, {
      n: 30 + Math.floor(rand() * 60), p: 2 + Math.floor(rand() * 12),
      rho: rand() * 0.8, mixedScale: rand() < 0.5,
    });
    const { lambdaMax: lm } = lambdaMax(X, y);
    const res = fitLassoLogistic(X, y, lm * 1.001);
    const nnz = [...res.beta].filter((v) => v !== 0).length;
    assert.equal(nnz, 0, `expected 0 nonzero above λ_max, got ${nnz}`);
    assertClose(res.beta0, logit(mean(y)), 1e-6, "intercept should be logit(ȳ)");
  }
});

test("λ just below λ_max ⇒ the argmax coefficient is the first to enter", () => {
  const rand = rng(202);
  for (let t = 0; t < 40; t++) {
    const { X, y } = makeNondegenerate(rand, {
      n: 30 + Math.floor(rand() * 60), p: 2 + Math.floor(rand() * 12),
      rho: rand() * 0.6, mixedScale: rand() < 0.5,
    });
    const { lambdaMax: lm, argmax } = lambdaMax(X, y);
    const res = fitLassoLogistic(X, y, lm * 0.99);
    const nz = [...res.beta].map((v, j) => (v !== 0 ? j : -1)).filter((j) => j >= 0);
    // The argmax (strongest null-model correlation) is guaranteed to be the first
    // coefficient to become nonzero as λ drops below λ_max. We do NOT assert
    // "exactly one": if a second feature's correlation sits within the 1% band it
    // enters too, which is common on correlated data and was only absent here by
    // luck of the tie-free generator. So we assert the always-true invariant:
    // λ_max is tight (something enters) and the entrant is the argmax.
    assert.ok(nz.length >= 1, "expected ≥1 nonzero just below λ_max, got none");
    assert.ok(nz.includes(argmax), `expected argmax ${argmax} among entrants, got [${nz}]`);
  }
});
