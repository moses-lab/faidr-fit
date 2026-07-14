// Test 2: λ_max closed form. Exercises the soft-threshold boundary and intercept
// handling directly. At λ just above λ_max the whole solution collapses to the
// null model; just below, exactly the argmax coordinate switches on.
import { test } from "node:test";
import assert from "node:assert/strict";
import { fitLassoLogistic } from "../js/solver.js";
import { lambdaMax, logit, mean } from "./helpers.js";
import { assertClose } from "./assertions.js";
import { rng, makeNondegenerate } from "./datagen.js";

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

test("λ just below λ_max ⇒ exactly one nonzero coef, at the argmax", () => {
  const rand = rng(202);
  for (let t = 0; t < 40; t++) {
    const { X, y } = makeNondegenerate(rand, {
      n: 30 + Math.floor(rand() * 60), p: 2 + Math.floor(rand() * 12),
      rho: rand() * 0.6, mixedScale: rand() < 0.5,
    });
    const { lambdaMax: lm, argmax } = lambdaMax(X, y);
    const res = fitLassoLogistic(X, y, lm * 0.99);
    const nz = [...res.beta].map((v, j) => (v !== 0 ? j : -1)).filter((j) => j >= 0);
    assert.deepEqual(nz, [argmax], `expected single nonzero at ${argmax}, got [${nz}]`);
  }
});
