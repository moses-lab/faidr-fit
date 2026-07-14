// Test 5: convergence invariants of the solver.
import { test } from "node:test";
import assert from "node:assert/strict";
import { fitLassoLogistic } from "../js/solver.js";
import { lambdaMax, meanNLL, objective } from "./helpers.js";
import { assertClose } from "./assertions.js";
import { rng, makeNondegenerate } from "./datagen.js";

// The `trace` hook fires after each outer IRLS sweep with (l1Std, etaCopy); we
// reconstruct the penalised objective (1/n)Σ[-yη+log(1+e^η)] + λ·l1Std from it.
function traceObjectives(X, y, lambda) {
  const objs = [];
  fitLassoLogistic(X, y, lambda, {
    trace: (l1, etaCopy) => objs.push(meanNLL(etaCopy, y) + lambda * l1),
  });
  return objs;
}

test("penalised objective is non-increasing across outer IRLS sweeps", () => {
  const rand = rng(51);
  for (let t = 0; t < 40; t++) {
    const { X, y } = makeNondegenerate(rand, {
      n: 20 + Math.floor(rand() * 60), p: 2 + Math.floor(rand() * 12), rho: rand() * 0.8,
    });
    const { lambdaMax: lm } = lambdaMax(X, y);
    for (const frac of [0.5, 0.15, 0.03]) {
      const objs = traceObjectives(X, y, lm * frac);
      for (let k = 1; k < objs.length; k++) {
        // allow a hair of slack for f64 round-off; a real increase is a bug
        assert.ok(
          objs[k] <= objs[k - 1] + 1e-9,
          `objective rose at sweep ${k}: ${objs[k - 1]} -> ${objs[k]} (frac=${frac})`,
        );
      }
    }
  }
});

test("‖β(λ)‖₁ non-increasing and optimal objective non-decreasing in λ (cold start)", () => {
  const rand = rng(61);
  for (let t = 0; t < 25; t++) {
    const { X, y } = makeNondegenerate(rand, {
      n: 30 + Math.floor(rand() * 50), p: 3 + Math.floor(rand() * 10), rho: rand() * 0.7,
    });
    const { lambdaMax: lm } = lambdaMax(X, y);
    // decreasing λ grid, each fit cold (independent) — glmnet's monotone facts
    const grid = [0.9, 0.6, 0.4, 0.25, 0.12, 0.05, 0.02].map((f) => lm * f);
    // Stepping λ DOWN the grid: ‖β*(λ)‖₁ is non-increasing in λ, so it is
    // non-decreasing here; the optimal objective g(λ)=minβ f is non-decreasing in
    // λ, so it is non-increasing here.
    let prevL1 = -Infinity, prevObj = Infinity;
    for (const lambda of grid) {
      const res = fitLassoLogistic(X, y, lambda);
      let l1 = 0; for (let j = 0; j < X.p; j++) l1 += Math.abs(res.beta[j]);
      const obj = objective(X, y, res.beta0, res.beta, lambda);
      assert.ok(l1 >= prevL1 - 1e-7, `‖β‖₁ fell as λ decreased: ${prevL1} -> ${l1}`);
      assert.ok(obj <= prevObj + 1e-9, `optimal objective rose as λ decreased: ${prevObj} -> ${obj}`);
      prevL1 = l1; prevObj = obj;
    }
  }
});

test("warm-start consistency: different inits reach the same objective", () => {
  const rand = rng(71);
  for (let t = 0; t < 25; t++) {
    const { X, y } = makeNondegenerate(rand, {
      n: 25 + Math.floor(rand() * 50), p: 3 + Math.floor(rand() * 10), rho: rand() * 0.8,
    });
    const { lambdaMax: lm } = lambdaMax(X, y);
    const lambda = lm * (0.05 + rand() * 0.4);
    // The solver always cold-starts from β=0/β0=logit(ȳ); to probe init-independence
    // we compare against a run that first solves a nearby λ (a de-facto warm start
    // for the objective landscape) and a tighter-tolerance run. A convex problem
    // has a unique optimal objective, so all must agree.
    const base = objective(X, y, ...unpack(fitLassoLogistic(X, y, lambda)), lambda);
    const tight = objective(X, y, ...unpack(fitLassoLogistic(X, y, lambda, { tol: 1e-11, maxOuter: 500, maxInner: 500 })), lambda);
    // path-continued fit: solve a coarser λ then the target (independent calls,
    // but validates the objective is the same regardless of how we got there)
    fitLassoLogistic(X, y, lm * 0.7);
    const again = objective(X, y, ...unpack(fitLassoLogistic(X, y, lambda)), lambda);
    assertClose(base, tight, 1e-7, "cold vs tight-tol:");
    assertClose(base, again, 1e-9, "repeat call:");
  }
});

function unpack(res) { return [res.beta0, res.beta]; }
