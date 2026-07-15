// Test 1: KKT residual check — the backbone. Runs the reusable assertion over a
// broad sweep of random problems (varying N, p, N<p, correlated columns, mixed
// scales) plus a couple of hand-built cases, each at several λ along the path.
import { test } from "node:test";
import { fitLassoLogistic, lambdaMax } from "../js/solver.js";
import { matrixFromRows } from "./helpers.js";
import { assertKKT } from "./assertions.js";
import { rng, makeNondegenerate } from "./datagen.js";

test("KKT holds across hundreds of random problems and λ values", () => {
  const rand = rng(7);
  let count = 0;
  for (let t = 0; t < 250; t++) {
    const n = 15 + Math.floor(rand() * 90);
    const p = 2 + Math.floor(rand() * 18);
    const { X, y } = makeNondegenerate(rand, {
      n, p, rho: rand() * 0.95, mixedScale: rand() < 0.5,
    });
    const { lambdaMax: lm } = lambdaMax(X, y);
    // a few λ along the active path (fractions of λ_max) plus one above it
    for (const frac of [0.9, 0.5, 0.15, 0.03, 1.2]) {
      const lambda = lm * frac;
      const res = fitLassoLogistic(X, y, lambda);
      assertKKT(X, y, res, lambda, 1e-5, `n=${n} p=${p} frac=${frac}:`);
      count++;
    }
  }
  // guard that the loop actually ran the intended volume
  if (count < 1000) throw new Error(`only ${count} KKT checks ran`);
});

test("KKT holds for N < p (underdetermined)", () => {
  const rand = rng(13);
  for (let t = 0; t < 60; t++) {
    const n = 8 + Math.floor(rand() * 10);
    const p = n + 5 + Math.floor(rand() * 20);
    const { X, y } = makeNondegenerate(rand, { n, p, rho: rand() * 0.7 });
    const { lambdaMax: lm } = lambdaMax(X, y);
    for (const frac of [0.6, 0.2, 0.05]) {
      const lambda = lm * frac;
      assertKKT(X, y, fitLassoLogistic(X, y, lambda), lambda, 1e-5, `N<p n=${n} p=${p}:`);
    }
  }
});

test("KKT holds on a hand-built 5×2 case", () => {
  const X = matrixFromRows([[1, 0], [0, 1], [1, 1], [-1, 2], [2, -1]]);
  const y = new Float64Array([1, 0, 1, 0, 1]);
  const { lambdaMax: lm } = lambdaMax(X, y);
  for (const frac of [0.8, 0.3, 0.05]) {
    const lambda = lm * frac;
    assertKKT(X, y, fitLassoLogistic(X, y, lambda), lambda, 1e-5, `frac=${frac}:`);
  }
});
