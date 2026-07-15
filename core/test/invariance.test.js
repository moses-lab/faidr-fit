// Test 3 (duplication invariance — proves the 1/N scaling) and Test 4 (label swap
// — catches gradient-sign bugs). Both are exact symmetries of the mean-loss
// objective, so the solutions must agree to solver tolerance.
import { test } from "node:test";
import { fitLassoLogistic, lambdaMax } from "../js/solver.js";
import { matrixFromRows } from "./helpers.js";
import { assertClose, assertKKT } from "./assertions.js";
import { rng, makeNondegenerate } from "./datagen.js";

// Stack rows [X;X], [y;y]. Mean-scaled loss and population standardization are
// both invariant to exact duplication, so β and β0 must be unchanged.
test("duplication invariance: fit([X;X],[y;y]) == fit(X,y)", () => {
  const rand = rng(31);
  for (let t = 0; t < 30; t++) {
    const { X, y } = makeNondegenerate(rand, {
      n: 20 + Math.floor(rand() * 40), p: 2 + Math.floor(rand() * 10), rho: rand() * 0.7,
    });
    // stack [X; X] and [y; y] — rows and labels blocked identically so they align
    const rows2 = [], y2 = new Float64Array(2 * X.n);
    for (let rep = 0; rep < 2; rep++) {
      for (let i = 0; i < X.n; i++) {
        rows2.push(X.cols.map((c) => c[i]));
        y2[rep * X.n + i] = y[i];
      }
    }
    const Xd = matrixFromRows(rows2);
    const { lambdaMax: lm } = lambdaMax(X, y);
    for (const frac of [0.6, 0.2, 0.05]) {
      const lambda = lm * frac;
      const a = fitLassoLogistic(X, y, lambda);
      const b = fitLassoLogistic(Xd, y2, lambda);
      assertClose(a.beta0, b.beta0, 1e-6, `β0 frac=${frac}:`);
      for (let j = 0; j < X.p; j++) {
        assertClose(a.beta[j], b.beta[j], 1e-6, `β[${j}] frac=${frac}:`);
      }
    }
  }
});

// y -> 1-y sends p̂ -> 1-p̂, η -> -η, so the optimum flips sign throughout.
test("label swap: fit(X,1-y) has β' ≈ -β and β0' ≈ -β0", () => {
  const rand = rng(41);
  for (let t = 0; t < 40; t++) {
    const { X, y } = makeNondegenerate(rand, {
      n: 20 + Math.floor(rand() * 50), p: 2 + Math.floor(rand() * 12), rho: rand() * 0.8,
    });
    const ys = new Float64Array(X.n);
    for (let i = 0; i < X.n; i++) ys[i] = 1 - y[i];
    const { lambdaMax: lm } = lambdaMax(X, y);
    for (const frac of [0.6, 0.2, 0.05]) {
      const lambda = lm * frac;
      const a = fitLassoLogistic(X, y, lambda);
      const b = fitLassoLogistic(X, ys, lambda);
      assertClose(a.beta0, -b.beta0, 1e-6, `β0 frac=${frac}:`);
      for (let j = 0; j < X.p; j++) {
        assertClose(a.beta[j], -b.beta[j], 1e-6, `β[${j}] frac=${frac}:`);
      }
      // both fits must independently be KKT points
      assertKKT(X, ys, b, lambda, 1e-5, `swapped frac=${frac}:`);
    }
  }
});
