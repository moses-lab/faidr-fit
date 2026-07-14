// The reusable KKT assertion (Test 1's backbone), called from every other test.
import assert from "node:assert/strict";
import { kktViolation } from "./helpers.js";

// Assert the returned fit is a KKT point of the penalised objective. τ defaults
// to 1e-5: ~100x the solver's 1e-7 convergence tol, and empirically the worst
// violation over hundreds of random standardized-f64 problems sits near 1e-6.
export function assertKKT(X, y, res, lambda, tau = 1e-5, msg = "") {
  const w = kktViolation(X, y, res.beta0, res.beta, lambda);
  assert.ok(
    w.viol <= tau,
    `${msg} KKT violated: ${w.viol.toExponential(3)} > ${tau} at ` +
      `${w.where === "intercept" ? "intercept" : `coord ${w.where}`}` +
      ` (residual-corr ${w.c.toExponential(3)}, λ=${lambda}). ` +
      (w.where === "intercept"
        ? "intercept/mean-residual bug."
        : "inactive-coord failure => screening/active-set bug; " +
          "uniform failure => under-convergence."),
  );
  return w;
}

export function assertClose(a, b, tol, msg = "") {
  assert.ok(
    Math.abs(a - b) <= tol,
    `${msg} expected ${a} ≈ ${b} (|Δ|=${Math.abs(a - b).toExponential(3)} > ${tol})`,
  );
}

export function assertAllFinite(arr, msg = "") {
  for (let i = 0; i < arr.length; i++) {
    assert.ok(Number.isFinite(arr[i]), `${msg} non-finite at index ${i}: ${arr[i]}`);
  }
}
