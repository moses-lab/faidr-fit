import { softThreshold } from './softThreshold.js';

/**
 * Solves the penalized reweighted-least-squares subproblem that IRLS
 * hands off at each outer iteration:
 *
 *   minimize_{beta0,beta}  (1/2N) sum_i w_i (z_i - beta0 - x_i.beta)^2
 *                          + lambda * sum_j |beta_j|
 *
 * via cyclical coordinate descent restricted to an active set
 * (Friedman, Hastie & Tibshirani 2010, Sec. 5): cycle over the current
 * active set to convergence, then sweep every variable once to check
 * for KKT violations; any violator is added to the active set and the
 * process repeats. This confines the expensive O(n) inner updates to
 * variables that are actually likely to be nonzero, which is where
 * coordinate descent's speed on sparse solutions comes from.
 *
 * Xstd is column-major (an array of p Float64Array columns of length
 * n), already standardized. beta0Init/betaInit are the warm-start
 * values carried over from the previous point on the lambda path.
 */
export function weightedLassoCD(Xstd, z, w, lambda, beta0Init, betaInit, n, p, opts) {
  const { convergenceThreshold } = opts;

  const beta = Float64Array.from(betaInit);
  let beta0 = beta0Init;

  // Weighted column norms (1/N) sum_i w_i x_ij^2. These depend only on
  // the current IRLS weights w, so they are computed once per call and
  // reused across every coordinate descent sweep below.
  const colNorm = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    const col = Xstd[j];
    let s = 0;
    for (let i = 0; i < n; i++) s += w[i] * col[i] * col[i];
    colNorm[j] = s / n;
  }

  // Residual r_i = z_i - beta0 - x_i.beta, maintained incrementally so
  // that a coordinate update only has to touch the one column it
  // changed, rather than recomputing every fitted value from scratch.
  const r = new Float64Array(n);
  for (let i = 0; i < n; i++) r[i] = z[i] - beta0;
  for (let j = 0; j < p; j++) {
    const b = beta[j];
    if (b === 0) continue;
    const col = Xstd[j];
    for (let i = 0; i < n; i++) r[i] -= col[i] * b;
  }

  let sumW = 0;
  for (let i = 0; i < n; i++) sumW += w[i];

  const active = new Set();
  for (let j = 0; j < p; j++) if (beta[j] !== 0) active.add(j);

  for (;;) {
    // Cycle over the active set until it stops moving.
    let activeSetConverged = false;
    while (!activeSetConverged) {
      let maxObjectiveDecrease = 0;

      // Intercept is unpenalized, so its exact minimizer given the rest
      // of beta is just the weighted mean of the current residual.
      if (sumW > 0) {
        let weightedResidualSum = 0;
        for (let i = 0; i < n; i++) weightedResidualSum += w[i] * r[i];
        const beta0Step = weightedResidualSum / sumW;

        if (beta0Step !== 0) {
          for (let i = 0; i < n; i++) r[i] -= beta0Step;
          beta0 += beta0Step;
          maxObjectiveDecrease = Math.max(maxObjectiveDecrease, Math.abs(beta0Step) * sumW / n);
        }
      }

      for (const j of active) {
        if (colNorm[j] === 0) continue;
        const col = Xstd[j];
        const oldBeta = beta[j];

        // rho_j = (1/N) sum_i w_i x_ij (r_i + x_ij*oldBeta): the
        // add-back term restores beta_j's own contribution to the
        // residual, giving the correlation as if beta_j were still 0.
        let rho = oldBeta * colNorm[j];
        for (let i = 0; i < n; i++) rho += (w[i] * col[i] * r[i]) / n;

        const newBeta = softThreshold(rho, lambda) / colNorm[j];
        const delta = newBeta - oldBeta;

        if (delta !== 0) {
          for (let i = 0; i < n; i++) r[i] -= col[i] * delta;
          beta[j] = newBeta;
          maxObjectiveDecrease = Math.max(maxObjectiveDecrease, Math.abs(delta) * colNorm[j]);
        }
      }

      activeSetConverged = maxObjectiveDecrease < convergenceThreshold;
    }

    // KKT sweep: any variable outside the active set whose correlation
    // with the residual exceeds lambda violates optimality and must
    // join the active set for another round of coordinate descent.
    let addedVariable = false;
    for (let j = 0; j < p; j++) {
      if (active.has(j) || colNorm[j] === 0) continue;
      const col = Xstd[j];
      let rho = 0;
      for (let i = 0; i < n; i++) rho += (w[i] * col[i] * r[i]) / n;

      if (Math.abs(rho) > lambda) {
        active.add(j);
        addedVariable = true;
      }
    }

    if (!addedVariable) break;
  }

  return { beta0, beta };
}
