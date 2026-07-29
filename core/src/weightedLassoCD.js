import { softThreshold } from './math/softThreshold.js';

/**
 * Coordinate descent solver for the weighted lasso subproblem that
 * arises from linearizing the logistic log-likelihood around a
 * current fit (the "inner loop" of Friedman, Hastie & Tibshirani
 * 2010's IRLS-coordinate-descent algorithm for logistic regression):
 *
 *   minimize_{beta0,beta}  (1/2N) sum_i w_i (z_i - beta0 - x_i.beta)^2
 *                          + lambda * sum_j |beta_j|
 *
 * w and z are held fixed for the duration of one call (they belong to
 * the outer IRLS loop; see irlsLogisticLasso.js). beta0 and beta are
 * passed in as a warm start and updated toward the solution of this
 * quadratic subproblem.
 *
 * Setting lambda = 0 turns this into a plain weighted least squares
 * solve via coordinate descent (Gauss-Seidel on the normal equations);
 * fitLogistic.js reuses this solver in exactly that way, so the
 * penalized and unpenalized fits share one implementation rather than
 * duplicating it.
 *
 * Performance: uses glmnet's active-set strategy (Friedman, Hastie &
 * Tibshirani 2010, Sec. 2.4-2.5). Most features are zero at any given
 * lambda, and once a feature is known to be zero it usually stays zero
 * from one sweep to the next, so cycling over every feature every
 * sweep is wasted work once p is large. Instead: (1) cycle only over
 * the current active set (nonzero coefficients) until that restricted
 * problem converges, which is cheap; (2) do one full sweep over every
 * feature to check whether any inactive feature now violates the KKT
 * condition and should enter the model; (3) if none do, the full
 * problem is solved; if some do, add them to the active set and repeat
 * from (1). This is exact, not an approximation: the full sweep in
 * step (2) is what guarantees correctness, the active-set restriction
 * in step (1) is what makes it fast.
 */
export function weightedLassoCD({ Xstd, w, z, beta0, beta, lambda, thresh, scale, maxIter }) {
  const p = Xstd.length;
  const n = z.length;

  let sumW = 0;
  for (let i = 0; i < n; i++) sumW += w[i];

  // Weighted sum of squares per feature, sum_i w_i x_ij^2 / N. This
  // depends only on w, which is fixed for this call, so it is
  // computed once and reused across every sweep.
  const denom = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    const col = Xstd[j];
    let s = 0;
    for (let i = 0; i < n; i++) s += w[i] * col[i] * col[i];
    denom[j] = s / n;
  }

  // eta tracks the current fit beta0 + x.beta, maintained
  // incrementally as each coefficient updates so no O(np)
  // recomputation is needed per sweep.
  const eta = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let e = beta0[0];
    for (let j = 0; j < p; j++) e += Xstd[j][i] * beta[j];
    eta[i] = e;
  }

  // Updates a single coordinate (intercept when j === -1, feature j
  // otherwise) and returns the resulting change in the quadratic
  // objective, used both to detect convergence and to decide whether
  // a coordinate newly became active.
  function updateCoordinate(j) {
    if (j === -1) {
      let weightedResidualSum = 0;
      for (let i = 0; i < n; i++) weightedResidualSum += w[i] * (z[i] - eta[i]);
      const delta = weightedResidualSum / sumW;
      if (delta === 0) return 0;
      beta0[0] += delta;
      for (let i = 0; i < n; i++) eta[i] += delta;
      return delta * delta * (sumW / n);
    }

    if (denom[j] === 0) return 0; // constant / zero-variance column
    const col = Xstd[j];
    const oldBeta = beta[j];

    let weightedCorr = 0;
    for (let i = 0; i < n; i++) {
      const partialResidual = z[i] - eta[i] + col[i] * oldBeta;
      weightedCorr += w[i] * col[i] * partialResidual;
    }
    weightedCorr /= n;

    const newBeta = softThreshold(weightedCorr, lambda) / denom[j];
    const delta = newBeta - oldBeta;
    if (delta === 0) return 0;

    beta[j] = newBeta;
    for (let i = 0; i < n; i++) eta[i] += delta * col[i];
    return delta * delta * denom[j];
  }

  let active = [];
  for (let j = 0; j < p; j++) if (beta[j] !== 0) active.push(j);

  for (let outer = 0; outer < maxIter; outer++) {
    // Phase 1: converge on the restricted (active-set) problem.
    for (let inner = 0; inner < maxIter; inner++) {
      let maxChange = updateCoordinate(-1);
      for (const j of active) {
        const change = updateCoordinate(j);
        if (change > maxChange) maxChange = change;
      }
      if (maxChange < thresh * scale) break;
    }

    // Phase 2: a full sweep to check for KKT violations among
    // currently-inactive features (i.e. features that should now
    // enter the model).
    let newlyActive = false;
    let maxChange = 0;
    for (let j = 0; j < p; j++) {
      const wasActive = beta[j] !== 0;
      const change = updateCoordinate(j);
      if (change > maxChange) maxChange = change;
      if (!wasActive && beta[j] !== 0) newlyActive = true;
    }

    if (newlyActive) {
      active = [];
      for (let j = 0; j < p; j++) if (beta[j] !== 0) active.push(j);
    }

    if (maxChange < thresh * scale && !newlyActive) break;
  }

  return { eta };
}
