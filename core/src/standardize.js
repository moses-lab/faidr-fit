/**
 * glmnet fits the lasso path in a standardized coordinate system: each
 * column of X is centered to mean 0 and scaled to a *population*
 * standard deviation of 1 (denominator N, not N-1). This keeps the L1
 * penalty comparable across features regardless of their original
 * units, and keeps coordinate descent well conditioned. Coefficients
 * are mapped back to the original scale once fitting is complete (see
 * `unstandardizeCoefficients`), so callers of the public API never see
 * the standardized representation.
 *
 * Takes column-major X (see math/logistic.js:toColumnMajor) and
 * returns column-major standardized output, matching the access
 * pattern used everywhere else in the solver.
 */
export function standardize(Xcol) {
  const p = Xcol.length;
  const n = Xcol[0].length;

  const mean = new Float64Array(p);
  const sd = new Float64Array(p);

  for (let j = 0; j < p; j++) {
    const col = Xcol[j];
    let sum = 0;
    for (let i = 0; i < n; i++) sum += col[i];
    mean[j] = sum / n;
  }

  for (let j = 0; j < p; j++) {
    const col = Xcol[j];
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const d = col[i] - mean[j];
      sumSq += d * d;
    }
    sd[j] = Math.sqrt(sumSq / n);
  }

  const Xstd = Array.from({ length: p }, () => new Float64Array(n));
  for (let j = 0; j < p; j++) {
    // A constant column (sd === 0) carries no information for the
    // lasso. It is left at all-zero rather than divided by zero, so
    // it can never enter the model (matches glmnet's handling of
    // zero-variance predictors).
    const scale = sd[j] === 0 ? 0 : 1 / sd[j];
    const col = Xcol[j];
    const out = Xstd[j];
    for (let i = 0; i < n; i++) {
      out[i] = (col[i] - mean[j]) * scale;
    }
  }

  return { Xstd, mean, sd };
}

/**
 * Maps coefficients fit in standardized space back to the original
 * scale of X: beta_original = beta_std / sd, with the intercept
 * absorbing the feature means that were removed during standardization.
 */
export function unstandardizeCoefficients(beta0Std, betaStd, mean, sd) {
  const p = betaStd.length;
  const beta = new Float64Array(p);
  let meanCorrection = 0;

  for (let j = 0; j < p; j++) {
    beta[j] = sd[j] === 0 ? 0 : betaStd[j] / sd[j];
    meanCorrection += beta[j] * mean[j];
  }

  return { beta0: beta0Std - meanCorrection, beta };
}
