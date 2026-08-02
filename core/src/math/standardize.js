/**
 * Standardizes each column of X to mean zero and (population) standard
 * deviation one, matching glmnet's internal convention of fitting on the
 * standardized scale and transforming coefficients back afterwards
 * (Friedman, Hastie & Tibshirani 2010, Sec. 2.3; the population, not
 * sample, standard deviation is used throughout glmnet).
 *
 * X is column-major: an array of p columns, each a Float64Array of
 * length n.
 */
export function standardize(X, n, p) {
  const means = new Float64Array(p);
  const sds = new Float64Array(p);
  const Xstd = new Array(p);

  for (let j = 0; j < p; j++) {
    const col = X[j];

    let sum = 0;
    for (let i = 0; i < n; i++) sum += col[i];
    const mean = sum / n;

    let sumSquaredDeviations = 0;
    for (let i = 0; i < n; i++) {
      const centered = col[i] - mean;
      sumSquaredDeviations += centered * centered;
    }
    const sd = Math.sqrt(sumSquaredDeviations / n);

    means[j] = mean;
    sds[j] = sd;

    const standardizedCol = new Float64Array(n);
    if (sd > 0) {
      for (let i = 0; i < n; i++) standardizedCol[i] = (col[i] - mean) / sd;
    }
    // A column with zero variance carries no information and is left at
    // zero; weightedLassoCD recognizes this via a zero column norm and
    // never lets such a coefficient enter the active set.
    Xstd[j] = standardizedCol;
  }

  return { Xstd, means, sds };
}

/**
 * Converts coefficients fit on the standardized scale back to the
 * original scale of X:
 *
 *   beta_j       = betaStd_j / sd_j
 *   beta0        = beta0Std - sum_j beta_j * mean_j
 */
export function unstandardizeCoefficients(beta0Std, betaStd, means, sds) {
  const p = betaStd.length;
  const beta = new Float64Array(p);

  let shift = 0;
  for (let j = 0; j < p; j++) {
    beta[j] = sds[j] > 0 ? betaStd[j] / sds[j] : 0;
    shift += beta[j] * means[j];
  }

  return { beta0: beta0Std - shift, beta };
}
