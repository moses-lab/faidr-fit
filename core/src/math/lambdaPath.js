import { mean } from './logistic.js';

/**
 * lambda_max: the smallest lambda for which every non-intercept
 * coefficient is exactly zero at the lasso solution.
 *
 * At beta = 0, the null-model intercept is beta0 = logit(ybar), so
 * p_i = ybar and w_i = ybar(1-ybar) for every observation, and the IRLS
 * working response satisfies w_i * (z_i - beta0) = y_i - p_i = y_i -
 * ybar exactly. Substituting into the coordinatewise KKT condition for
 * beta_j = 0 to be optimal, |1/N sum_i w_i x_ij (z_i - beta0)| <=
 * lambda, the weight cancels and leaves a closed form independent of
 * the IRLS weights:
 *
 *   lambda_max = max_j | (1/N) sum_i x_ij_std * (y_i - ybar) |
 *
 * (Friedman, Hastie & Tibshirani 2010, Sec. 2.5).
 */
export function computeLambdaMax(Xstd, y, n, p) {
  const ybar = mean(y, n);

  let maxAbsCorrelation = 0;
  for (let j = 0; j < p; j++) {
    const col = Xstd[j];
    let correlation = 0;
    for (let i = 0; i < n; i++) correlation += col[i] * (y[i] - ybar);
    correlation = Math.abs(correlation) / n;
    if (correlation > maxAbsCorrelation) maxAbsCorrelation = correlation;
  }

  return maxAbsCorrelation;
}

/**
 * Full lambda sequence of `nlambda` values, log-spaced from lambda_max
 * down to lambda_min = epsilon * lambda_max, where epsilon follows the
 * standard glmnet convention (Hastie, Tibshirani & Wainwright 2015,
 * Sec. 2.2.4): 1e-4 when there are at least as many observations as
 * features, 1e-2 otherwise (a smaller ratio would let lambda_min
 * approach the unregularized fit, which is unstable when p > n).
 */
export function buildLambdaPath(lambdaMax, n, p, nlambda) {
  if (nlambda === 1) return [lambdaMax];

  const epsilon = n >= p ? 1e-4 : 1e-2;
  const lambdaMin = epsilon * lambdaMax;

  const logMax = Math.log(lambdaMax);
  const logMin = Math.log(lambdaMin);
  const step = (logMin - logMax) / (nlambda - 1);

  const path = new Array(nlambda);
  for (let k = 0; k < nlambda; k++) path[k] = Math.exp(logMax + step * k);
  return path;
}
