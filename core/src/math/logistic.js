/**
 * Shared logistic-link building blocks used by both the penalized path
 * solver (irlsLogisticLasso.js) and the unpenalized refit
 * (fitLogistic.js), so the two never drift apart on how deviance or the
 * IRLS working response are computed.
 *
 * X is column-major throughout: an array of p columns, each a
 * Float64Array of length n.
 */

// Fitted probabilities are floored away from {0, 1} at sqrt(machine
// epsilon). Beyond that scale, p*(1-p) underflows toward zero in double
// precision and the IRLS working weight blows up, which is the classic
// divergence failure mode for near-separable data (Friedman, Hastie &
// Tibshirani 2010, Sec. 4, note "Care is taken to avoid coefficients
// diverging in order to achieve fitted probabilities of 0 or 1").
export const PROBABILITY_FLOOR = Math.sqrt(Number.EPSILON);

export function mean(v, n) {
  let s = 0;
  for (let i = 0; i < n; i++) s += v[i];
  return s / n;
}

export function logit(p) {
  return Math.log(p / (1 - p));
}

export function sigmoid(eta) {
  return 1 / (1 + Math.exp(-eta));
}

/**
 * Linear predictor eta_i = beta0 + sum_j X_ij * beta_j, looping over
 * columns in the outer loop so each inner loop is a contiguous,
 * cache-friendly sweep down one column.
 */
export function linearPredictor(X, n, beta0, beta) {
  const eta = new Float64Array(n).fill(beta0);
  const p = beta.length;
  for (let j = 0; j < p; j++) {
    const b = beta[j];
    if (b === 0) continue;
    const col = X[j];
    for (let i = 0; i < n; i++) eta[i] += col[i] * b;
  }
  return eta;
}

/**
 * Binomial deviance D = 2 * sum_i [log(1 + exp(eta_i)) - y_i * eta_i].
 * The saturated-model deviance contribution is exactly zero for binary
 * response data, so this is the full deviance, not just a difference.
 * log(1 + exp(eta)) is evaluated via log1p in whichever branch avoids
 * overflow.
 */
export function computeDeviance(X, y, n, beta0, beta) {
  const eta = linearPredictor(X, n, beta0, beta);

  let deviance = 0;
  for (let i = 0; i < n; i++) {
    const e = eta[i];
    const logOnePlusExpE = e > 0 ? e + Math.log1p(Math.exp(-e)) : Math.log1p(Math.exp(e));
    deviance += 2 * (logOnePlusExpE - y[i] * e);
  }
  return deviance;
}

export function computeNullDeviance(y, n) {
  const ybar = mean(y, n);
  let deviance = 0;
  for (let i = 0; i < n; i++) {
    deviance += y[i] ? -Math.log(ybar) : -Math.log(1 - ybar);
  }
  return 2 * deviance;
}

/**
 * logit(ybar): the exact MLE of an intercept-only logistic model, and
 * therefore the correct warm-start intercept at lambda_max, where every
 * slope coefficient is exactly zero.
 */
export function nullIntercept(y, n) {
  return logit(mean(y, n));
}

/**
 * IRLS working response and weights at the current (beta0, beta)
 * (Friedman, Hastie & Tibshirani 2010, Eq. 12-13):
 *
 *   p_i = sigmoid(eta_i)
 *   w_i = p_i * (1 - p_i)
 *   z_i = eta_i + (y_i - p_i) / w_i
 */
export function workingResponse(X, y, n, beta0, beta) {
  const eta = linearPredictor(X, n, beta0, beta);

  const z = new Float64Array(n);
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let p = sigmoid(eta[i]);
    if (p < PROBABILITY_FLOOR) p = PROBABILITY_FLOOR;
    else if (p > 1 - PROBABILITY_FLOOR) p = 1 - PROBABILITY_FLOOR;

    const weight = p * (1 - p);
    w[i] = weight;
    z[i] = eta[i] + (y[i] - p) / weight;
  }
  return { z, w };
}
