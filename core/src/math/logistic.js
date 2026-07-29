/**
 * Shared numerical primitives for logistic regression, used by both the
 * penalized path fit (irlsLogisticLasso.js) and the unpenalized refit
 * (fitLogistic.js), so the two solvers agree on how probabilities,
 * deviance, and IRLS weights are computed.
 */

// glmnet bounds fitted probabilities away from 0 and 1 before computing
// IRLS weights w = p(1-p) and the working response z = eta + (y-p)/w.
// Without this, a perfectly (or near-perfectly) separated feature drives
// p toward 0 or 1, w toward 0, and z toward +/-infinity, which is the
// classic logistic-IRLS divergence failure mode. This bound is glmnet's
// own numerical safeguard (Friedman, Hastie & Tibshirani 2010, Sec. 3),
// not a modeling threshold.
export const PROB_EPS = 1e-5;

export function sigmoid(eta) {
  return 1 / (1 + Math.exp(-eta));
}

export function clipProbability(p) {
  return Math.min(Math.max(p, PROB_EPS), 1 - PROB_EPS);
}

/**
 * Binomial deviance, -2 * log-likelihood, computed directly from eta
 * (rather than from p = sigmoid(eta)) via log1p so it stays accurate
 * even when eta is large in magnitude and p would round to exactly 0
 * or 1 in floating point.
 */
export function deviance(y, eta) {
  let dev = 0;
  for (let i = 0; i < y.length; i++) {
    const logOnePlusExpEta = eta[i] > 0
      ? eta[i] + Math.log1p(Math.exp(-eta[i]))
      : Math.log1p(Math.exp(eta[i]));
    dev += logOnePlusExpEta - y[i] * eta[i];
  }
  return 2 * dev;
}

/**
 * Deviance of the intercept-only model. This is used throughout as the
 * `scale` in relative convergence checks (thresh * nullDeviance), which
 * is what makes the convergence criterion scale-free rather than an
 * absolute cutoff that behaves differently on differently-sized datasets.
 */
export function nullModelDeviance(y) {
  const n = y.length;
  let ybar = 0;
  for (let i = 0; i < n; i++) ybar += y[i];
  ybar /= n;

  let dev = 0;
  for (let i = 0; i < n; i++) {
    dev += y[i] * Math.log(ybar) + (1 - y[i]) * Math.log(1 - ybar);
  }
  return -2 * dev;
}

/** logit(ybar): the exact MLE of an intercept-only logistic model, and
 * therefore the correct warm-start intercept at lambda_max, where every
 * slope coefficient is exactly zero. */
export function nullIntercept(y) {
  const n = y.length;
  let ybar = 0;
  for (let i = 0; i < n; i++) ybar += y[i];
  ybar /= n;
  return Math.log(ybar / (1 - ybar));
}

/** Converts a row-major design matrix (X[i][j], observations x features)
 * into column-major storage (Xcol[j][i]), which is the access pattern
 * coordinate descent needs: for a fixed feature j, scan all i. */
export function toColumnMajor(X) {
  const n = X.length;
  const p = X[0].length;
  const Xcol = Array.from({ length: p }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      Xcol[j][i] = X[i][j];
    }
  }
  return Xcol;
}
