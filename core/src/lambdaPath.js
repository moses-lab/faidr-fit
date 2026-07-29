/**
 * lambda_max: the smallest lambda at which every slope coefficient is
 * exactly zero.
 *
 * At beta = 0, the fitted probability for every observation equals the
 * intercept-only MLE, ybar (exact for logistic regression: the MLE of
 * an intercept-only model is logit(ybar)). The lasso's KKT stationarity
 * condition says beta_j = 0 remains optimal for feature j exactly as
 * long as the gradient of the unpenalized log-likelihood at beta = 0
 * does not exceed lambda in absolute value. The smallest lambda for
 * which that holds simultaneously for every feature is the max over j
 * of those absolute gradients (Friedman, Hastie & Tibshirani 2010,
 * Sec. 2.5).
 *
 * Xstd is column-major and already standardized (see standardize.js);
 * lambda_max is computed in the same standardized space the path is
 * fit in.
 */
export function computeLambdaMax(Xstd, y) {
  const p = Xstd.length;
  const n = y.length;

  let ybar = 0;
  for (let i = 0; i < n; i++) ybar += y[i];
  ybar /= n;

  let lambdaMax = 0;
  for (let j = 0; j < p; j++) {
    const col = Xstd[j];
    let grad = 0;
    for (let i = 0; i < n; i++) {
      grad += col[i] * (y[i] - ybar);
    }
    grad = Math.abs(grad) / n;
    if (grad > lambdaMax) lambdaMax = grad;
  }

  return lambdaMax;
}

/**
 * The full lambda sequence, log-spaced from lambda_max down to
 * lambda_max * lambdaMinRatio. nlambda and lambdaMinRatio follow
 * glmnet's own defaults (100 values; ratio 1e-4 when n > p, 1e-2
 * otherwise), included here as documented literature defaults rather
 * than arbitrary constants (Friedman, Hastie & Tibshirani 2010, Sec. 2.5).
 */
export function buildLambdaPath(lambdaMax, nFeatures, nObservations, nlambda = 100) {
  const lambdaMinRatio = nObservations > nFeatures ? 1e-4 : 1e-2;
  const lambdas = new Float64Array(nlambda);
  const logMax = Math.log(lambdaMax);
  const logMin = Math.log(lambdaMax * lambdaMinRatio);

  for (let k = 0; k < nlambda; k++) {
    const frac = nlambda === 1 ? 0 : k / (nlambda - 1);
    lambdas[k] = Math.exp(logMax + frac * (logMin - logMax));
  }

  return lambdas;
}
