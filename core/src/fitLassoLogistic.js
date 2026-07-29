import { toColumnMajor, nullModelDeviance, nullIntercept } from './math/logistic.js';
import { standardize, unstandardizeCoefficients } from './standardize.js';
import { computeLambdaMax, buildLambdaPath } from './lambdaPath.js';
import { irlsLogisticLasso } from './irlsLogisticLasso.js';

/**
 * Fits the full L1-penalized logistic regression path: from lambda_max
 * (where every slope coefficient is exactly zero) down to a small
 * lambda, warm-starting each fit from the previous, larger lambda's
 * solution (Friedman, Hastie & Tibshirani 2010, Sec 2.5, "pathwise
 * coordinate descent"). Warm starts are what make fitting an entire
 * path cheap: consecutive lambdas differ only slightly, so each
 * solution is typically just a few coordinate descent sweeps away from
 * the last one, rather than requiring a fresh solve from zero.
 *
 * X: design matrix, row-major (X[i][j] = observation i, feature j).
 * y: 0/1 labels, one per row of X.
 * opts.nlambda: number of lambdas in the (full) path (default 100,
 *   matching glmnet's default).
 * opts.dfmax: stop walking the path once a fit selects this many
 *   features (default p + 1, i.e. effectively no early stop).
 * opts.thresh: relative convergence threshold, applied throughout both
 *   the outer Newton loop and inner coordinate descent loop (default
 *   1e-7, matching glmnet's `thresh`).
 */
export function fitLassoLogistic(X, y, opts = {}) {
  const p = X[0].length;
  const { nlambda = 100, dfmax = p + 1, thresh = 1e-7 } = opts;

  const Xcol = toColumnMajor(X);
  const { Xstd, mean, sd } = standardize(Xcol);

  const lambdaMax = computeLambdaMax(Xstd, y);
  const fullLambdaPath = buildLambdaPath(lambdaMax, p, X.length, nlambda);
  const nullDeviance = nullModelDeviance(y);

  let beta0 = nullIntercept(y);
  let beta = new Float64Array(p);

  const lambdaPath = [];
  const df = [];
  const coefficients = [];

  for (let k = 0; k < fullLambdaPath.length; k++) {
    const lambda = fullLambdaPath[k];

    const result = irlsLogisticLasso({
      Xstd, y, lambda,
      beta0Init: beta0, betaInit: beta,
      nullDeviance, thresh,
    });

    beta0 = result.beta0;
    beta = result.beta;

    const { beta0: beta0Orig, beta: betaOrig } = unstandardizeCoefficients(beta0, beta, mean, sd);

    let selected = 0;
    for (let j = 0; j < p; j++) if (betaOrig[j] !== 0) selected++;

    lambdaPath.push(lambda);
    df.push(selected);
    coefficients.push({ beta0: beta0Orig, beta: betaOrig });

    if (selected >= dfmax) break;
  }

  return { lambdaPath, df, coefficients };
}
