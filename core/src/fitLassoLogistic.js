import { standardize, unstandardizeCoefficients } from './math/standardize.js';
import { computeLambdaMax, buildLambdaPath } from './math/lambdaPath.js';
import { irlsLogisticLasso } from './math/irlsLogisticLasso.js';
import { computeNullDeviance, nullIntercept } from './math/logistic.js';

const DEFAULT_NLAMBDA = 100;

// glmnet's default `thresh`: the outer IRLS loop stops once an
// iteration improves the deviance by less than this fraction of the
// null deviance.
const DEFAULT_CONVERGENCE_THRESHOLD = 1e-7;

// A generous safety bound against non-termination on pathological
// input; ordinary fits converge in a handful of iterations, with
// termination actually governed by DEFAULT_CONVERGENCE_THRESHOLD above.
const DEFAULT_MAX_ITERATIONS = 100;

// Exhausts double precision: 2^-52 is the smallest step-halving still
// representable relative to a unit-scale iterate, so no tuned cutoff is
// needed for how many times step-halving may retry.
const DEFAULT_MAX_HALVINGS = Math.ceil(-Math.log2(Number.EPSILON));

/**
 * Fits an entire L1-penalized logistic regression path, from
 * lambda_max (the smallest lambda with an all-zero solution) down to
 * lambda_min, warm-starting each fit from the previous lambda's
 * solution (Friedman, Hastie & Tibshirani 2010, Sec. 2.5).
 *
 * X is a design matrix in column-major layout: an array of p columns,
 * each a Float64Array of length n. y is a length-n array/typed array of
 * 0/1 labels.
 *
 * opts:
 *   nlambda - number of lambdas in the path (default 100)
 *   dfmax   - stop the path once a fit selects this many features
 *
 * Returns { lambda, df, beta0, beta, n, p }, where lambda/df/beta0 are
 * arrays indexed by path position and beta is an array of Float64Array
 * coefficient vectors (original X scale, not standardized). This
 * object is exactly the `fit` argument predictLogistic expects.
 */
export function fitLassoLogistic(X, y, opts = {}) {
  const n = X[0].length;
  const p = X.length;

  const nlambda = opts.nlambda ?? DEFAULT_NLAMBDA;
  const dfmax = opts.dfmax ?? p;
  const solverOpts = {
    convergenceThreshold: opts.convergenceThreshold ?? DEFAULT_CONVERGENCE_THRESHOLD,
    maxIterations: opts.maxIterations ?? DEFAULT_MAX_ITERATIONS,
    maxHalvings: opts.maxHalvings ?? DEFAULT_MAX_HALVINGS,
  };

  const { Xstd, means, sds } = standardize(X, n, p);
  const nulldev = computeNullDeviance(y, n);
  const lambdaMax = computeLambdaMax(Xstd, y, n, p);
  const lambdaPath = buildLambdaPath(lambdaMax, n, p, nlambda);

  // Warm start at lambda_max: the null model, beta = 0 and beta0 at the
  // intercept-only MLE.
  let beta0 = nullIntercept(y, n);
  let beta = new Float64Array(p);

  const lambdaOut = [lambdaMax];
  const dfOut = [0];
  const coefficients = [{ beta0, beta }];

  for (let i = 1; i < lambdaPath.length; i++) {
    const lambda = lambdaPath[i];
    const fit = irlsLogisticLasso(Xstd, y, n, p, lambda, { beta0, beta }, nulldev, solverOpts);
    beta0 = fit.beta0;
    beta = fit.beta;

    const unstd = unstandardizeCoefficients(beta0, beta, means, sds);
    const df = countNonzero(beta);

    lambdaOut.push(lambda);
    dfOut.push(df);
    coefficients.push({ beta0: unstd.beta0, beta: unstd.beta });

    if (df >= dfmax) break;
  }

  return { lambdaPath: lambdaOut, df: dfOut, coefficients };
}

function countNonzero(beta) {
  let count = 0;
  for (let j = 0; j < beta.length; j++) if (beta[j] !== 0) count++;
  return count;
}
