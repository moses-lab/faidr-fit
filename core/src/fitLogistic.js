import { matrixInverse } from './math/matrixInverse.js';
import { mean, logit, computeNullDeviance, computeDeviance, workingResponse } from './math/logistic.js';

const CONVERGENCE_THRESHOLD = 1e-7; // see fitLassoLogistic.js for rationale
const MAX_ITERATIONS = 100;
const MAX_HALVINGS = Math.ceil(-Math.log2(Number.EPSILON));

/**
 * Plain (unpenalized) logistic regression via IRLS / Newton-Raphson.
 * Intended to be run on the small feature set fitLassoLogistic has
 * already selected, in order to obtain proper Wald z-statistics
 * (McCullagh & Nelder 1989, Ch. 2 and 4) for those features, which a
 * penalized fit cannot provide directly since shrinkage biases the
 * coefficients.
 *
 * X uses the same column-major layout as fitLassoLogistic: an array of
 * p columns, each a Float64Array of length n.
 */
export function fitLogistic(X, y, opts = {}) {
  const n = X[0].length;
  const p = X.length;

  const convergenceThreshold = opts.convergenceThreshold ?? CONVERGENCE_THRESHOLD;
  const maxIterations = opts.maxIterations ?? MAX_ITERATIONS;
  const maxHalvings = opts.maxHalvings ?? MAX_HALVINGS;

  const nulldev = computeNullDeviance(y, n);

  let beta0 = logit(mean(y, n));
  let beta = new Float64Array(p);
  let deviance = computeDeviance(X, y, n, beta0, beta);

  for (let iter = 0; iter < maxIterations; iter++) {
    const { z, w } = workingResponse(X, y, n, beta0, beta);
    const { coefficients } = weightedLeastSquares(X, z, w, n, p);

    const fullBeta0 = coefficients[0];
    const fullBeta = coefficients.subarray(1);

    let shrink = 1;
    let stepBeta0, stepBeta, newDeviance;
    let halvings = 0;

    do {
      stepBeta0 = beta0 + shrink * (fullBeta0 - beta0);
      stepBeta = new Float64Array(p);
      for (let j = 0; j < p; j++) stepBeta[j] = beta[j] + shrink * (fullBeta[j] - beta[j]);

      newDeviance = computeDeviance(X, y, n, stepBeta0, stepBeta);
      if (Number.isFinite(newDeviance) && newDeviance <= deviance) break;

      shrink *= 0.5;
      halvings++;
    } while (halvings <= maxHalvings);

    const improvement = deviance - newDeviance;
    beta0 = stepBeta0;
    beta = stepBeta;
    deviance = newDeviance;

    if (improvement < convergenceThreshold * nulldev) break;
  }

  // Fisher information evaluated exactly at the converged (beta0, beta),
  // rather than reusing the last IRLS step's information matrix, which
  // was computed one (tiny, near-converged) update earlier.
  const finalWorking = workingResponse(X, y, n, beta0, beta);
  const { informationMatrix } = weightedLeastSquares(X, finalWorking.z, finalWorking.w, n, p);
  const covariance = matrixInverse(informationMatrix);

  const waldZ = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    const standardError = Math.sqrt(covariance[j + 1][j + 1]);
    waldZ[j] = beta[j] / standardError;
  }

  return { beta0, beta, waldZ };
}

/**
 * Builds and solves the weighted normal equations for the augmented
 * design [1 | X] (intercept in column 0), and returns both the solution
 * and the information matrix X_aug^T W X_aug itself, since the latter
 * doubles as the (inverse) asymptotic covariance of the coefficients at
 * convergence.
 */
function weightedLeastSquares(X, z, w, n, p) {
  const dim = p + 1;
  const A = Array.from({ length: dim }, () => new Float64Array(dim));
  const b = new Float64Array(dim);

  let sumW = 0;
  let sumWz = 0;
  for (let i = 0; i < n; i++) {
    sumW += w[i];
    sumWz += w[i] * z[i];
  }
  A[0][0] = sumW;
  b[0] = sumWz;

  for (let j = 0; j < p; j++) {
    const col = X[j];
    let sumWx = 0;
    let sumWxz = 0;
    for (let i = 0; i < n; i++) {
      sumWx += w[i] * col[i];
      sumWxz += w[i] * col[i] * z[i];
    }
    A[0][j + 1] = sumWx;
    A[j + 1][0] = sumWx;
    b[j + 1] = sumWxz;
  }

  for (let j = 0; j < p; j++) {
    const colJ = X[j];
    for (let k = j; k < p; k++) {
      const colK = X[k];
      let sumWxx = 0;
      for (let i = 0; i < n; i++) sumWxx += w[i] * colJ[i] * colK[i];
      A[j + 1][k + 1] = sumWxx;
      A[k + 1][j + 1] = sumWxx;
    }
  }

  const informationMatrix = A;
  const coefficients = solveLinearSystem(A, b);
  return { coefficients, informationMatrix };
}

function solveLinearSystem(A, b) {
  const inv = matrixInverse(A);
  const dim = b.length;
  const x = new Float64Array(dim);
  for (let i = 0; i < dim; i++) {
    let s = 0;
    for (let k = 0; k < dim; k++) s += inv[i][k] * b[k];
    x[i] = s;
  }
  return x;
}
