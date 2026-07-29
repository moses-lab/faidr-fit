import {
  sigmoid, clipProbability, deviance, nullModelDeviance, nullIntercept, toColumnMajor,
} from './math/logistic.js';
import { weightedLassoCD } from './weightedLassoCD.js';
import { invertMatrix } from './math/matrixInverse.js';

/**
 * Unpenalized (lambda = 0) logistic regression via Newton-Raphson/IRLS
 * to convergence, intended to be run on the small design matrix of
 * features already selected by fitLassoLogistic. The lasso path gives
 * a shrunken, biased set of coefficients by construction; this refit
 * is what gives calibrated coefficients and their Wald z-statistics
 * (beta_j / se_j), computed from the inverse observed Fisher
 * information at convergence.
 *
 * Reuses weightedLassoCD with lambda = 0, so the outer Newton loop
 * here is the same IRLS scheme as the penalized path fit (same
 * step-halving, same relative convergence check), just without an L1
 * term softening the inner solve.
 */
export function fitLogistic(X, y, opts = {}) {
  const {
    thresh = 1e-7, maxOuterIter = 100, maxInnerIter = 1000, maxHalvings = 20,
  } = opts;

  const n = X.length;
  const p = X[0].length;
  const Xcol = toColumnMajor(X);
  const nullDeviance = nullModelDeviance(y);

  let beta0 = nullIntercept(y);
  let beta = new Float64Array(p);

  let eta = new Float64Array(n).fill(beta0);
  let dev = deviance(y, eta);

  for (let outerIter = 0; outerIter < maxOuterIter; outerIter++) {
    const w = new Float64Array(n);
    const z = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const p_i = clipProbability(sigmoid(eta[i]));
      const w_i = p_i * (1 - p_i);
      w[i] = w_i;
      z[i] = eta[i] + (y[i] - p_i) / w_i;
    }

    const beta0Box = [beta0];
    const candidateBeta = Float64Array.from(beta);
    const { eta: solvedEta } = weightedLassoCD({
      Xstd: Xcol, w, z, beta0: beta0Box, beta: candidateBeta,
      lambda: 0, thresh, scale: nullDeviance, maxIter: maxInnerIter,
    });

    let candidateBeta0 = beta0Box[0];
    let candidateEta = solvedEta;
    let candidateDev = deviance(y, candidateEta);

    let halvings = 0;
    while (candidateDev > dev && halvings < maxHalvings) {
      halvings++;
      const step = Math.pow(0.5, halvings);

      const halvedBeta0 = beta0 + step * (candidateBeta0 - beta0);
      const halvedBeta = new Float64Array(p);
      for (let j = 0; j < p; j++) halvedBeta[j] = beta[j] + step * (candidateBeta[j] - beta[j]);
      const halvedEta = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let e = halvedBeta0;
        for (let j = 0; j < p; j++) e += Xcol[j][i] * halvedBeta[j];
        halvedEta[i] = e;
      }

      candidateBeta0 = halvedBeta0;
      candidateBeta.set(halvedBeta);
      candidateEta = halvedEta;
      candidateDev = deviance(y, candidateEta);
    }

    const devChange = Math.abs(candidateDev - dev);

    beta0 = candidateBeta0;
    beta = candidateBeta;
    eta = candidateEta;
    dev = candidateDev;

    if (devChange < thresh * nullDeviance) break;
  }

  const waldZ = computeWaldZ(X, eta, beta);

  return { beta0, beta, waldZ };
}

/**
 * Wald z-statistics from the inverse observed Fisher information,
 * Xint^T W Xint, where Xint = [1, X] and W = diag(p_i(1-p_i)) at
 * convergence. This is the same quantity R's summary(glm(...)) reports
 * as the "z value" column for a binomial GLM.
 */
function computeWaldZ(X, eta, beta) {
  const n = X.length;
  const p = beta.length;
  const dim = p + 1;

  const info = Array.from({ length: dim }, () => new Float64Array(dim));
  const row = new Float64Array(dim);

  for (let i = 0; i < n; i++) {
    const p_i = clipProbability(sigmoid(eta[i]));
    const w_i = p_i * (1 - p_i);

    row[0] = 1;
    for (let j = 0; j < p; j++) row[j + 1] = X[i][j];

    for (let a = 0; a < dim; a++) {
      const wRowA = w_i * row[a];
      for (let b = 0; b < dim; b++) {
        info[a][b] += wRowA * row[b];
      }
    }
  }

  const cov = invertMatrix(info);
  const waldZ = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    waldZ[j] = beta[j] / Math.sqrt(cov[j + 1][j + 1]);
  }
  return waldZ;
}
