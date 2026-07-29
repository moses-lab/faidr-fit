import { sigmoid, clipProbability, deviance } from './math/logistic.js';
import { weightedLassoCD } from './weightedLassoCD.js';

/**
 * Solves the L1-penalized logistic regression problem at a single
 * lambda, warm-started from (beta0Init, betaInit), by alternating:
 *
 *   1. an outer Newton step: linearize the log-likelihood around the
 *      current fit to get IRLS weights w and working response z
 *      (Friedman, Hastie & Tibshirani 2010, eq. 8-9);
 *   2. an inner coordinate descent solve of the resulting weighted
 *      lasso subproblem, to convergence (weightedLassoCD.js).
 *
 * Step-halving guards the outer step: a Newton step is only kept if it
 * actually decreases the true (non-quadratic-approximated) deviance.
 * If the quadratic approximation overshot, the step is repeatedly
 * halved, interpolating back toward the previous fit, until deviance
 * decreases or the halving budget is exhausted. This is what prevents
 * the divergence that a naive IRLS implementation is prone to when the
 * quadratic approximation is poor (e.g. near-separable data).
 *
 * Convergence of the outer loop is judged the same way as the inner
 * loop: relative to nullDeviance, not an absolute cutoff.
 */
export function irlsLogisticLasso({
  Xstd, y, lambda, beta0Init, betaInit, nullDeviance,
  thresh = 1e-7, maxOuterIter = 100, maxInnerIter = 1000, maxHalvings = 20,
}) {
  const n = y.length;
  const p = Xstd.length;

  let beta0 = beta0Init;
  let beta = Float64Array.from(betaInit);

  let eta = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let e = beta0;
    for (let j = 0; j < p; j++) e += Xstd[j][i] * beta[j];
    eta[i] = e;
  }

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
      Xstd, w, z, beta0: beta0Box, beta: candidateBeta,
      lambda, thresh, scale: nullDeviance, maxIter: maxInnerIter,
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
      for (let j = 0; j < p; j++) {
        halvedBeta[j] = beta[j] + step * (candidateBeta[j] - beta[j]);
      }
      const halvedEta = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let e = halvedBeta0;
        for (let j = 0; j < p; j++) e += Xstd[j][i] * halvedBeta[j];
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

  return { beta0, beta, deviance: dev, eta };
}
