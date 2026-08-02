import { weightedLassoCD } from './weightedLassoCD.js';
import { computeDeviance, workingResponse } from './logistic.js';

/**
 * Fits the L1-penalized logistic regression at a single lambda via
 * IRLS: repeatedly form the weighted-least-squares approximation to the
 * log-likelihood (Friedman, Hastie & Tibshirani 2010, Sec. 4) and solve
 * the resulting penalized subproblem with weightedLassoCD, until the
 * deviance stops improving by more than thresh * nulldev (glmnet's
 * relative convergence rule, applied here to the outer Newton loop).
 *
 * Step-halving guards against the overshoot a full Newton step can
 * cause when the working weights become extreme near p = 0 or p = 1: a
 * step is only accepted once it does not increase the deviance,
 * otherwise it is repeatedly shrunk toward the previous iterate. The
 * number of times a step can be halved is bounded by how many halvings
 * a double can represent before the step underflows to nothing
 * relative to the current iterate, so the bound is a property of
 * floating-point arithmetic rather than a tuned constant.
 */
export function irlsLogisticLasso(Xstd, y, n, p, lambda, init, nulldev, opts) {
  const { convergenceThreshold, maxIterations, maxHalvings } = opts;

  let beta0 = init.beta0;
  let beta = Float64Array.from(init.beta);
  let deviance = computeDeviance(Xstd, y, n, beta0, beta);

  for (let iter = 0; iter < maxIterations; iter++) {
    const { z, w } = workingResponse(Xstd, y, n, beta0, beta);
    const fullStep = weightedLassoCD(Xstd, z, w, lambda, beta0, beta, n, p, opts);

    let shrink = 1;
    let stepBeta0, stepBeta, newDeviance;
    let halvings = 0;

    do {
      stepBeta0 = beta0 + shrink * (fullStep.beta0 - beta0);
      stepBeta = new Float64Array(p);
      for (let j = 0; j < p; j++) stepBeta[j] = beta[j] + shrink * (fullStep.beta[j] - beta[j]);

      newDeviance = computeDeviance(Xstd, y, n, stepBeta0, stepBeta);
      if (Number.isFinite(newDeviance) && newDeviance <= deviance) break;

      shrink *= 0.5;
      halvings++;
    } while (halvings <= maxHalvings);

    const improvement = deviance - newDeviance;
    beta0 = stepBeta0;
    beta = stepBeta;
    deviance = newDeviance;

    if (improvement < convergenceThreshold * nulldev) {
      return { beta0, beta, deviance, converged: true, iterations: iter + 1 };
    }
  }

  return { beta0, beta, deviance, converged: false, iterations: maxIterations };
}
