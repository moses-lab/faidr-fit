import assert from 'node:assert/strict';
import { toColumnMajor, sigmoid, nullModelDeviance, nullIntercept } from '../src/math/logistic.js';
import { standardize } from '../src/standardize.js';
import { computeLambdaMax, buildLambdaPath } from '../src/lambdaPath.js';
import { irlsLogisticLasso } from '../src/irlsLogisticLasso.js';

/**
 * This test does not compare against glmnet or any other reference
 * implementation. Instead it checks the solution against the lasso's
 * own optimality condition: for the objective
 *   L(beta0,beta) = -(1/N) loglik(beta0,beta) + lambda * sum|beta_j|
 * a point is optimal iff, writing g_j for the gradient of the
 * unpenalized term w.r.t. beta_j:
 *   beta_j != 0  =>  g_j = -lambda * sign(beta_j)
 *   beta_j == 0  =>  |g_j| <= lambda
 * This is checked in the standardized coordinate system the solver
 * actually optimizes in.
 */

function makeData(seed) {
  let s = seed;
  const rand = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  const gaussian = () => {
    const u1 = rand() || 1e-9;
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
  const n = 300;
  const p = 8;
  const trueBeta = [2, 0, -1.5, 0, 0.8, 0, -0.5, 0];
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const row = Array.from({ length: p }, () => gaussian());
    let eta = 0.1;
    for (let j = 0; j < p; j++) eta += trueBeta[j] * row[j];
    const prob = 1 / (1 + Math.exp(-eta));
    X.push(row);
    y.push(rand() < prob ? 1 : 0);
  }
  return { X, y };
}

const { X, y } = makeData(999);
const Xcol = toColumnMajor(X);
const { Xstd } = standardize(Xcol);
const p = Xstd.length;
const lambdaMax = computeLambdaMax(Xstd, y);
const lambdaPath = buildLambdaPath(lambdaMax, p, X.length, 30);
const nullDeviance = nullModelDeviance(y);

let beta0 = nullIntercept(y);
let beta = new Float64Array(p);

const KKT_TOLERANCE = 1e-4; // slack for floating point + finite thresh,
// not a modeling threshold: the solver's own convergence thresh (1e-7,
// relative) leaves a small residual gap in the KKT conditions, this is
// just how much of that gap the test tolerates.

let checkedCount = 0;

for (let k = 0; k < lambdaPath.length; k++) {
  const lambda = lambdaPath[k];
  const result = irlsLogisticLasso({
    Xstd, y, lambda, beta0Init: beta0, betaInit: beta, nullDeviance,
  });
  beta0 = result.beta0;
  beta = result.beta;

  // Gradient of the unpenalized (1/N) negative log-likelihood at the
  // actual fitted probabilities (not the IRLS working values).
  const n = y.length;
  const eta = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let e = beta0;
    for (let j = 0; j < p; j++) e += Xstd[j][i] * beta[j];
    eta[i] = e;
  }
  const grad = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    const col = Xstd[j];
    let g = 0;
    for (let i = 0; i < n; i++) {
      const prob = sigmoid(eta[i]);
      g += col[i] * (y[i] - prob);
    }
    grad[j] = -g / n;
  }

  for (let j = 0; j < p; j++) {
    checkedCount++;
    if (beta[j] !== 0) {
      const expected = -lambda * Math.sign(beta[j]);
      assert.ok(
        Math.abs(grad[j] - expected) < KKT_TOLERANCE,
        `lambda=${lambda.toFixed(5)} feature ${j}: active coefficient violates stationarity `
        + `(grad=${grad[j].toFixed(6)}, expected=${expected.toFixed(6)})`,
      );
    } else {
      assert.ok(
        Math.abs(grad[j]) <= lambda + KKT_TOLERANCE,
        `lambda=${lambda.toFixed(5)} feature ${j}: zero coefficient violates KKT bound `
        + `(|grad|=${Math.abs(grad[j]).toFixed(6)}, lambda=${lambda.toFixed(6)})`,
      );
    }
  }
}

console.log(`PASS: KKT conditions hold at every lambda on the path (${checkedCount} coefficient checks)`);
