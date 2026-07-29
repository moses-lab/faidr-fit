import assert from 'node:assert/strict';
import { fitLogistic } from '../src/fitLogistic.js';
import { predictLogistic } from '../src/predictLogistic.js';
import { sigmoid } from '../src/math/logistic.js';

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
  const n = 500;
  const trueBeta0 = -0.3;
  const trueBeta = [2.0, -1.0]; // feature 0 strong positive, feature 1 moderate negative
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const row = [gaussian(), gaussian()];
    let eta = trueBeta0;
    for (let j = 0; j < 2; j++) eta += trueBeta[j] * row[j];
    const prob = sigmoid(eta);
    X.push(row);
    y.push(rand() < prob ? 1 : 0);
  }
  return { X, y };
}

const { X, y } = makeData(42);
const { beta0, beta, waldZ } = fitLogistic(X, y);

// 1. Unpenalized fit should be an unbiased-ish estimate: recovered
// coefficients should be in the right ballpark and the right sign.
assert.ok(beta[0] > 1.0 && beta[0] < 3.0, `beta[0]=${beta[0]} should be near true value 2.0`);
assert.ok(beta[1] < -0.3 && beta[1] > -2.0, `beta[1]=${beta[1]} should be near true value -1.0`);
console.log('PASS: unpenalized coefficients recover true signal within a plausible range');

// 2. z-stats should reflect that feature 0's effect is stronger and
// more precisely estimated than feature 1's in this design.
assert.ok(Math.abs(waldZ[0]) > Math.abs(waldZ[1]) * 0.8,
  `expected |z0| to be comparably large or larger than |z1|, got z0=${waldZ[0]}, z1=${waldZ[1]}`);
console.log(`PASS: waldZ = [${waldZ[0].toFixed(2)}, ${waldZ[1].toFixed(2)}], signs and magnitudes are sensible`);

// 3. Zero-gradient check (KKT condition for the unpenalized case is
// just stationarity: gradient of the log-likelihood is ~0 at every
// coefficient, including the intercept).
{
  let eta = X.map((row) => beta0 + row[0] * beta[0] + row[1] * beta[1]);
  let gradIntercept = 0;
  const gradBeta = [0, 0];
  for (let i = 0; i < X.length; i++) {
    const resid = y[i] - sigmoid(eta[i]);
    gradIntercept += resid;
    gradBeta[0] += resid * X[i][0];
    gradBeta[1] += resid * X[i][1];
  }
  const scale = X.length;
  assert.ok(Math.abs(gradIntercept / scale) < 1e-4, `intercept gradient should be ~0, got ${gradIntercept / scale}`);
  assert.ok(Math.abs(gradBeta[0] / scale) < 1e-4, `beta0 gradient should be ~0, got ${gradBeta[0] / scale}`);
  assert.ok(Math.abs(gradBeta[1] / scale) < 1e-4, `beta1 gradient should be ~0, got ${gradBeta[1] / scale}`);
  console.log('PASS: unpenalized fit is a stationary point (gradient ~0)');
}

// 4. predictLogistic round-trips correctly against a fitLassoLogistic
// fit shape.
const fakeFit = { lambdaPath: [0], coefficients: [{ beta0, beta }] };
const { eta: predEta, lambda: predLambda } = predictLogistic(fakeFit, X.slice(0, 5), 0);
assert.equal(predLambda, 0);
for (let i = 0; i < 5; i++) {
  const expected = beta0 + X[i][0] * beta[0] + X[i][1] * beta[1];
  assert.ok(Math.abs(predEta[i] - expected) < 1e-10, 'predictLogistic eta must match manual computation');
}
console.log('PASS: predictLogistic matches manual eta computation');

console.log('\nAll fitLogistic/predictLogistic tests passed.');
