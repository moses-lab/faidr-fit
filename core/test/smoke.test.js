import assert from 'node:assert/strict';
import { fitLassoLogistic } from '../src/fitLassoLogistic.js';

// Synthetic dataset: 200 observations, 6 features. y depends strongly
// on features 0 and 2, weakly on feature 4, not at all on 1, 3, 5.
function makeData(seed = 12345) {
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

  const n = 200;
  const p = 6;
  const trueBeta = [1.5, 0, -1.2, 0, 0.3, 0];
  const trueBeta0 = -0.2;

  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const row = Array.from({ length: p }, () => gaussian());
    let eta = trueBeta0;
    for (let j = 0; j < p; j++) eta += trueBeta[j] * row[j];
    const prob = 1 / (1 + Math.exp(-eta));
    X.push(row);
    y.push(rand() < prob ? 1 : 0);
  }
  return { X, y };
}

const { X, y } = makeData();

const { fit, lambdaPath, df, coefficients } = fitLassoLogistic(X, y, { nlambda: 50 });

// 1. No NaNs/Infs anywhere in the path.
for (const { beta0, beta } of coefficients) {
  assert.ok(Number.isFinite(beta0), 'beta0 must be finite');
  for (const b of beta) assert.ok(Number.isFinite(b), 'beta_j must be finite');
}
console.log('PASS: all coefficients finite across the path');

// 2. At lambda_max (first entry), every slope coefficient is exactly zero.
assert.equal(df[0], 0, `expected df=0 at lambda_max, got ${df[0]}`);
console.log('PASS: df=0 at lambda_max');

// 3. df is non-decreasing as lambda decreases down the path (standard
//    lasso path behavior; not a strict theorem but a reasonable
//    invariant for a path this short and this well-separated).
let nonDecreasing = true;
for (let k = 1; k < df.length; k++) {
  if (df[k] < df[k - 1]) nonDecreasing = false;
}
assert.ok(nonDecreasing, 'df should be non-decreasing along the path');
console.log('PASS: df is non-decreasing along the path');

// 4. The two strong true features (0 and 2) should be selected by the
//    end of the path, with the correct sign.
const last = coefficients[coefficients.length - 1];
assert.ok(last.beta[0] > 0, 'feature 0 (true beta=1.5) should end up positive');
assert.ok(last.beta[2] < 0, 'feature 2 (true beta=-1.2) should end up negative');
console.log('PASS: sign of recovered coefficients matches true signal');

// 5. lambdaPath is strictly decreasing.
for (let k = 1; k < lambdaPath.length; k++) {
  assert.ok(lambdaPath[k] < lambdaPath[k - 1], 'lambda path must be strictly decreasing');
}
console.log('PASS: lambda path strictly decreasing');

console.log('\nAll smoke tests passed.');
console.log(`Path length: ${lambdaPath.length}, final df: ${df[df.length - 1]}`);
