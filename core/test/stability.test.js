import assert from 'node:assert/strict';
import { fitLassoLogistic } from '../src/fitLassoLogistic.js';

function makeData(seed, p) {
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
  const trueBeta = Array.from({ length: p }, (_, j) => (j < 5 ? (j % 2 === 0 ? 1 : -1) * (1 + j * 0.3) : 0));
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const row = Array.from({ length: p }, () => gaussian());
    let eta = 0;
    for (let j = 0; j < p; j++) eta += trueBeta[j] * row[j];
    const prob = 1 / (1 + Math.exp(-eta));
    X.push(row);
    y.push(rand() < prob ? 1 : 0);
  }
  return { X, y };
}

// 1. dfmax stops the path early.
{
  const { X, y } = makeData(7, 20);
  const dfmax = 4;
  const { df } = fitLassoLogistic(X, y, { nlambda: 100, dfmax });
  const last = df[df.length - 1];
  assert.ok(last >= dfmax, `path should run until dfmax is reached, got final df=${last}`);
  // every df strictly before the last entry must be below dfmax,
  // otherwise the path should have stopped sooner.
  for (let k = 0; k < df.length - 1; k++) {
    assert.ok(df[k] < dfmax, `df at step ${k} is ${df[k]}, path should have stopped at dfmax=${dfmax} already`);
  }
  console.log(`PASS: dfmax=${dfmax} stops the path (path length ${df.length}, final df ${last})`);
}

// 2. Near-perfect separation: one feature almost fully determines y.
// This is exactly the situation where naive IRLS diverges (p -> 0/1,
// weights -> 0, working response -> +/-infinity). Step-halving and
// probability clipping should keep the fit finite and stable.
{
  let s = 55;
  const rand = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  const n = 150;
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const x0 = rand() * 10 - 5;
    const x1 = rand() * 2 - 1; // unrelated noise feature
    X.push([x0, x1]);
    // near-perfect separation on x0, tiny noise chance of a flip
    y.push(x0 > 0 ? (rand() < 0.98 ? 1 : 0) : (rand() < 0.98 ? 0 : 1));
  }

  const { coefficients, df } = fitLassoLogistic(X, y, { nlambda: 40 });
  for (const { beta0, beta } of coefficients) {
    assert.ok(Number.isFinite(beta0), 'beta0 must stay finite under near-separation');
    for (const b of beta) assert.ok(Number.isFinite(b), 'beta must stay finite under near-separation');
  }
  assert.ok(df[df.length - 1] >= 1, 'the separating feature should eventually be selected');
  console.log('PASS: near-separable data stays numerically finite across the whole path (step-halving + probability clipping working)');
}

// 3. Constant (zero-variance) column is handled without division by
// zero and never enters the model.
{
  const { X, y } = makeData(3, 5);
  for (const row of X) row.push(7); // append a constant column
  const { coefficients } = fitLassoLogistic(X, y, { nlambda: 20 });
  const last = coefficients[coefficients.length - 1];
  assert.equal(last.beta[5], 0, 'constant column must never get a nonzero coefficient');
  assert.ok(Number.isFinite(last.beta0), 'beta0 must remain finite with a constant column present');
  console.log('PASS: constant column handled safely (zero coefficient, no NaN)');
}

console.log('\nAll stability/dfmax tests passed.');
