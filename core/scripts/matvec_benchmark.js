#!/usr/bin/env node
/**
 * Benchmark: naive vs column-major-aware matrix-vector multiplication.
 *
 * X is stored column-major: X[j] is a contiguous Float64Array of length n
 * holding feature j across all n samples (matches faidr-lasso's Xstd).
 *
 * Implementation A (given): outer loop i (samples), inner loop j (features).
 *   For fixed i, each inner step jumps to a DIFFERENT Float64Array (X[j])
 *   and touches a single element -> no spatial locality, plus an extra
 *   array-of-arrays dereference every inner iteration.
 *
 * Implementation B: outer loop j (features), inner loop i (samples).
 *   For fixed j, the inner loop scans X[j] sequentially start to end
 *   -> full spatial locality, one dereference per outer iteration.
 *
 * Both do the exact same arithmetic in the exact same order (beta0 first,
 * then j = 0..p-1 accumulated into eta[i]), so outputs should match exactly.
 */

import { performance } from "node:perf_hooks";

// ---------- config (override via CLI: node matvec_benchmark.js n p warmup runs) ----------
const n = Number(process.argv[2]) || 20000;   // samples (rows)
const p = Number(process.argv[3]) || 1000;    // features (columns)
const WARMUP = Number(process.argv[4]) || 5;
const RUNS = Number(process.argv[5]) || 25;

console.log(`n=${n} p=${p}  (~${((n * p * 8) / 1e6).toFixed(1)} MB matrix)  warmup=${WARMUP} runs=${RUNS}\n`);

// ---------- data ----------
function makeColumnMajor(n, p) {
  const X = new Array(p);
  for (let j = 0; j < p; j++) {
    const col = new Float64Array(n);
    for (let i = 0; i < n; i++) col[i] = Math.random() * 2 - 1;
    X[j] = col;
  }
  return X;
}

const X = makeColumnMajor(n, p);
const beta = new Float64Array(p);
for (let j = 0; j < p; j++) beta[j] = Math.random() * 2 - 1;
const beta0 = Math.random();

// ---------- implementation A: row-order access on column-major storage (as given) ----------
function matVecRowOrder(X, beta, beta0, n, p) {
  const eta = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let e = beta0;
    for (let j = 0; j < p; j++) e += X[j][i] * beta[j];
    eta[i] = e;
  }
  return eta;
}

// ---------- implementation B: exploits column-major storage ----------
function matVecColOrder(X, beta, beta0, n, p) {
  const eta = new Float64Array(n);
  eta.fill(beta0);
  for (let j = 0; j < p; j++) {
    const col = X[j];
    const bj = beta[j];
    for (let i = 0; i < n; i++) {
      eta[i] += col[i] * bj;
    }
  }
  return eta;
}

// ---------- correctness check ----------
{
  const a = matVecRowOrder(X, beta, beta0, n, p);
  const b = matVecColOrder(X, beta, beta0, n, p);
  let maxDiff = 0;
  for (let i = 0; i < n; i++) maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
  console.log(`max |A - B| over ${n} entries: ${maxDiff}\n`);
}

// ---------- timing harness ----------
function stats(times) {
  const k = times.length;
  const mean = times.reduce((s, t) => s + t, 0) / k;
  const variance = times.reduce((s, t) => s + (t - mean) ** 2, 0) / (k - 1);
  const sd = Math.sqrt(variance);
  const se = sd / Math.sqrt(k);
  const ci95 = 1.96 * se; // normal approximation
  const sorted = [...times].sort((x, y) => x - y);
  return { mean, sd, se, ci95, median: sorted[k >> 1], min: sorted[0], max: sorted[k - 1] };
}

function benchmark(fn, label) {
  let checksum = 0; // touch the result each call so it can't be dead-code-eliminated
  for (let i = 0; i < WARMUP; i++) {
    const r = fn(X, beta, beta0, n, p);
    checksum += r[0];
  }
  const times = [];
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now();
    const r = fn(X, beta, beta0, n, p);
    const t1 = performance.now();
    times.push(t1 - t0);
    checksum += r[r.length - 1];
  }
  const s = stats(times);
  console.log(
    `${label}: mean=${s.mean.toFixed(3)}ms  ±${s.ci95.toFixed(3)}ms (95% CI)  ` +
    `median=${s.median.toFixed(3)}ms  sd=${s.sd.toFixed(3)}ms  [min ${s.min.toFixed(3)}, max ${s.max.toFixed(3)}]  ` +
    `(checksum ${checksum.toFixed(3)}, ignore)`
  );
  return s;
}

const rowStats = benchmark(matVecRowOrder, 'Row-order    (given, bad access pattern)');
const colStats = benchmark(matVecColOrder, 'Column-order (exploits layout)          ');

console.log(`\nSpeedup (row-order mean / column-order mean): ${(rowStats.mean / colStats.mean).toFixed(2)}x`);
