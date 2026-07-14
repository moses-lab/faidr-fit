// Small dense linear algebra for the unpenalised refit: Cholesky factorisation
// of a symmetric positive-definite matrix, plus solve and inverse. The matrices
// here are tiny (selected features + intercept, ~16 x 16), so a plain Cholesky
// is ample and keeps us dependency-free (and directly comparable to the WASM port).

// A = L Lᵀ for symmetric positive-definite A (array of Float64Array rows).
// Returns lower-triangular L. Throws if A is not positive-definite.
export function cholesky(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) throw new Error("matrix not positive-definite");
        L[i][j] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  return L;
}

// Solve A x = b given L = chol(A). b is length n; returns x.
export function cholSolve(L, b) {
  const n = L.length;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

// Inverse of the SPD matrix whose Cholesky is L (solve against identity columns).
export function cholInverse(L) {
  const n = L.length;
  const inv = Array.from({ length: n }, () => new Float64Array(n));
  const e = new Float64Array(n);
  for (let c = 0; c < n; c++) {
    e.fill(0); e[c] = 1;
    const col = cholSolve(L, e);
    for (let r = 0; r < n; r++) inv[r][c] = col[r];
  }
  return inv;
}
