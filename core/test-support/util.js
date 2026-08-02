/** Converts a row-major design matrix (X[i][j], observations x features)
 * into column-major storage (Xcol[j][i]), which is the access pattern
 * coordinate descent needs: for a fixed feature j, scan all i. */
export function t(X) {
  const n = X.length;
  const p = X[0].length;
  const Xcol = Array.from({ length: p }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      Xcol[j][i] = X[i][j];
    }
  }
  return Xcol;
}
