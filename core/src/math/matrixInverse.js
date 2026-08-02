/**
 * Inverts a small square matrix via Gauss-Jordan elimination with
 * partial pivoting. This is only used for the (p+1) x (p+1) information
 * matrix in fitLogistic, where p is the small number of features
 * fitLassoLogistic has already selected, so a plain dense inverse (not
 * an iterative or sparse method) is the appropriate, readable choice.
 *
 * A is an array of p rows, each an array/typed array of p numbers.
 * Returns the inverse in the same shape.
 */
export function matrixInverse(A) {
  const n = A.length;

  // Build the augmented [A | I] matrix.
  const M = A.map((row, i) => {
    const augmented = new Float64Array(2 * n);
    for (let k = 0; k < n; k++) augmented[k] = row[k];
    augmented[n + i] = 1;
    return augmented;
  });

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivotRow][col])) pivotRow = r;
    }
    if (pivotRow !== col) {
      const swap = M[col];
      M[col] = M[pivotRow];
      M[pivotRow] = swap;
    }

    const pivot = M[col][col];
    if (pivot === 0) {
      throw new Error('matrixInverse: matrix is singular');
    }
    for (let k = 0; k < 2 * n; k++) M[col][k] /= pivot;

    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      if (factor === 0) continue;
      for (let k = 0; k < 2 * n; k++) M[r][k] -= factor * M[col][k];
    }
  }

  return M.map((row) => row.slice(n, 2 * n));
}
