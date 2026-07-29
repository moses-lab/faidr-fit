/**
 * Gauss-Jordan inversion with partial pivoting. Used only for the
 * observed Fisher information matrix in fitLogistic.js, whose
 * dimension is (number of lasso-selected features + 1) and is
 * therefore small: a direct, dependency-free inversion is simpler to
 * review than pulling in a linear algebra library for one small
 * symmetric positive-definite matrix.
 */
export function invertMatrix(M) {
  const dim = M.length;
  const A = M.map((row) => Float64Array.from(row));
  const inv = Array.from({ length: dim }, (_, i) => {
    const row = new Float64Array(dim);
    row[i] = 1;
    return row;
  });

  for (let col = 0; col < dim; col++) {
    let pivotRow = col;
    for (let r = col + 1; r < dim; r++) {
      if (Math.abs(A[r][col]) > Math.abs(A[pivotRow][col])) pivotRow = r;
    }
    if (pivotRow !== col) {
      [A[col], A[pivotRow]] = [A[pivotRow], A[col]];
      [inv[col], inv[pivotRow]] = [inv[pivotRow], inv[col]];
    }

    const pivot = A[col][col];
    for (let c = 0; c < dim; c++) {
      A[col][c] /= pivot;
      inv[col][c] /= pivot;
    }

    for (let r = 0; r < dim; r++) {
      if (r === col) continue;
      const factor = A[r][col];
      if (factor === 0) continue;
      for (let c = 0; c < dim; c++) {
        A[r][c] -= factor * A[col][c];
        inv[r][c] -= factor * inv[col][c];
      }
    }
  }

  return inv;
}
