/**
 * Predicts log-odds (eta) for new observations using the coefficients
 * at a single point (lambdaIdx) along a fitLassoLogistic path.
 *
 * newx uses the same column-major layout as fitLassoLogistic: an array
 * of p columns, each a Float64Array of length m (the number of new
 * observations). fit is exactly the object fitLassoLogistic returns.
 *
 * Returns eta (the predicted log-odds for each new observation) and the
 * actual lambda value at lambdaIdx, for inspection/debugging.
 */
export function predictLogistic(fit, newx, lambdaIdx) {
  const beta0 = fit.coefficients[lambdaIdx].beta0;
  const beta = fit.coefficients[lambdaIdx].beta;
  const p = beta.length;
  const m = newx[0].length;

  const eta = new Float64Array(m).fill(beta0);
  for (let j = 0; j < p; j++) {
    const coef = beta[j];
    if (coef === 0) continue;
    const col = newx[j];
    for (let i = 0; i < m; i++) eta[i] += col[i] * coef;
  }

  return { eta, lambda: fit.lambdaPath[lambdaIdx] };
}
