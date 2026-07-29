/**
 * Predicts log-odds (eta) for new observations at a given point on the
 * lambda path. Coefficients in `fit` are already on the original scale
 * of X (fitLassoLogistic unstandardizes before returning), so `newx`
 * should be supplied on that same original scale, row-major
 * (newx[i][j]) like the X passed to fitLassoLogistic.
 */
export function predictLogistic(fit, newx, lambdaIdx) {
  const { beta0, beta } = fit.coefficients[lambdaIdx];
  const lambda = fit.lambdaPath[lambdaIdx];

  const eta = newx.map((row) => {
    let e = beta0;
    for (let j = 0; j < beta.length; j++) e += row[j] * beta[j];
    return e;
  });

  return { eta, lambda };
}
