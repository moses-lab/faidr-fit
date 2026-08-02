/**
 * Soft-thresholding operator S(z, gamma) = sign(z) * max(|z| - gamma, 0).
 *
 * This is the proximal operator of the L1 penalty and is the closed-form
 * coordinatewise minimizer used throughout coordinate descent for the
 * lasso (Friedman, Hastie & Tibshirani 2010, Eq. 8; Hastie, Tibshirani &
 * Wainwright 2015, Sec. 5.3).
 */
export function softThreshold(z, gamma) {
  if (z > gamma) return z - gamma;
  if (z < -gamma) return z + gamma;
  return 0;
}
