/**
 * The soft-thresholding operator: the closed-form solution to
 *   argmin_beta  (1/2)(beta - x)^2 + lambda|beta|
 * This is the elementary update coordinate descent repeats for every
 * feature, every sweep (Friedman, Hastie & Tibshirani 2010, eq. 6).
 */
export function softThreshold(x, lambda) {
  if (x > lambda) return x - lambda;
  if (x < -lambda) return x + lambda;
  return 0;
}
