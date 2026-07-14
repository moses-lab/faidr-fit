// Deterministic random-problem generators for the property-based tests. Seeded
// so CI is reproducible; vary N, p (including N<p), column correlation, and
// per-column scale so the KKT sweep exercises screening, active-set, and
// standardization paths.
import { rng, gauss, matrixFromRows } from "./helpers.js";

// One random (X, y) problem. `rand` is a seeded PRNG so callers control the seed.
//  - rho in [0,1): shared latent factor injected into every column (correlation)
//  - scales: per-column multiplicative spread so columns live on mixed scales
export function makeProblem(rand, { n, p, rho = 0, mixedScale = false } = {}) {
  const scales = new Float64Array(p);
  for (let j = 0; j < p; j++) scales[j] = mixedScale ? Math.exp((rand() - 0.5) * 6) : 1;

  const rows = [];
  // a couple of "true" directions so y carries real signal (not pure noise)
  const w = new Float64Array(p);
  for (let j = 0; j < Math.min(p, 3); j++) w[j] = gauss(rand);

  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const latent = gauss(rand);
    const row = new Array(p);
    let signal = 0;
    for (let j = 0; j < p; j++) {
      const raw = rho * latent + Math.sqrt(1 - rho * rho) * gauss(rand);
      row[j] = raw * scales[j];
      // build the signal on the standardized direction so scale doesn't dominate
      signal += w[j] * raw;
    }
    rows.push(row);
    const pr = 1 / (1 + Math.exp(-(signal + 0.3 * gauss(rand))));
    y[i] = rand() < pr ? 1 : 0;
  }
  return { X: matrixFromRows(rows), y };
}

// Reject degenerate label vectors (all 0 or all 1) so the fit is well-posed.
export function makeNondegenerate(rand, opts) {
  for (let attempt = 0; attempt < 50; attempt++) {
    const prob = makeProblem(rand, opts);
    const first = prob.y[0];
    if (!prob.y.every((v) => v === first)) return prob;
  }
  throw new Error("could not generate a non-degenerate label vector");
}

export { rng };
