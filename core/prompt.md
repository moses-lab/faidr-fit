Our goal here is to build a pure, self contained, JavaScript, penalized (lasso) logistic regression library. The constraints are as follows.

# General
- It should follow conventions in Friedman et al's 2010 "Regularization Paths for Generalized Linear Models via Coordinate Descent" and Hastie et al's 2015 "Statistical Learning with Sparsity"
  - As corollary, it should follow the conventions adopted in the glmnet's implementation
- It should focus on L1 penalized logistic regression only
- It should be as fast as possible
- It should be very readable, especially to a human reviewer
  - e. g., avoid one liner standard deviation loops in the middle of a function
- It should not have arbitrary thresholds in the code

# Public API
## `fitLassoLogistic(X,y,opts={})`
Given a design matrix `X` (given column-major), and 0/1 labels in vector array `y`, compute a L1 penalized logistic regression fit. Not just one, but an entire lambda path, starting at lambda_{max}. It is essential to exploit warm-start for maximum performance.

`opts`:
- `nlambda`: the number of desired lambdas in a (full) lambda path
- `dfmax`: stop path once we have a fit with this many features selected

Returns:
- The lambda path. First element should be lambda max.
- A corresponding df array of number of selected features.
- A corresponding array with fitted coefficients.

## `fitLogistic(X,y,opts={})`
`X` is again a design matrix, but it's a expected to be one with only desired features (first selected with `fitLassoLogistic`). `fitLogistic` is a plain logistic regression (not penalized). Its goal is to get proper z statistics for the strength of features and should return those.

Returns:
- beta0 and beta array, the coefficients
- waldZ array of z statistics

## `predictLogistic(fit,newx,lambda_idx)`
Returns:
- The log odds (eta vector) predicted by the coefficients at lambda_idx within the fit object.
- The actual lambda value at that idx.

The actual lambda value will be more for inspection/debugging.

# Behaviour
The numerical stability is super important. Glmnet has additional techniques to avoid divergence, these need to be in as well. Pieces that were identified:

- Converged threshold needs to be relative
- Step-halving for both fast convergence and preventing overshooting
- Active set optimization so we're only doing coordinate descent on non zero coefficients

# Testing
Write a few tests checking major invariants of the algorithm. Use node's native test() framework.

# Gotchas
All matrices are expected to be given (and used internally) in column-major order. In addition, all matrix-matrix, matrix-vector multiplications should exploit that ordering for maximum cache locality. This is an example of what *NOT* to do:

```
  const eta = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let e = beta0;
    for (let j = 0; j < p; j++) e += Xstd[j][i] * beta[j];
    eta[i] = e;
  }
```
