// Test 7: prediction helper for fitted logistic coefficients.
import { test } from "node:test";
import { predictLogistic, fitLassoLogistic } from "../js/solver.js";
import { evaluatedInR } from "../test-support/r-oracle.js";
import { matrixFromRows } from "./helpers.js";
import { assertClose, assertAllFinite, assertVectorClose } from "./assertions.js";

test("predictLogistic matches glmnet coefficients", () => {
  // R-ORACLE-TAG-START
  const env = {
    X: [
      [1.0, 0.0, 2.0],
      [0.0, 1.0, 1.0],
      [2.0, 1.0, 0.0],
      [-1.0, 0.5, 1.5],
    ],
    y: [1, 0, 1, 0],
    lambda: 0.01,
  };
  const r = String.raw`
fit <- glmnet(X, y, family = "binomial", alpha = 1, lambda = lambda)
list(
  beta = as.numeric(coef(fit)),
  link = as.numeric(predict(fit, newx = X, type = "link")),
  response = as.numeric(predict(fit, newx = X, type = "response"))
)
`;
  // R-ORACLE-TAG-END

  const X = matrixFromRows(env.X, ["f0", "f1", "f2"]);
  const y = new Float64Array(env.y);

  const fit = fitLassoLogistic(X, y, env.lambda);
  const expected = evaluatedInR(r, env);
  assertAllFinite(fit.beta, "fitBeta:");
  // TODO I suspect the closeness at only 1e-4 is due to glmnet using a different
  // (relative) convergence criterion. Try to tighten comparison here once our
  // implememtation has the same.
  assertClose(fit.beta0, expected.beta[0], 1e-4, "glmnet beta0:");
  assertVectorClose(fit.beta, expected.beta.slice(1), 1e-4, "glmnet beta:");

  const fitEta = predictLogistic(X, fit.beta0, fit.beta, { type: "link" });
  const fitProb = predictLogistic(X, fit.beta0, fit.beta);
  assertAllFinite(fitEta, "fitEta:");
  assertAllFinite(fitProb, "fitProb:");
  assertVectorClose(fitEta, expected.link, 1e-4, "glmnet link");
  assertVectorClose(fitProb, expected.response, 1e-4, "glmnet response");
});
