// Test 7: prediction helper for fitted logistic coefficients.
import { test } from "node:test";
import { fitLassoLogistic } from "../src/fitLassoLogistic.js";
import { predictLogistic } from "../src/predictLogistic.js";
import { evaluatedInR } from "../test-support/r-oracle.js";
import { assertClose, assertVectorClose } from "./assertions.js";

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
    lambda_idx: 41, // 42nd lambda in glmnet's default path of 100 lambdas
                    // gives lambda < 0.01 on this example, to compare with
                    // the previous implementation. The convergence on
                    // coefficients and log odds haven't improved and are
                    // still close to only ~1e-4.
  };
  const r = String.raw`
fit <- glmnet(X, y, family = "binomial", alpha = 1)
lambda <- fit$lambda[lambda_idx+1]
list(
  lambda = lambda,
  beta = as.numeric(coef(fit, s = lambda)),
  link = as.numeric(predict(fit, newx = X, s = lambda, type = "link"))
)
`;
  // R-ORACLE-TAG-END

  const fit = fitLassoLogistic(env.X, env.y);
  const expected = evaluatedInR(r, env);
  const lambda = fit.lambdaPath[env.lambda_idx];
  const coefs = fit.coefficients[env.lambda_idx];
  assertClose(lambda, expected.lambda, 1e-12, "glmnet lambda:");
  // TODO I suspect the closeness at only 1e-4 is due to glmnet using a different
  // (relative) convergence criterion. Try to tighten comparison here once our
  // implementation has the same.
  //
  // Update: The convergence on coefficients and log odds haven't improved and
  // are still close to only ~1e-4. However note that both implementations work
  // with the same convergence threshold at 1e-7. Decreasing it claws back some
  // of the difference.
  assertClose(coefs.beta0, expected.beta[0], 1e-4, "glmnet beta0:");
  assertVectorClose(coefs.beta, expected.beta.slice(1), 1e-4, "glmnet beta");

  const { eta } = predictLogistic(fit, env.X, env.lambda_idx);
  assertVectorClose(eta, expected.link, 1e-4, "glmnet link");
});
