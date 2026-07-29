import { test } from "node:test";
import { fitLassoLogistic } from "../src/index.js";
import { assertVectorClose } from "./assertions.js";
import { evaluatedInR } from "../test-support/r-oracle.js";

test("lambdaPath matches glmnet's own lambda path (fit$lambda)", () => {
  // R-ORACLE-TAG-START
  const env = {
    X: [
      [1.0, -1.0],
      [2.0, 0.5],
      [0.0, 3.0],
      [-1.0, 0.5]
    ],
    y: [1, 0, 1, 0],  // Note: glmnet needs at least 2 of each class to avoid a degenerate fit
  };
  const r = String.raw`
fit <- glmnet(X, y, family = "binomial", alpha = 1, nlambda = 5)
list(glmnetPath = fit$lambda)
`;
  // R-ORACLE-TAG-END

  const fit = fitLassoLogistic(env.X, env.y, { nlambda: 5 });
  const expected = evaluatedInR(r, env);
  assertVectorClose(fit.lambdaPath, expected.glmnetPath, 1e-12,"lambdaPath vs glmnet path:");
});
