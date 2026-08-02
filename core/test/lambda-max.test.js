import { test } from "node:test";
import { fitLassoLogistic } from "../src/index.js";
import { assertVectorClose } from "../test-support/assertions.js";
import { evaluatedInR } from "../test-support/r-oracle.js";
import { t } from "../test-support/util.js";

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
    nlambda: 5,
  };
  const r = String.raw`
fit <- glmnet(X, y, family = "binomial", alpha = 1, nlambda = nlambda)
list(glmnetPath = fit$lambda)
`;
  // R-ORACLE-TAG-END

  const opts = {
    nlambda: env.nlambda,
    dfmax: 99,  // Force it to continue until nlambda is reached
  };
  const fit = fitLassoLogistic(t(env.X), env.y, opts);
  const expected = evaluatedInR(r, env);
  assertVectorClose(fit.lambdaPath, expected.glmnetPath, 1e-12,"lambdaPath vs glmnet path:");
});
