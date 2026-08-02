import assert from "node:assert/strict";

export function assertClose(a, b, tol, msg = "") {
  assert.ok(
    Math.abs(a - b) <= tol,
    `${msg} expected ${a} ≈ ${b} (|Δ|=${Math.abs(a - b).toExponential(3)} > ${tol})`,
  );
}

export function assertVectorClose(actual, expected, tol, label) {
  assert.equal(actual.length, expected.length, `${label} length`);
  for (let i = 0; i < actual.length; i++) {
    assertClose(actual[i], expected[i], tol, `${label} ${i}:`);
  }
}
