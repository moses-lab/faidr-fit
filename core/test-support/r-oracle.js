import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const fixtureFile = new URL("../test/fixtures/predict-oracles.json", import.meta.url);
let fixtureCache = null;

function stableStringify(value) {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(",")}]`;
  }
  const keys = Object.keys(value).sort();
  return `{${keys.map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
}

export function oracleKey(r, env) {
  const input = stableStringify({ r, env });
  return createHash("sha256").update(input).digest("hex");
}

function loadFixtureMap() {
  if (fixtureCache) return fixtureCache;
  const raw = JSON.parse(readFileSync(fixtureFile, "utf8"));
  assert.equal(raw.version, 7, `unexpected predict oracle version ${raw.version}`);
  fixtureCache = raw.cases;
  return fixtureCache;
}

export function evaluatedInR(r, env) {
  const key = oracleKey(r, env);
  const oracle = loadFixtureMap()[key];
  if (!oracle) {
    throw new Error(`missing R oracle fixture for key ${key}`);
  }
  return oracle.value;
}