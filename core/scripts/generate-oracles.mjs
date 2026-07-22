#!/usr/bin/env node
// Walks every test/*.test.js file, extracts its R-ORACLE-TAG-START/END blocks,
// evaluates each in R (via glmnet/jsonlite), and writes one fixture per source
// file to test/fixtures/<name>.oracle.json. Safe to re-run: existing values are
// recomputed fresh each time (no incremental caching), so this is a from-scratch
// rebuild, not a diff.
import { mkdir, mkdtemp, readFile, readdir, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { basename, dirname, resolve } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { oracleKey } from "../test-support/r-oracle.js";

const FIXTURE_VERSION = 1;

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const testDir = resolve(rootDir, "test");
const fixturesDir = resolve(testDir, "fixtures");

async function walk(dirPath, results = []) {
  for (const entry of await readdir(dirPath, { withFileTypes: true })) {
    const full = resolve(dirPath, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "fixtures") continue;
      await walk(full, results);
    } else if (entry.isFile() && full.endsWith(".test.js")) {
      results.push(full);
    }
  }
  return results;
}

// Runs inside Rscript: reads {r, env} from a JSON spec file, evaluates r in an
// environment built from env, and prints the result back out as JSON.
const rScript = String.raw`
suppressPackageStartupMessages({library(glmnet); library(jsonlite)})

spec <- fromJSON(commandArgs(trailingOnly = TRUE)[1], simplifyVector = TRUE)
value <- eval(parse(text = spec$r), envir = spec$env)
cat(toJSON(value, auto_unbox = TRUE, digits = 17))
`;

async function runCase(spec) {
  const dir = await mkdtemp(resolve(tmpdir(), "faidr-oracle-"));
  const specFile = resolve(dir, "spec.json");
  await writeFile(specFile, JSON.stringify(spec));
  const child = spawnSync("Rscript", ["-e", rScript, specFile], { encoding: "utf8" });
  if (child.status !== 0) {
    throw new Error(`R oracle generation failed for ${spec.id}: ${child.stderr || child.stdout}`);
  }
  return JSON.parse(child.stdout);
}

// Pulls out every R-ORACLE-TAG block. A block is self-contained JS (only
// literals/inline helpers, no references to outer-scope variables, since it's
// re-evaluated standalone here, independent of the test) that defines:
//   const env = {...}; const r = `...`;
function extractTaggedCases(source) {
  const cases = [];
  const matches = source.matchAll(/\/\/ R-ORACLE-TAG-START\s*\n([\s\S]*?)\n\s*\/\/ R-ORACLE-TAG-END/g);
  for (const match of matches) {
    const body = match[1];
    const fn = new Function("String", `"use strict"; ${body}; return { env, r };`);
    const { env, r } = fn(String);
    cases.push({ env, r });
  }
  return cases;
}

function formatValue(value, indent) {
  if (Array.isArray(value)) return JSON.stringify(value);
  if (value && typeof value === "object") return formatObject(value, indent);
  return JSON.stringify(value);
}

function formatObject(value, indent) {
  const keys = Object.keys(value);
  if (keys.length === 0) return "{}";
  const inner = `${indent}  `;
  const lines = ["{"];
  keys.forEach((key, index) => {
    const comma = index === keys.length - 1 ? "" : ",";
    lines.push(`${inner}${JSON.stringify(key)}: ${formatValue(value[key], inner)}${comma}`);
  });
  lines.push(`${indent}}`);
  return lines.join("\n");
}

const files = await walk(testDir);
await mkdir(fixturesDir, { recursive: true });

let totalCases = 0;
let filesWritten = 0;

for (const filePath of files) {
  const source = await readFile(filePath, "utf8");
  const specs = extractTaggedCases(source);
  if (specs.length === 0) continue;

  const name = basename(filePath).replace(/\.test\.js$/, "");
  const outFile = resolve(fixturesDir, `${name}.oracle.json`);
  const oracles = { version: FIXTURE_VERSION, cases: {} };

  for (const spec of specs) {
    const key = oracleKey(spec.r, spec.env);
    if (oracles.cases[key]) continue; // already computed for this file
    oracles.cases[key] = {
      env: spec.env,
      r: spec.r,
      value: await runCase({ env: spec.env, r: spec.r, id: `${name}:${key}` }),
    };
    totalCases++;
  }

  await writeFile(outFile, `${formatObject(oracles, "")}\n`);
  console.log(`wrote ${outFile} (${specs.length} case${specs.length === 1 ? "" : "s"})`);
  filesWritten++;
}

console.log(`done: ${totalCases} case(s) across ${filesWritten} fixture file(s)`);
