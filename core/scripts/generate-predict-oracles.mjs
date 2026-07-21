#!/usr/bin/env node
import { mkdir, mkdtemp, readFile, readdir, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { oracleKey } from "../test-support/r-oracle.js";

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const testDir = resolve(rootDir, "test");

const tagStart = /\/\/ R-ORACLE-TAG-START\s*\n/g;
const tagEnd = /\n\s*\/\/ R-ORACLE-TAG-END/g;

async function walk(dirPath, results = []) {
  for (const entry of await readdir(dirPath, { withFileTypes: true })) {
    const full = resolve(dirPath, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "fixtures") continue;
      await walk(full, results);
    } else if (entry.isFile() && full.endsWith(".js")) {
      results.push(full);
    }
  }
  return results;
}

const script = String.raw`
suppressPackageStartupMessages({library(glmnet); library(jsonlite)})

spec <- fromJSON(commandArgs(trailingOnly = TRUE)[1], simplifyVector = T)
value <- eval(parse(text = spec$r), envir = spec$env)
cat(toJSON(value, auto_unbox = TRUE, digits = 17))
`;

async function runCase(spec) {
  const dir = await mkdtemp(resolve(tmpdir(), "faidr-predict-oracle-"));
  const specFile = resolve(dir, "spec.json");
  await writeFile(specFile, JSON.stringify(spec));
  const child = spawnSync("Rscript", ["-e", script, specFile], { encoding: "utf8" });
  if (child.status !== 0) {
    throw new Error(
      `R oracle generation failed for ${spec.id}: ${child.stderr || child.stdout}`,
    );
  }
  return JSON.parse(child.stdout);
}

function extractTaggedCases(source, filePath) {
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

function formatFixture(oracles) {
  return formatObject(oracles, "");
}

const files = await walk(testDir);
const oracles = { version: 7, cases: {} };

for (const filePath of files) {
  const source = await readFile(filePath, "utf8");
  const cases = extractTaggedCases(source, filePath);
  for (const spec of cases) {
    const key = oracleKey(spec.r, spec.env);
    oracles.cases[key] = {
      env: spec.env,
      r: spec.r,
      value: await runCase({ env: spec.env, r: spec.r, id: key }),
    };
  }
}

const outFile = resolve(dirname(fileURLToPath(import.meta.url)), "../test/fixtures/predict-oracles.json");
await mkdir(dirname(outFile), { recursive: true });
await writeFile(outFile, `${formatFixture(oracles)}\n`);

console.log(`wrote ${outFile}`);