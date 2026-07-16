// One-time data preparation: read the CDT-format ES matrix and emit the compact
// blobs the app ships. Run manually when the source data changes:
//
//   node app/prep-data.js /path/to/HUMAN_ES.txt
//
// Output (into app/.cache, git-ignored):
//   data.bin.gz  int z-scores, column-major, gzipped
//   meta.json    { n, p, bytes, scale, features[p], ids[n] }
//
// The matrix is 19,032 IDRs × 144 evolutionary-signature features. z-scores are
// clamped to ±15 in the source. Two quantizations, chosen by the second CLI arg
// (`16` default, or `8`):
//   16-bit: ×1000 → lossless to 3 decimals (±15000 « ±32767), ~4.7 MB gz.
//   8-bit:  ×8    → 0.125 resolution over ±15 (fits int8's ±127), ~half the size.
// Column-major layout lets the loader slice each feature column contiguously.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const BITS = process.argv[3] === "8" ? 8 : 16;
const SCALE = BITS === 8 ? 8 : 1000;
const CLAMP = BITS === 8 ? 120 : 15000;   // ±15 · SCALE, inside the int range
const BYTES = BITS / 8;
const src = process.argv[2] || "/home/guillaume/p/IDR_ES/ZENODO/MAP/HUMAN_ES.txt";
const appDir = dirname(fileURLToPath(import.meta.url));
const outDir = join(appDir, ".cache");

const lines = readFileSync(src, "utf8").split("\n");
const header = lines[0].split("\t");
const features = header.slice(4);           // GID, IDRID, NAME, GWEIGHT, then features
const p = features.length;

// data rows start at line index 2 (skip header and the EWEIGHT row)
const ids = [];
const rows = [];
for (let li = 2; li < lines.length; li++) {
  if (!lines[li]) continue;                 // trailing blank line
  const f = lines[li].split("\t");
  ids.push(f[1]);                           // IDRID identifies the row
  rows.push(f);
}
const n = ids.length;

// column-major int buffer: value(feature j, row i) at index j*n + i
const buf = BYTES === 1 ? new Int8Array(p * n) : new Int16Array(p * n);
for (let i = 0; i < n; i++) {
  const f = rows[i];
  for (let j = 0; j < p; j++) {
    const raw = f[4 + j];
    let q = raw === "NA" || raw === undefined ? 0 : Math.round(parseFloat(raw) * SCALE);
    if (q > CLAMP) q = CLAMP; else if (q < -CLAMP) q = -CLAMP;
    buf[j * n + i] = q;
  }
}

// authoritative human-readable names + descriptions, from the paper's SI
const labelMap = JSON.parse(readFileSync(join(appDir, "feature-labels.json"), "utf8"));
const resolved = features.map((f) => labelMap[f] ?? { name: f, desc: "", source: "raw" });
const labels = resolved.map((r) => r.name);
const descs = resolved.map((r) => r.desc);

mkdirSync(outDir, { recursive: true });
const gz = gzipSync(Buffer.from(buf.buffer), { level: 9 });
writeFileSync(join(outDir, "data.bin.gz"), gz);
const metaJson = JSON.stringify({ n, p, bytes: BYTES, scale: SCALE, features, labels, descs, ids }, null, 2);
writeFileSync(join(outDir, "meta.json"), metaJson);

const mb = (b) => (b / 1024 / 1024).toFixed(2);
const sources = resolved.reduce((a, r) => ((a[r.source] = (a[r.source] || 0) + 1), a), {});
console.log(`${n} IDRs × ${p} features  (int${BITS} ×${SCALE})`);
console.log(`data.bin.gz  ${mb(gz.length)} MB  (raw int${BITS} ${mb(buf.byteLength)} MB)`);
console.log(`meta.json    ${mb(Buffer.byteLength(metaJson))} MB`);
console.log(`labels by source:`, sources);
