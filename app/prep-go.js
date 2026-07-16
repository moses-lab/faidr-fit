// Build the GO-term membership blob the app's benchmark-set dropdown uses. Run
// once (like prep-data.js), pointing at the paper's high-AUC/PPV GO label files:
//
//   node app/prep-go.js [path/to/HIGH_AUC_PPV_FILTERED_FUNCTIONS]
//
// Output (git-ignored):
//   go.bin.gz  gzipped Uint16 array of positive IDR row-indices (into meta.ids),
//              all terms concatenated (~70 KB gzipped for the 148 terms).
//   go.json    per-term { go, label, proteins, idrs, start, len } — `start`/`len`
//              slice the inflated Uint16 array for that term's positives.
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const outDir = join(dirname(fileURLToPath(import.meta.url)), ".cache");
const goDir = process.argv[2] ||
  "/home/guillaume/p/IDR_ES/ZENODO/HIGH_AUC_PPV_FILTERED_FUNCTIONS";

const meta = JSON.parse(readFileSync(join(outDir, "meta.json"), "utf8"));
const idIndex = new Map(meta.ids.map((v, i) => [v, i]));   // idr_name -> matrix row

const parsed = [];
for (const file of readdirSync(goDir).filter((f) => f.endsWith(".txt"))) {
  const lines = readFileSync(join(goDir, file), "utf8").split("\n");
  const head = lines[0].split("\t");
  const idC = head.indexOf("idr_name"), nameC = head.indexOf("NAME");
  const labC = head.findIndex((h) => /^GO\d/.test(h));
  const m = head[labC].match(/^GO(\d+)_(.+)$/);        // GO0005730_nucleolus
  const rows = [], proteins = new Set();
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue;
    const c = lines[i].split("\t");
    if (c[labC] === "1") {
      const r = idIndex.get(c[idC]);
      if (r !== undefined) { rows.push(r); proteins.add(c[nameC]); }
    }
  }
  rows.sort((a, b) => a - b);
  parsed.push({ go: `GO:${m[1]}`, label: m[2].replace(/_/g, " "), proteins: proteins.size, rows });
}
parsed.sort((a, b) => a.label.localeCompare(b.label));

const terms = [], flat = [];
for (const t of parsed) {
  terms.push({ go: t.go, label: t.label, proteins: t.proteins, idrs: t.rows.length, start: flat.length, len: t.rows.length });
  for (const r of t.rows) flat.push(r);
}
const gz = gzipSync(Buffer.from(Uint16Array.from(flat).buffer), { level: 9 });
writeFileSync(join(outDir, "go.bin.gz"), gz);
const json = JSON.stringify({ terms });
writeFileSync(join(outDir, "go.json"), json);

const kb = (b) => (b / 1024).toFixed(0);
console.log(`${terms.length} GO terms, ${flat.length} positive (term,IDR) pairs`);
console.log(`go.bin.gz ${kb(gz.length)} KB · go.json ${kb(Buffer.byteLength(json))} KB`);
