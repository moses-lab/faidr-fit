// Build step (not a bundler in the webpack sense): concatenate the readable
// sources plus the data blob into one self-contained faidr.html that runs from
// file:// with no network. Run after prep-data.js:
//
//   node app/bundle.js   ->   app/dist/faidr.html
//
// Strategy: read index.html, inline the modules and the solver as one
// <script type="module">, and embed meta.json + data.bin.gz (base64) so the
// page decodes them via loadFromEmbedded instead of fetch. Everything stays
// legible in the sources; this file is the only place that stitches them together.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const appDir = dirname(fileURLToPath(import.meta.url));
const coreDir = join(appDir, "..", "core", "src");
const read = (...p) => readFileSync(join(...p), "utf8");

// strip local `import ... from "./x.js"` / "../core/..." lines: we concatenate
// the modules directly, so cross-module imports are resolved by placement order.
const stripImports = (src) => src.replace(/^\s*import\s+[^;]*?from\s*["'][^"']+["'];?\s*$/gm, "");

// export-order matters: define solver, then chart, then proteome, then ui
const modules = [
  // core modules
  ["fitLassoLogistic.js", read(coreDir, "fitLassoLogistic.js")],
  ["fitLogistic.js", read(coreDir, "fitLogistic.js")],
  ["predictLogistic.js", read(coreDir, "predictLogistic.js")],
  ["math/irlsLogisticLasso.js", read(coreDir, "math/irlsLogisticLasso.js")],
  ["math/lambdaPath.js", read(coreDir, "math/lambdaPath.js")],
  ["math/logistic.js", read(coreDir, "math/logistic.js")],
  ["math/matrixInverse.js", read(coreDir, "math/matrixInverse.js")],
  ["math/softThreshold.js", read(coreDir, "math/softThreshold.js")],
  ["math/standardize.js", read(coreDir, "math/standardize.js")],
  ["math/weightedLassoCD.js", read(coreDir, "math/weightedLassoCD.js")],
  // app modules
  ["chart.js", read(appDir, "chart.js")],
  ["proteome.js", read(appDir, "proteome.js")],
  ["ui.js", read(appDir, "ui.js")],
].map(([name, src]) => `// ---- ${name} ----\n${stripImports(src).replace(/^export\s+/gm, "")}`).join("\n\n");

const cacheDir = join(appDir, ".cache");
const meta = read(cacheDir, "meta.json");
const gzB64 = readFileSync(join(cacheDir, "data.bin.gz")).toString("base64");
// embed the GO benchmark sets (if generated) so the dropdown works offline
let goJson = "", goGzB64 = "";
try {
  goJson = read(cacheDir, "go.json");
  goGzB64 = readFileSync(join(cacheDir, "go.bin.gz")).toString("base64");
} catch { /* GO sets optional */ }

const bootstrap = `
const META = ${JSON.stringify(meta)};
const GZ_B64 = "${gzB64}";
const GO_JSON = ${JSON.stringify(goJson)};
const GO_GZ_B64 = "${goGzB64}";
(async () => {
  const status = document.getElementById("status");
  status.textContent = "Decoding proteome…";
  try {
    const proteome = await loadFromEmbedded(META, GZ_B64);
    const go = GO_JSON ? await loadGoFromEmbedded(GO_JSON, GO_GZ_B64) : null;
    startApp(proteome, go);
  } catch (e) { status.textContent = "Failed to decode embedded data"; console.error(e); }
})();`;

// take the dev page, drop its module <script>, inject the merged one
let html = read(appDir, "index.html");
html = html.replace(/<script type="module">[\s\S]*?<\/script>/, `<script type="module">\n${modules}\n${bootstrap}\n</script>`);

const outDir = join(appDir, "dist");
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, "faidr.html"), html);
console.log(`dist/faidr.html  ${(Buffer.byteLength(html) / 1024 / 1024).toFixed(2)} MB`);
