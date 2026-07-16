// Loads the shipped proteome into the shape the solver wants:
//   X = { n, p, cols: Float64Array[p], features: string[p] }, plus ids[n].
//
// Two entry points share one decoder so dev (fetch loose files) and the bundled
// single-file build (embedded base64) run identical code from `decode` onward.

// Inflate a gzip byte array using the platform's native DecompressionStream
// (browser and Node 18+), so we need no compression library in the bundle.
export async function gunzip(bytes) {
  const ds = new DecompressionStream("gzip");
  const stream = new Response(bytes).body.pipeThrough(ds);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// Build X from the inflated column-major int buffer and the metadata. The
// quantization width is carried in meta.bytes (1 = int8, 2 = int16; default 2).
export function decode(intBuffer, meta) {
  const { n, p, bytes, scale, features, labels, descs, ids } = meta;
  const Q = bytes === 1 ? Int8Array : Int16Array;
  const q = new Q(intBuffer.buffer, intBuffer.byteOffset, p * n);
  const inv = 1 / scale;
  const cols = [];
  for (let j = 0; j < p; j++) {
    const col = new Float64Array(n), base = j * n;
    for (let i = 0; i < n; i++) col[i] = q[base + i] * inv;
    cols.push(col);
  }
  // labels/descs are the authoritative SI names aligned to features
  return { X: { n, p, cols, features }, ids, labels: labels || features, descs: descs || [] };
}

// Dev path: fetch meta.json + data.bin.gz next to the page, inflate, decode.
export async function loadFromFiles(base = ".cache") {
  const [meta, gz] = await Promise.all([
    fetch(`${base}/meta.json`).then((r) => r.json()),
    fetch(`${base}/data.bin.gz`).then((r) => r.arrayBuffer()),
  ]);
  return decode(await gunzip(new Uint8Array(gz)), meta);
}

// Bundled path: the build inlines meta as JSON and the gzip as base64.
export async function loadFromEmbedded(metaJson, gzBase64) {
  const bin = Uint8Array.from(atob(gzBase64), (c) => c.charCodeAt(0));
  return decode(await gunzip(bin), JSON.parse(metaJson));
}

// GO benchmark-set membership: { terms, members }. `members` is one flat Uint16
// array of positive IDR row-indices; each term slices it by its start/len.
function decodeGo(json, bytes) {
  // copy into an aligned buffer so the Uint16 view is valid regardless of offset
  const buf = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? bytes.buffer : bytes.slice().buffer;
  return { terms: JSON.parse(json).terms, members: new Uint16Array(buf) };
}
export async function loadGoFromFiles(base = ".cache") {
  const [json, gz] = await Promise.all([
    fetch(`${base}/go.json`).then((r) => r.text()),
    fetch(`${base}/go.bin.gz`).then((r) => r.arrayBuffer()),
  ]);
  return decodeGo(json, await gunzip(new Uint8Array(gz)));
}
export async function loadGoFromEmbedded(json, gzBase64) {
  const bin = Uint8Array.from(atob(gzBase64), (c) => c.charCodeAt(0));
  return decodeGo(json, await gunzip(bin));
}
