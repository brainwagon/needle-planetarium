// Thin Node wrapper over the Needle 3 WASM engine (same build the browser uses).
const fs = require("node:fs");
const path = require("node:path");

const VENDOR = path.join(__dirname, "..", "vendor", "needle");
const createNeedle = require(path.join(VENDOR, "needle.js"));
const { dateFact } = require("./common.mjs");

async function loadEngine() {
  const M = await createNeedle();
  const str = (s) => {
    if (s == null) return 0;
    const bytes = new TextEncoder().encode(s);
    const p = M._malloc(bytes.length + 1);
    M.HEAPU8.set(bytes, p);
    M.HEAPU8[p + bytes.length] = 0;
    return p;
  };
  const lastError = () => M.UTF8ToString(M._needle_last_error());

  const load = (file) => {
    const bytes = fs.readFileSync(file);
    const p = M._malloc(bytes.length);       // engine reads weights in place; never freed
    M.HEAPU8.set(bytes, p);
    const rc = M._needle_load(p, BigInt(bytes.length));
    if (rc < 0) throw new Error(`needle_load(${file}): ${lastError()}`);
  };

  const OUT_CAP = 1 << 16;
  const out = M._malloc(OUT_CAP);

  return {
    M, load, lastError,
    init(system, tools) {
      const s = str(system), t = str(tools);
      const rc = M._needle_init(s, t, 0);
      M._free(s); M._free(t);
      if (rc < 0) throw new Error(`needle_init: ${lastError()}`);
      return rc;
    },
    reset() { M._needle_reset(); },
    complete(text, maxNew = 256) {
      const t = str(text);
      const rc = M._needle_complete(t, 0, 0, maxNew, out, OUT_CAP);
      M._free(t);
      if (rc < 0) throw new Error(`needle_complete: ${lastError()}`);
      return JSON.parse(M.UTF8ToString(out));
    },
    embed(text) {
      const t = str(text);
      const dim = M._needle_embed(t, 0, 0, 0, 0);
      const buf = M._malloc(dim * 4);
      M._needle_embed(t, 0, 0, buf, dim);
      const v = Array.from(new Float32Array(M.HEAPU8.buffer, buf, dim));
      M._free(buf); M._free(t);
      return v;
    },
  };
}

module.exports = { VENDOR, dateFact, loadEngine };
