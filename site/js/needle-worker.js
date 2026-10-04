// Web Worker hosting one Needle 3 WASM engine instance.
// The engine holds one text model (and optionally Whistle for speech) per
// instance, so the page runs one worker per model it wants to compare.
//
// Messages in:  {id, op, ...args}      Messages out: {id, ok, value|error} or {op:"progress", ...}
//   load       {urls: [cactUrl, ...]}        fetch (Cache API) + needle_load each archive
//   init       {system, tools}               needle_init; returns static-prefix token count
//   complete   {text, maxNew}                 needle_complete on text; returns {result, ms}
//   listen     {pcm: Float32Array, maxNew}    speech -> Whistle transcript -> tool calls, one engine call
//   transcribe {pcm}                          speech -> text only
//   embed      {texts: [..]}                  needle_embed per text; returns Float32Array[]
//   reset      {}

importScripts("../engine/needle.js");   // defines createNeedle (Emscripten MODULARIZE)

const CACHE = "needle-weights-v1";
let M = null;
let outPtr = 0;
const OUT_CAP = 1 << 17;

function post(msg, transfer) { self.postMessage(msg, transfer || []); }

function cstr(s) {
  const bytes = new TextEncoder().encode(s);
  const p = M._malloc(bytes.length + 1);
  M.HEAPU8.set(bytes, p);
  M.HEAPU8[p + bytes.length] = 0;
  return p;
}

function lastError() { return M.UTF8ToString(M._needle_last_error()); }

async function fetchCached(url) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch { /* no Cache API (e.g. file://) */ }
  let resp = cache && await cache.match(url);
  if (resp) {
    post({ op: "progress", url, loaded: 1, total: 1, cached: true });
    return new Uint8Array(await resp.arrayBuffer());
  }
  resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} fetching ${url}`);
  const total = Number(resp.headers.get("content-length")) || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0, lastPost = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    const now = performance.now();
    if (now - lastPost > 100) { post({ op: "progress", url, loaded, total }); lastPost = now; }
  }
  const bytes = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { bytes.set(c, off); off += c.length; }
  post({ op: "progress", url, loaded, total: loaded });
  if (cache) {
    try { await cache.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } })); }
    catch { /* quota: fine, just re-download next visit */ }
  }
  return bytes;
}

function withPcm(pcm, fn) {
  const p = M._malloc(pcm.length * 4);
  new Float32Array(M.HEAPU8.buffer, p, pcm.length).set(pcm);
  try { return fn(p); } finally { M._free(p); }
}

function readOut(rc, what) {
  if (rc < 0) throw new Error(`${what}: ${lastError() || M.UTF8ToString(outPtr) || rc}`);
  return JSON.parse(M.UTF8ToString(outPtr));
}

const ops = {
  async load({ urls }) {
    if (!M) {
      // this Emscripten build ignores locateFile, so hand it the binary directly
      const wasm = await fetch(new URL("../engine/needle.wasm", self.location.href)).then((r) => r.arrayBuffer());
      M = await createNeedle({ wasmBinary: wasm });
      outPtr = M._malloc(OUT_CAP);
    }
    for (const url of urls) {
      const bytes = await fetchCached(url);
      const p = M._malloc(bytes.length);       // engine reads weights in place: never freed
      M.HEAPU8.set(bytes, p);
      const rc = M._needle_load(p, BigInt(bytes.length));
      if (rc < 0) throw new Error(`needle_load ${url}: ${lastError()}`);
    }
    return M._needle_models();
  },

  init({ system, tools }) {
    const s = cstr(system || ""), t = cstr(tools);
    try {
      const rc = M._needle_init(s, t, 0);
      if (rc < 0) throw new Error(`needle_init: ${lastError()}`);
      return rc;
    } finally { M._free(s); M._free(t); }
  },

  complete({ text, maxNew = 256 }) {
    const t = cstr(text);
    const t0 = performance.now();
    try {
      const result = readOut(M._needle_complete(t, 0, 0, maxNew, outPtr, OUT_CAP), "needle_complete");
      return { result, ms: performance.now() - t0 };
    } finally { M._free(t); }
  },

  listen({ pcm, maxNew = 256 }) {
    const t0 = performance.now();
    const result = withPcm(pcm, (p) =>
      readOut(M._needle_complete(0, p, pcm.length, maxNew, outPtr, OUT_CAP), "needle_complete(audio)"));
    return { result, ms: performance.now() - t0 };
  },

  transcribe({ pcm }) {
    const t0 = performance.now();
    const result = withPcm(pcm, (p) =>
      readOut(M._needle_transcribe(p, pcm.length, 0, 0, 0, outPtr, OUT_CAP), "needle_transcribe"));
    return { result, ms: performance.now() - t0 };
  },

  embed({ texts }) {
    return texts.map((text) => {
      const t = cstr(text);
      try {
        const dim = M._needle_embed(t, 0, 0, 0, 0);
        if (dim <= 0) throw new Error(`needle_embed: ${lastError()}`);
        const buf = M._malloc(dim * 4);
        M._needle_embed(t, 0, 0, buf, dim);
        const v = new Float32Array(M.HEAPU8.buffer, buf, dim).slice();
        M._free(buf);
        return v;
      } finally { M._free(t); }
    });
  },

  reset() { M._needle_reset(); },
};

self.onmessage = async ({ data }) => {
  const { id, op, ...args } = data;
  try {
    post({ id, ok: true, value: await ops[op](args) });
  } catch (e) {
    post({ id, ok: false, error: String(e && e.message || e) });
  }
};
