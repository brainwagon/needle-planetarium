// Spike: run the planetarium tool set through the Needle 3 WASM engine in Node
// and score it against cases.json.  Usage: node spike/run.cjs [--verbose] [--system "..."]
const fs = require("node:fs");
const path = require("node:path");

const { VENDOR, dateFact, loadEngine } = require("./engine.cjs");

const args = process.argv.slice(2);
const verbose = args.includes("--verbose");
const sysIdx = args.indexOf("--system");
const SYSTEM_BASE = sysIdx >= 0 ? args[sysIdx + 1]
  : "app: planetarium";

function argMatches(want, got) {
  if (got === undefined) return false;
  if (typeof want === "string") {
    const g = String(got).toLowerCase();
    return want.toLowerCase().split("|").some((w) => g.includes(w) || w.includes(g) && g.length > 2);
  }
  if (typeof want === "number") return Number(got) === want;
  return got === want;
}

function callMatches(want, got) {
  if (!got || got.name !== want.name) return false;
  const a = got.arguments || {};
  return Object.entries(want.args).every(([k, v]) => argMatches(v, a[k]));
}

(async () => {
  const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
  const tools = fs.readFileSync(path.join(__dirname, opt("--tools", "tools.json")), "utf8");
  const cases = JSON.parse(fs.readFileSync(path.join(__dirname, opt("--cases", "cases.json")), "utf8"));

  let t0 = performance.now();
  const eng = await loadEngine();
  eng.load(opt("--model", path.join(VENDOR, "needle3.cact")));
  const prefix = eng.init(`${dateFact()}; ${SYSTEM_BASE}`, JSON.stringify(JSON.parse(tools)));
  console.log(`engine + weights ready in ${(performance.now() - t0).toFixed(0)} ms; static prefix ${prefix} tokens`);

  // --oracle: per case, offer the wanted tools plus the nearest distractors (by needle_embed)
  // up to 5, so retrieval can't hide the right answer.  Measures the decoder alone.
  const oracle = args.includes("--oracle");
  const toolList = JSON.parse(tools);
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  // embeddings come from the base model: local fine-tunes ship without the embedding readout
  const embedder = oracle ? await loadEngine() : null;
  if (embedder) embedder.load(path.join(VENDOR, "needle3.cact"));
  const toolVecs = oracle ? toolList.map((t) => embedder.embed(t.description)) : null;
  const subsetFor = (c) => {
    const names = new Set(c.want.map((w) => w.name));
    const qv = embedder.embed(c.q);
    const ranked = toolList.map((t, j) => [t, dot(qv, toolVecs[j])]).sort((a, b) => b[1] - a[1]);
    for (const [t] of ranked) { if (names.size >= 5) break; names.add(t.name); }
    return toolList.filter((t) => names.has(t.name));
  };

  let pass = 0, totalMs = 0;
  const rows = [];
  for (const c of cases) {
    if (oracle) eng.init(`${dateFact()}; ${SYSTEM_BASE}`, JSON.stringify(subsetFor(c)));
    eng.reset();
    t0 = performance.now();
    const r = eng.complete(c.q);
    const ms = performance.now() - t0;
    totalMs += ms;
    const calls = r.function_calls || [];
    // order-insensitive, per the Needle test-suite convention; "alt" lists other acceptable answers
    const okFor = (want) => calls.length === want.length && want.every((w) => calls.some((k) => callMatches(w, k)));
    const ok = [c.want, ...(c.alt || [])].some(okFor);
    if (ok) pass++;
    rows.push({ ok, ms, q: c.q, calls, conf: r.confidence, r });
    const callStr = calls.map((k) => `${k.name}(${JSON.stringify(k.arguments)})`).join(" ; ") || "(none)";
    const held = (r.suppressed_calls || []).length ? `  [suppressed: ${JSON.stringify(r.suppressed_calls)}]` : "";
    console.log(`${ok ? "PASS" : "FAIL"} ${ms.toFixed(0).padStart(5)}ms conf=${r.confidence?.toFixed?.(2) ?? r.confidence}  ${c.q}\n      -> ${callStr}${held}`);
    if (verbose) console.log("      raw:", JSON.stringify(r));
  }
  console.log(`\n${pass}/${cases.length} passed; mean latency ${(totalMs / cases.length).toFixed(0)} ms`);
  if (verbose) console.log("embedding dim:", eng.embed("show me Saturn").length);
})().catch((e) => { console.error(e); process.exit(1); });
