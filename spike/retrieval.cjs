// Spike: does needle_embed retrieval put the wanted tool in the top 5?
// Tries several ways of turning a tool schema into embedding text.
// Usage: node spike/retrieval.cjs [tools2.json] [cases2.json]
const fs = require("node:fs");
const path = require("node:path");
const { VENDOR, loadEngine } = require("./engine.cjs");

const [toolsFile = "tools2.json", casesFile = "cases2.json"] = process.argv.slice(2);
const tools = JSON.parse(fs.readFileSync(path.join(__dirname, toolsFile), "utf8"));
const cases = JSON.parse(fs.readFileSync(path.join(__dirname, casesFile), "utf8")).filter((c) => c.want.length);

const views = {
  description: (t) => t.description,
  name_desc: (t) => `${t.name.replace(/_/g, " ")}: ${t.description}`,
  json: (t) => JSON.stringify(t),
};

const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

(async () => {
  const eng = await loadEngine();
  eng.load(path.join(VENDOR, "needle3.cact"));
  const qv = cases.map((c) => eng.embed(c.q));
  for (const [view, fmt] of Object.entries(views)) {
    const tv = tools.map((t) => eng.embed(fmt(t)));
    let hits = 0, need = 0;
    const misses = [];
    cases.forEach((c, i) => {
      const ranked = tools.map((t, j) => [t.name, dot(qv[i], tv[j])]).sort((a, b) => b[1] - a[1]);
      const top5 = ranked.slice(0, 5).map((r) => r[0]);
      for (const w of c.want) {
        need++;
        if (top5.includes(w.name)) hits++;
        else misses.push(`${c.q}  [want ${w.name} at #${ranked.findIndex((r) => r[0] === w.name) + 1}; top: ${top5.slice(0, 3).join(", ")}]`);
      }
    });
    console.log(`\n${view}: recall@5 ${hits}/${need}`);
    misses.forEach((m) => console.log("   miss: " + m));
  }
})().catch((e) => { console.error(e); process.exit(1); });
