import { Sky } from "./sky.js";
import { Resolver, makeExecutors, fmtLocal } from "./tools.js";
import { NeedleModel, dateFact } from "./needle.js";
import { Recorder } from "./voice.js";
import { renderSpans, renderCalls, renderSpeed, renderWhy, renderHistory, showLoad, ToolMap } from "./inspector.js";

// Weights are pinned to a Hugging Face revision so the demo can't drift under us.
const NEEDLE3 = "https://huggingface.co/Cactus-Compute/needle3/resolve/c7c415a3d1b3d929014bc6e866d51ebb971f7089/needle3.cact";
const WHISTLE = "https://huggingface.co/Cactus-Compute/whistle/resolve/b358ddadd89b7a713b5aa131f23032d3cca1b251/whistle.cact";
const TUNED = "https://huggingface.co/mvandewettering/needle3-planetarium/resolve/8fc6bd97af258907a65cebecf9f95b2211b4d901/needle3-planetarium.cact";

const EXAMPLES = [
  "show me Saturn", "view the sky from Tokyo", "speed up time to an hour per second", "draw Mars's path over 120 days",
  "turn on star names", "which planets are up right now", "when does the moon rise", "show the sky on July 4 2027 at 9pm",
  "telescope view of the Orion Nebula", "tell me about Betelgeuse", "face east", "hide the constellation lines",
  "show me Jupiter from Sydney", "back to the present", "what's the weather tomorrow",
];

// Calls that change where/when we are run before calls that aim the view.
const ORDER = { set_location: 0, set_date_time: 1, reset_to_now: 1, set_time_speed: 2 };

const $ = (id) => document.getElementById(id);
const state = { mode: "base", models: {}, history: [], last: null, busy: false, toolMap: null };
window.planetarium = state;     // handy from the devtools console

function toast(msg, bad = false) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "show" + (bad ? " bad" : "");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.className = ""; }, 4200);
}

function progressHtml(label, p) {
  if (!p) return `${label}…`;
  const mb = (x) => (x / 1048576).toFixed(1);
  if (p.cached) return `${label}: from cache`;
  const pct = p.total ? (100 * p.loaded / p.total) : 0;
  return `${label}: ${mb(p.loaded)}${p.total ? ` / ${mb(p.total)}` : ""} MB<div class="bar"><i style="width:${pct}%"></i></div>`;
}

/** A daytime sky has no stars: open on tonight's darkness instead, and say so. */
function startAtNight(sky) {
  const A = globalThis.Astronomy;
  const obs = sky.observer();
  const now = A.MakeTime(new Date());
  const sun = A.Equator("Sun", now, obs, true, true);
  if (A.Horizon(now, obs, sun.ra, sun.dec).altitude < -12) return;
  const dark = A.SearchAltitude("Sun", obs, -1, now, 1, -15);
  if (!dark) return;
  sky.setTime(new Date(dark.date.getTime() + 45 * 60e3));
  setTimeout(() => toast(`It's daytime in ${sky.place.name}, so the sky starts tonight. Say “back to the present” for now.`), 600);
}

async function main() {
  const [catalog, places, tools] = await Promise.all(
    ["data/sky.json", "data/places.json", "data/tools.json"].map((u) => fetch(u).then((r) => r.json())));

  const sky = new Sky($("sky"), catalog);
  const resolver = new Resolver(sky, catalog, places);
  const ui = {
    showCard({ title, facts, clickable }) {
      $("card-title").textContent = title;
      $("card-facts").innerHTML = "";
      for (const [k, v] of facts) {
        const dt = document.createElement("dt"), dd = document.createElement("dd");
        dt.textContent = k; dd.textContent = v;
        if (clickable && k) {
          for (const el of [dt, dd]) { el.className = "click"; el.onclick = () => submit(`show me ${k.replace(/ \(.*\)$/, "")}`); }
        }
        $("card-facts").append(dt, dd);
      }
      $("card").hidden = false;
    },
  };
  $("card-close").onclick = () => { $("card").hidden = true; };
  const exec = makeExecutors(sky, resolver, ui);
  state.sky = sky;
  startAtNight(sky);

  // HUD
  const paintHud = () => {
    $("hud-place").textContent = sky.place.name;
    $("hud-time").textContent = fmtLocal(sky.time, sky.place.tz) + " local";
    const r = sky.rate, a = Math.abs(r);
    $("hud-rate").textContent = r === 1 ? "real time" : r === 0 ? "stopped"
      : `${r < 0 ? "−" : "+"}${a >= 86400 ? `${a / 86400} day` : a >= 3600 ? `${a / 3600} hr` : a >= 60 ? `${a / 60} min` : `${a} s`}/s`;
    const pressed = { rewind: r < 0, stop: r === 0, start: r === 1, ffwd: r > 1, now: false };
    document.querySelectorAll("#timebar button").forEach((b) => b.setAttribute("aria-pressed", String(pressed[b.dataset.t])));
  };
  setInterval(paintHud, 250);

  // Transport buttons drive the clock directly; repeated rewind / fast-forward
  // presses step up through the speeds.
  const SPEED_STEPS = [60, 600, 3600, 86400];
  const nextSpeed = (dir) => {
    const cur = sky.rate * dir;                            // current speed in the pressed direction
    const step = SPEED_STEPS.find((s) => s > cur) ?? SPEED_STEPS[SPEED_STEPS.length - 1];
    return step * dir;
  };
  const transport = {
    rewind: () => sky.setRate(nextSpeed(-1)),
    stop: () => sky.setRate(0),
    start: () => sky.setRate(1),
    ffwd: () => sky.setRate(nextSpeed(+1)),
    now: () => { sky.setTime(new Date()); sky.setRate(1); },
  };
  document.querySelectorAll("#timebar button").forEach((b) => {
    b.onclick = () => { transport[b.dataset.t](); paintHud(); };
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, button") || e.ctrlKey || e.metaKey || e.altKey) return;
    const key = { " ": sky.rate === 0 ? "start" : "stop", ArrowLeft: "rewind", ArrowRight: "ffwd", n: "now", N: "now" }[e.key];
    if (!key) return;
    e.preventDefault();
    transport[key]();
    paintHud();
  });

  // chips
  for (const ex of EXAMPLES) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = ex;
    b.onclick = () => submit(ex);
    $("chips").append(b);
  }

  // ---- models ----
  const system = `${dateFact()}; app: planetarium`;
  const status = { base: "Needle 3 base: waiting", tuned: "", whistle: "" };
  const paint = () => showLoad([status.base, status.tuned, status.whistle].filter(Boolean).join("<br>"));

  const base = new NeedleModel("Base", [NEEDLE3], {
    onProgress: (p) => { if (p.url === NEEDLE3) { status.base = progressHtml("Needle 3 (35 MB)", p); paint(); } },
  });
  state.models.base = base;
  await base.load();
  status.base = "Needle 3: indexing the 15 planetarium tools…"; paint();
  const t0 = performance.now();
  const prefix = await base.init(system, tools);
  status.base = `Needle 3 ready — ${prefix} prompt tokens indexed in ${((performance.now() - t0) / 1000).toFixed(1)} s`; paint();

  const vectors = await base.embed(tools.map((t) => t.description));
  state.toolMap = new ToolMap($("toolmap"), tools, vectors);

  $("query").disabled = false; $("go").disabled = false;
  $("query").placeholder = "Ask the sky… e.g. “show me Saturn from Tokyo”";
  $("query").focus();

  $("command").onsubmit = (e) => { e.preventDefault(); submit($("query").value); };
  $("why").onclick = explain;

  async function submit(text) {
    text = text.trim();
    if (!text || state.busy) return;
    $("query").value = text;
    state.busy = true; $("go").disabled = true; $("why").disabled = true;
    try { await run(text); } catch (e) { console.error(e); toast(String(e.message || e), true); }
    state.busy = false; $("go").disabled = false; $("why").disabled = false;
  }
  state.submit = submit;

  function sig(result) {
    return JSON.stringify((result.function_calls || []).map((c) => [c.name, Object.entries(c.arguments || {}).map(([k, v]) => [k, String(v).toLowerCase()]).sort()]));
  }

  async function run(text) {
    const which = state.mode === "both" ? ["base", "tuned"] : [state.mode];
    const runs = [];
    for (const key of which) {
      const m = state.models[key];
      const { result, ms } = await m.ask(text);
      runs.push({ key, label: key === "base" ? "Base Needle 3" : "Fine-tuned", result, ms });
    }
    const primary = runs[runs.length - 1];
    const calls = [...(primary.result.function_calls || [])].sort((a, b) => (ORDER[a.name] ?? 5) - (ORDER[b.name] ?? 5));
    primary.outcomes = calls.map((c) => {
      try { return exec[c.name] ? exec[c.name](c.arguments || {}) : { ok: false, summary: `Unknown tool ${c.name}` }; }
      catch (e) { console.error(e); return { ok: false, summary: `${c.name} failed: ${e.message}` }; }
    });
    primary.executed = true;

    $("toolmap-caption").textContent = primary.key === "tuned"
      ? "Each star is a tool; your command lands among the nearest. The fine-tuned build has no retrieval head, so it chose from all 15 tools — the lines show the 5 the base model would have offered."
      : "Each star is a tool. Your command lands among them; lines mark the five closest — the only tools the model is allowed to choose from.";
    renderSpans(text, primary.result);
    renderCalls(runs);
    renderSpeed(primary);
    const msg = primary.outcomes.map((o) => o.summary).join(" ");
    if (msg) toast(msg, primary.outcomes.some((o) => !o.ok));
    else if ((primary.result.suppressed_calls || []).length) toast("Not sure enough to act — see the inspector.", true);
    else toast("That's not something the planetarium can do.", true);

    state.last = { text, run: primary, sig: sig(primary.result) };
    state.history.push({ query: text, ok: primary.outcomes.length > 0 && primary.outcomes.every((o) => o.ok),
      calls: (primary.result.function_calls || []).map((c) => c.name).join(", ") });
    renderHistory(state.history, submit);

    const [qv] = await base.embed([text]);
    state.toolMap.setQuery(qv, (primary.result.function_calls || primary.result.suppressed_calls || []).map((c) => c.name));
  }

  async function explain() {
    const last = state.last;
    if (!last || state.busy) return;
    state.busy = true; $("why").disabled = true; $("go").disabled = true;
    const model = state.models[last.run.key];
    const words = last.text.split(/\s+/).filter(Boolean);
    const effects = [];
    const conf0 = last.run.result.confidence;
    for (let i = 0; i < words.length; i++) {
      $("why-out").textContent = `Re-running without “${words[i]}” (${i + 1}/${words.length})…`;
      const { result } = await model.ask(words.filter((_, j) => j !== i).join(" "));
      const flipped = sig(result) !== last.sig;
      const dc = conf0 != null && result.confidence != null ? Math.abs(result.confidence - conf0) : 0;
      const calls = (result.function_calls || []).map((c) => `${c.name}(${Object.values(c.arguments || {}).join(", ")})`).join("; ") || "nothing";
      effects.push({ flipped, score: flipped ? 1 : dc, note: flipped ? `without it: ${calls}` : `confidence ${conf0 != null ? `${Math.round(conf0 * 100)}% → ${Math.round((result.confidence ?? 0) * 100)}%` : "n/a"}` });
    }
    renderWhy(words, effects);
    state.busy = false; $("why").disabled = false; $("go").disabled = false;
  }

  // model switch
  document.querySelectorAll("#model-switch button").forEach((b) => {
    b.onclick = () => {
      if (b.disabled) return;
      state.mode = b.dataset.model;
      document.querySelectorAll("#model-switch button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    };
  });

  const q = new URLSearchParams(location.search).get("q");
  if (q) submit(q);

  // ---- background: speech model, then the fine-tuned model ----
  (async () => {
    try {
      await base.call("load", { urls: [WHISTLE] });
      status.whistle = "Whistle speech model ready — click the mic to talk"; paint();
      setupMic(base);
    } catch (e) { status.whistle = `Speech unavailable: ${e.message}`; paint(); }
  })();
  base.worker.addEventListener("message", ({ data }) => {
    if (data.op === "progress" && data.url === WHISTLE) { status.whistle = progressHtml("Whistle speech (17 MB)", data); paint(); }
  });

  const tunedBtns = document.querySelectorAll('#model-switch [data-model="tuned"], #model-switch [data-model="both"]');
  tunedBtns.forEach((b) => { b.disabled = true; });
  (async () => {
    const head = await fetch(TUNED, { method: "HEAD" }).catch(() => null);
    if (!head || !head.ok) { status.tuned = "Fine-tuned model unavailable"; paint(); return; }
    const tuned = new NeedleModel("Tuned", [TUNED], {
      onProgress: (p) => { status.tuned = progressHtml("Fine-tuned Needle (63 MB)", p); paint(); },
    });
    await tuned.load();
    status.tuned = "Fine-tuned: indexing tools…"; paint();
    await tuned.init(system, tools);
    state.models.tuned = tuned;
    status.tuned = "Fine-tuned model ready"; paint();
    tunedBtns.forEach((b) => { b.disabled = false; });
    document.querySelector('#model-switch [data-model="tuned"]').click();
  })().catch((e) => { status.tuned = `Fine-tuned model failed: ${e.message}`; paint(); });

  function setupMic(model) {
    const mic = $("mic");
    mic.disabled = false;
    let rec = null;
    const stop = async () => {
      if (!rec) return;
      const r = rec; rec = null;
      mic.classList.remove("rec"); mic.classList.add("busy");
      const pcm = await r.stop();
      if (!pcm) { mic.classList.remove("busy"); return; }
      $("query").value = "…listening";
      try {
        const { result, ms } = await model.transcribe(pcm);
        const text = (result.text || "").trim();
        mic.classList.remove("busy");
        if (!text) { $("query").value = ""; toast("Didn't catch that.", true); return; }
        toast(`Heard “${text}” (${ms.toFixed(0)} ms on-device)`);
        submit(text);
      } catch (e) { mic.classList.remove("busy"); toast(`Speech failed: ${e.message}`, true); }
    };
    mic.onclick = async () => {
      if (rec) return stop();
      try {
        rec = new Recorder();
        rec.onLimit = stop;
        await rec.start();
        mic.classList.add("rec");
      } catch (e) { rec = null; toast(`Microphone unavailable: ${e.message}`, true); }
    };
  }
}

main().catch((e) => { console.error(e); showLoad(`<span style="color:var(--bad)">Failed to start: ${e.message}</span>`); });
