// "Inside the Needle": renders what the model heard, what it called, how sure
// it was, and where the command landed among the tools.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const SPAN_COLORS = ["#ffd27a", "#7fb2ff", "#7ddca0", "#ff9ec7", "#c3a6ff", "#ffb36b"];

/** Pull "'span' -> field 'value'" triples out of Needle's reasoning line. */
export function reasoningSpans(reasoning) {
  const out = [];
  const re = /'([^']+)'\s*->\s*([a-z_]+)(?:\s+'([^']*)')?/gi;
  let m;
  while ((m = re.exec(reasoning || ""))) out.push({ text: m[1], field: m[2], value: m[3] });
  return out;
}

export function renderSpans(query, result) {
  const el = $("spans");
  el.classList.remove("muted");
  let spans = reasoningSpans(result?.reasoning);
  if (!spans.length) {                     // fall back to argument values that appear verbatim
    for (const c of result?.function_calls || [])
      for (const [k, v] of Object.entries(c.arguments || {}))
        if (typeof v === "string") spans.push({ text: v, field: k });
  }
  const marks = [];
  const lower = query.toLowerCase();
  spans.forEach((s, i) => {
    // the quoted source span is sometimes the whole request; the value it derived is tighter
    const vAt = s.value ? lower.indexOf(s.value.toLowerCase()) : -1;
    if (vAt >= 0 && s.value.length < s.text.length) s = { ...s, text: s.value };
    const at = lower.indexOf(s.text.toLowerCase());
    if (at >= 0 && !marks.some((m) => at < m.end && at + s.text.length > m.start))
      marks.push({ start: at, end: at + s.text.length, field: s.field, color: SPAN_COLORS[i % SPAN_COLORS.length] });
  });
  marks.sort((a, b) => a.start - b.start);
  let html = "", pos = 0;
  for (const m of marks) {
    html += esc(query.slice(pos, m.start));
    html += `<span class="span" style="background:${m.color}33;box-shadow:inset 0 -2px 0 ${m.color}">${esc(query.slice(m.start, m.end))}<small style="color:${m.color}">${esc(m.field)}</small></span>`;
    pos = m.end;
  }
  html += esc(query.slice(pos));
  el.innerHTML = html || "&nbsp;";
}

function callHtml(c) {
  const args = Object.entries(c.arguments || {}).map(([k, v]) =>
    `<span class="k">${esc(k)}</span>: ${typeof v === "string" ? `<span class="s">"${esc(v)}"</span>` : esc(JSON.stringify(v))}`).join(", ");
  return `<span class="fn">${esc(c.name)}</span>(${args})`;
}

function confHtml(conf) {
  if (conf == null) return `<div class="conf">confidence <span class="muted">— not calibrated for local fine-tunes</span></div>`;
  const pct = Math.round(conf * 100);
  const col = conf >= 0.5 ? "var(--ok)" : conf >= 0.1 ? "var(--warn)" : "var(--bad)";
  return `<div class="conf">confidence <div class="meter" title="Calls under 10% are withheld by the engine"><i style="width:${pct}%;background:${col}"></i><b></b></div> ${pct}%</div>`;
}

/** runs: [{label, result, ms, outcomes: [{summary, ok}], executed: bool}] */
export function renderCalls(runs) {
  const el = $("calls");
  const cards = runs.map((run) => {
    const r = run.result;
    const calls = r.function_calls || [];
    const held = r.suppressed_calls || [];
    let body = "";
    if (calls.length) body = `<pre>${calls.map(callHtml).join("\n")}</pre>`;
    else if (held.length) body = `<pre>${held.map(callHtml).join("\n")}</pre><div class="result bad">Withheld — too unsure to act.</div>`;
    else body = `<div class="none">No tool fits — the model declined.</div>`;
    const outcomes = (run.outcomes || []).map((o) => `<div class="result ${o.ok ? "" : "bad"}">${esc(o.summary)}</div>`).join("");
    return `<div class="call ${!calls.length && held.length ? "suppressed" : ""}">
      <div class="head"><span class="who">${esc(run.label)}${run.executed ? " · drove the sky" : ""}</span><span class="ms">${run.ms.toFixed(0)} ms</span></div>
      ${body}${outcomes}${confHtml(r.confidence)}</div>`;
  });
  el.className = runs.length > 1 ? "compare" : "";
  el.innerHTML = cards.join("");
}

export function renderSpeed(run, tokensOut) {
  const r = run.result;
  $("speed").classList.remove("muted");
  $("speed").innerHTML = `
    <div><b>${run.ms.toFixed(0)}<small> ms</small></b><span>command → call</span></div>
    <div><b>${r.prefill_tps ? r.prefill_tps.toFixed(0) : "—"}</b><span>prefill tok/s</span></div>
    <div><b>${r.decode_tps ? r.decode_tps.toFixed(0) : "—"}</b><span>decode tok/s</span></div>`;
}

// ---- tool space map ---------------------------------------------------------

function pca2(vectors) {
  const n = vectors.length, d = vectors[0].length;
  const mean = new Float64Array(d);
  for (const v of vectors) for (let j = 0; j < d; j++) mean[j] += v[j] / n;
  const X = vectors.map((v) => Float64Array.from(v, (x, j) => x - mean[j]));
  const comps = [];
  for (let c = 0; c < 2; c++) {
    let w = Float64Array.from({ length: d }, (_, j) => Math.sin(j * 12.9898 + c * 78.233));   // deterministic start
    for (let it = 0; it < 80; it++) {
      const s = X.map((x) => x.reduce((acc, xi, j) => acc + xi * w[j], 0));
      const nw = new Float64Array(d);
      X.forEach((x, i) => { for (let j = 0; j < d; j++) nw[j] += s[i] * x[j]; });
      for (const p of comps) { const dot = nw.reduce((a, x, j) => a + x * p[j], 0); for (let j = 0; j < d; j++) nw[j] -= dot * p[j]; }
      const norm = Math.hypot(...nw) || 1;
      w = nw.map((x) => x / norm);
    }
    comps.push(w);
  }
  return X.map((x) => comps.map((w) => x.reduce((a, xi, j) => a + xi * w[j], 0)));
}

const cos = (a, b) => { let s = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return s / Math.sqrt(na * nb); };

export class ToolMap {
  constructor(canvas, tools, vectors) {
    this.canvas = canvas;
    this.tools = tools;
    this.vectors = vectors;
    const pts = pca2(vectors);
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    this.pos = pts.map(([x, y]) => [(x - x0) / (x1 - x0 || 1), (y - y0) / (y1 - y0 || 1)]);
    // nudge apart stars whose labels would overlap; the layout stays roughly faithful
    for (let it = 0; it < 300; it++) {
      for (let i = 0; i < this.pos.length; i++) for (let j = i + 1; j < this.pos.length; j++) {
        const a = this.pos[i], b = this.pos[j];
        const dx = b[0] - a[0], dy = b[1] - a[1];
        if (Math.abs(dx) < 0.26 && Math.abs(dy) < 0.09) {
          const push = 0.004 * (dy >= 0 ? 1 : -1), side = 0.002 * (dx >= 0 ? 1 : -1);
          a[1] -= push; b[1] += push; a[0] -= side; b[0] += side;
        }
      }
      for (const p of this.pos) { p[0] = Math.max(0, Math.min(1, p[0])); p[1] = Math.max(0, Math.min(1, p[1])); }
    }
    this.query = null;
    this.t0 = performance.now();
    new ResizeObserver(() => this.draw()).observe(canvas);
    const loop = () => { this.draw(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  /** Place the command by similarity and remember its top-5 neighbours. */
  setQuery(vec, chosen) {
    const sims = this.vectors.map((v) => cos(vec, v));
    const order = sims.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]);
    const top = order.slice(0, 5).map(([, i]) => i);
    const w = sims.map((s) => Math.exp((s - order[0][0]) * 40));
    const W = w.reduce((a, b) => a + b, 0);
    const target = [0, 1].map((k) => this.pos.reduce((acc, p, i) => acc + p[k] * w[i], 0) / W);
    const from = this.query ? this.query.at : [0.5, 0.5];
    this.query = { from, target, at: from, top, chosen: new Set(chosen), t0: performance.now(), sims };
  }

  draw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const pad = 38;
    const P = ([x, y]) => [pad + x * (w - 2 * pad), pad * 0.7 + (1 - y) * (h - 1.4 * pad)];
    const now = performance.now();
    const q = this.query;
    if (q) {
      const k = Math.min(1, (now - q.t0) / 700), e = 1 - Math.pow(1 - k, 3);
      q.at = [q.from[0] + (q.target[0] - q.from[0]) * e, q.from[1] + (q.target[1] - q.from[1]) * e];
      const qp = P(q.at);
      for (const i of q.top) {
        const tp = P(this.pos[i]);
        const chosen = q.chosen.has(this.tools[i].name);
        ctx.strokeStyle = chosen ? "rgba(255,210,122,0.95)" : "rgba(127,178,255,0.35)";
        ctx.lineWidth = chosen ? 2 : 1;
        ctx.setLineDash(chosen ? [] : [3, 4]);
        ctx.beginPath(); ctx.moveTo(qp[0], qp[1]); ctx.lineTo(tp[0], tp[1]); ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center";
    this.pos.forEach((p, i) => {
      const [x, y] = P(p);
      const inTop = q && q.top.includes(i);
      const chosen = q && q.chosen.has(this.tools[i].name);
      const tw = 1 + 0.25 * Math.sin(now / 600 + i * 1.7);
      const r = (chosen ? 4.5 : inTop ? 3.2 : 2.2) * tw;
      ctx.fillStyle = chosen ? "#ffd27a" : inTop ? "#cfe0ff" : "rgba(160,175,210,0.55)";
      if (chosen) { ctx.shadowColor = "#ffd27a"; ctx.shadowBlur = 14; }
      ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = chosen ? "#ffe4a8" : inTop ? "#aac4ee" : "rgba(133,146,176,0.6)";
      ctx.fillText(this.tools[i].name, x, y + 14);
    });
    if (q) {
      const [x, y] = P(q.at);
      ctx.fillStyle = "#fff";
      ctx.shadowColor = "#fff"; ctx.shadowBlur = 10;
      ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 2 * Math.PI); ctx.fill();
      ctx.shadowBlur = 0;
    }
  }
}

// ---- word importance --------------------------------------------------------

export function renderWhy(words, effects) {
  const max = Math.max(0.05, ...effects.map((e) => e.score));
  $("why-out").classList.remove("muted");
  $("why-out").innerHTML = `<div class="why">${words.map((w, i) => {
    const e = effects[i];
    const hgt = 4 + 46 * (e.score / max);
    return `<span class="w ${e.flipped ? "flip" : ""}" title="${esc(e.note)}"><i style="height:${hgt}px"></i>${esc(w)}</span>`;
  }).join("")}</div>
  <div class="why-legend"><span style="color:var(--bad)">■</span> removing the word changes the call &nbsp; <span style="color:var(--accent)">■</span> only shifts confidence</div>`;
}

export function renderHistory(items, onPick) {
  const el = $("history");
  el.innerHTML = items.slice().reverse().map((it, i) =>
    `<li data-i="${items.length - 1 - i}"><span class="${it.ok ? "ok" : "no"}">${it.ok ? "✓" : "·"}</span> ${esc(it.query)} <span class="muted">→ ${esc(it.calls || "nothing")}</span></li>`).join("");
  el.querySelectorAll("li").forEach((li) => li.addEventListener("click", () => onPick(items[+li.dataset.i].query)));
}

export function showLoad(html) { $("load-status").innerHTML = html; }
