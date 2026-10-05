// Main-thread handles on models running in their own workers: Needle 3 engines
// (needle-worker.js) and the fine-tuned LFM2.5 (lfm-worker.js).

class WorkerModel {
  constructor(label, worker, { onProgress } = {}) {
    this.label = label;
    this.worker = worker;
    this.pending = new Map();
    this.nextId = 1;
    this.queue = Promise.resolve();     // the engine is single-threaded: serialize calls
    this.worker.onmessage = ({ data }) => {
      if (data.op === "progress") { onProgress?.(data); return; }
      const p = this.pending.get(data.id);
      this.pending.delete(data.id);
      data.ok ? p.resolve(data.value) : p.reject(new Error(data.error));
    };
  }

  call(op, args = {}, transfer) {
    const run = () => new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, op, ...args }, transfer || []);
    });
    const result = this.queue.then(run, run);
    this.queue = result.catch(() => {});
    return result;
  }

  /** One stateless turn: reset history, then answer. */
  async ask(text) {
    await this.reset();
    return this.call("complete", { text });
  }

  reset() { return this.call("reset"); }
}

export class NeedleModel extends WorkerModel {
  constructor(label, urls, opts) {
    super(label, new Worker(new URL("./needle-worker.js", import.meta.url)), opts);
    this.urls = urls;
  }

  load() { return this.call("load", { urls: this.urls }); }
  init(system, tools) { return this.call("init", { system, tools: JSON.stringify(tools) }); }
  embed(texts) { return this.call("embed", { texts }); }

  async listen(pcm) {
    await this.reset();
    return this.call("listen", { pcm });
  }

  transcribe(pcm) { return this.call("transcribe", { pcm }); }
}

export class LfmModel extends WorkerModel {
  constructor(label, opts) {
    super(label, new Worker(new URL("./lfm-worker.js", import.meta.url), { type: "module" }), opts);
  }

  /** Fetch the weights and prefill the system + tool prompt; returns {prefixTokens, prefixMs}. */
  load(repo, revision, system, tools) { return this.call("load", { repo, revision, system, tools }); }
}

/** WebGPU with a usable adapter: what the LFM worker needs. */
export async function hasWebGPU() {
  try { return !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch { return false; }
}

export function dateFact(d = new Date()) {
  const p = (n) => String(n).padStart(2, "0");
  const day = d.toLocaleDateString("en-US", { weekday: "short" });
  return `date: ${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${day} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
