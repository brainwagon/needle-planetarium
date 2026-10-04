// Main-thread handle on a Needle engine running in its own worker.

export class NeedleModel {
  constructor(label, urls, { onProgress } = {}) {
    this.label = label;
    this.urls = urls;
    this.worker = new Worker(new URL("./needle-worker.js", import.meta.url));
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

  load() { return this.call("load", { urls: this.urls }); }
  init(system, tools) { return this.call("init", { system, tools: JSON.stringify(tools) }); }
  reset() { return this.call("reset"); }
  embed(texts) { return this.call("embed", { texts }); }

  /** One stateless turn: reset history, then answer. */
  async ask(text) {
    await this.reset();
    return this.call("complete", { text });
  }

  async listen(pcm) {
    await this.reset();
    return this.call("listen", { pcm });
  }

  transcribe(pcm) { return this.call("transcribe", { pcm }); }
}

export function dateFact(d = new Date()) {
  const p = (n) => String(n).padStart(2, "0");
  const day = d.toLocaleDateString("en-US", { weekday: "short" });
  return `date: ${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${day} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
