// Module Web Worker hosting the fine-tuned LFM2.5-350M on WebGPU via Transformers.js.
// Same message protocol as needle-worker.js, and results shaped like Needle's
// ({function_calls, prefill_tps, decode_tps, ...}) so the inspector can show either.
//
// Messages in:  {id, op, ...args}      Messages out: {id, ok, value|error} or {op:"progress", ...}
//   load      {repo, revision, system, tools}   fetch weights (Cache API) and prefill the prompt cache
//   complete  {text}                             returns {result, ms}
//   reset     {}                                 no-op: every command starts from the cached prompt

import * as tf from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js";
import { createRunner, parseLfm } from "./lfm.js";

let runner = null;

function post(msg) { self.postMessage(msg); }

const ops = {
  async load({ repo, revision, system, tools }) {
    const progress_callback = (p) => {
      // only the weights are worth a progress bar; the configs are a few KB
      if (p.status === "progress" && /\.onnx_data$/.test(p.file)) post({ op: "progress", url: p.file, loaded: p.loaded, total: p.total });
    };
    const tok = await tf.AutoTokenizer.from_pretrained(repo, { revision });
    const model = await tf.AutoModelForCausalLM.from_pretrained(repo, { revision, dtype: "q4", device: "webgpu", progress_callback });
    runner = await createRunner({ tf, tok, model, system, tools: tools.map((f) => ({ type: "function", function: f })) });
    await runner.complete("show me Saturn");   // the first decode compiles shaders; pay for it here, not on the user's command
    return { prefixTokens: runner.prefixTokens, prefixMs: runner.prefixMs };
  },

  async complete({ text }) {
    const r = await runner.complete(text);
    const decodeMs = r.ms - r.firstTokenMs;
    const result = {
      function_calls: parseLfm(r.text),
      raw: r.text,
      confidence: null,                    // nothing calibrated to report
      prefill_tps: r.firstTokenMs ? r.promptTokens / (r.firstTokenMs / 1000) : null,
      decode_tps: decodeMs > 0 && r.outTokens > 1 ? (r.outTokens - 1) / (decodeMs / 1000) : null,
    };
    return { result, ms: r.ms };
  },

  reset() {},
};

self.onmessage = async ({ data }) => {
  const { id, op, ...args } = data;
  try {
    post({ id, ok: true, value: await ops[op](args) });
  } catch (e) {
    post({ id, ok: false, error: String(e && e.message || e) });
  }
};
