// LFM2.5-350M fine-tuned for the planetarium: the call parser and a runner that
// prefills the system + tool prompt once and reuses its cache, so a command only
// pays for its own few tokens.  Shared by the site's worker (lfm-worker.js), the
// Node harness (spike/run-tf.mjs) and the prototype (spike/webgpu/).
//
// Transformers.js is passed in (`tf`) rather than imported: the browser loads it
// from a CDN URL, Node from its package.

// The system prompt the model was fine-tuned with; the date line follows it.
export const LFM_SYSTEM = "You control a planetarium app. Call tools to carry out the user's request.";

// A bare value in a call: quoted string, number, boolean, or a bare word.
export function scalar(s) {
  s = s.trim();
  if (/^-?\d+(\.\d+)?$/.test(s)) return Number(s);
  if (/^(true|True)$/.test(s)) return true;
  if (/^(false|False)$/.test(s)) return false;
  const q = s.match(/^(["'])([\s\S]*)\1$/);
  return q ? q[2] : s;
}

/** <|tool_call_start|>[name(key="value", n=3), other()]<|tool_call_end|> -> [{name, arguments}] */
export function parseLfm(text) {
  const calls = [];
  for (const block of text.matchAll(/<\|tool_call_start\|>([\s\S]*?)<\|tool_call_end\|>/g))
    for (const m of block[1].matchAll(/(\w+)\(((?:"[^"]*"|'[^']*'|[^)])*)\)/g)) {
      const a = {};
      for (const kv of m[2].matchAll(/(\w+)\s*=\s*("[^"]*"|'[^']*'|[^,]+)/g)) a[kv[1]] = scalar(kv[2]);
      calls.push({ name: m[1], arguments: a });
    }
  return calls;
}

// Cache outputs are named present*; the model reads them back as past*.
const pastName = (name) => name.replace("present_conv", "past_conv").replace("present", "past_key_values");

/**
 * tools: OpenAI-form [{type: "function", function: {...}}].
 * complete(query) -> {text, ms, promptTokens, firstTokenMs, outTokens}
 */
export async function createRunner({ tf, tok, model, system, tools, maxNewTokens = 128 }) {
  const t0 = performance.now();
  const prefixText = tok.apply_chat_template([{ role: "system", content: system }], { tools, tokenize: false });
  const prefix = tok(prefixText, { add_special_tokens: false });
  const n = prefix.input_ids.dims[1];

  const out = await model({ ...prefix, num_logits_to_keep: new tf.Tensor("int64", [1n], []) });
  // generate() updates a cache in place and frees the GPU tensors it replaces, so the
  // prefix state is kept on the CPU and wrapped afresh for every command.
  const saved = {};
  for (const [name, t] of Object.entries(out)) {
    if (!name.startsWith("present")) continue;
    if (t.location === "gpu-buffer") {
      saved[pastName(name)] = new tf.Tensor(t.type, await t.ort_tensor.getData(), t.dims);
      t.dispose();
    } else {
      saved[pastName(name)] = t;
    }
  }
  const prefixMs = performance.now() - t0;

  async function complete(query) {
    const t1 = performance.now();
    const inputs = tok.apply_chat_template([{ role: "system", content: system }, { role: "user", content: query }],
      { tools, add_generation_prompt: true, return_dict: true });
    const ids = inputs.input_ids.data;
    for (let i = 0; i < n; i++) {
      if (ids[i] !== prefix.input_ids.data[i]) throw new Error(`prompt diverges from the cached prefix at token ${i}`);
    }
    // generate() hands the streamer the prompt first, then each new token
    let puts = 0, firstTokenMs = 0;
    const streamer = { put() { if (puts++ === 1) firstTokenMs = performance.now() - t1; }, end() {} };
    const cache = new tf.DynamicCache({ ...saved });
    const seq = await model.generate({ ...inputs, past_key_values: cache, max_new_tokens: maxNewTokens, do_sample: false, streamer });
    await cache.dispose();                         // only GPU tensors; the saved CPU prefix survives
    const text = tok.decode(seq.slice(null, [ids.length, null])[0], { skip_special_tokens: false });
    return { text, ms: performance.now() - t1, promptTokens: ids.length - n, firstTokenMs, outTokens: puts - 1 };
  }

  return { complete, prefixTokens: n, prefixMs };
}
