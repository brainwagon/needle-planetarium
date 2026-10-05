// Shared by the Node harnesses (run.cjs, run-tf.mjs) and the browser prototype
// (webgpu/): the date fact, tool-call parsers and case scoring.

import { scalar } from "../site/js/lfm.js";
export { parseLfm } from "../site/js/lfm.js";
export { dateFact } from "../site/js/needle.js";

// FunctionGemma: <start_function_call>call:name{key:<escape>value<escape>,n:3}<end_function_call>
export function parseFunctionGemma(text) {
  const calls = [];
  for (const m of text.matchAll(/<start_function_call>call:(\w+)\{([\s\S]*?)\}<end_function_call>/g)) {
    const a = {};
    for (const kv of m[2].matchAll(/(\w+):(?:<escape>([\s\S]*?)<escape>|([^,{}]*))/g))
      a[kv[1]] = kv[2] !== undefined ? kv[2] : scalar(kv[3]);
    calls.push({ name: m[1], arguments: a });
  }
  return calls;
}


export function argMatches(want, got) {
  if (got === undefined) return false;
  if (typeof want === "string") {
    const g = String(got).toLowerCase();
    return want.toLowerCase().split("|").some((w) => g.includes(w) || w.includes(g) && g.length > 2);
  }
  if (typeof want === "number") return Number(got) === want;
  return got === want;
}

export function callMatches(want, got) {
  if (!got || got.name !== want.name) return false;
  const a = got.arguments || {};
  return Object.entries(want.args).every(([k, v]) => argMatches(v, a[k]));
}

// order-insensitive, per the Needle test-suite convention; "alt" lists other acceptable answers
export function caseMatches(c, calls) {
  const okFor = (want) => calls.length === want.length && want.every((w) => calls.some((k) => callMatches(w, k)));
  return [c.want, ...(c.alt || [])].some(okFor);
}

