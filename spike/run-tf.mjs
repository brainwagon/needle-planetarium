// Score a general small tool-calling model (via Transformers.js / ONNX) on the same
// cases as run.cjs, for comparison with Needle.
//   node spike/run-tf.mjs --model functiongemma|lfm [--cases cases3.json] [--dtype q4] [--verbose]
//   --path DIR loads a local export (e.g. a fine-tune) in place of the model's Hugging Face id
//   --cache prefills the system + tool prompt once and reuses it (site/js/lfm.js)
import fs from "node:fs";
import path from "node:path";
import * as tf from "@huggingface/transformers";

import { dateFact, caseMatches, parseFunctionGemma, parseLfm } from "./common.mjs";
import { LFM_SYSTEM, createRunner } from "../site/js/lfm.js";

const here = path.dirname(new URL(import.meta.url).pathname);
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const verbose = args.includes("--verbose");

const MODELS = {
  functiongemma: {
    id: "onnx-community/functiongemma-270m-it-ONNX",
    // the developer-turn wording FunctionGemma was trained with
    system: "You are a model that can do function calling with the following functions",
    parse: parseFunctionGemma,
  },
  lfm: {
    id: "LiquidAI/LFM2.5-350M-ONNX",
    system: LFM_SYSTEM,
    parse: parseLfm,
  },
};

const spec = MODELS[opt("--model", "lfm")];
if (!spec) throw new Error(`--model must be one of ${Object.keys(MODELS).join(", ")}`);
const tools = JSON.parse(fs.readFileSync(path.join(here, opt("--tools", "tools2.json")), "utf8"))
  .map((f) => ({ type: "function", function: f }));
const cases = JSON.parse(fs.readFileSync(path.join(here, opt("--cases", "cases3.json")), "utf8"));
// --no-date: drop the date line (some models are thrown by anything beyond their trained system text)
const system = args.includes("--no-date") ? spec.system : `${spec.system}\n${dateFact()}`;

let id = spec.id;
if (opt("--path")) {
  const dir = path.resolve(opt("--path"));
  tf.env.localModelPath = path.dirname(dir) + "/";
  tf.env.allowRemoteModels = false;
  id = path.basename(dir);
}

let t0 = performance.now();
const tok = await tf.AutoTokenizer.from_pretrained(id);
const model = await tf.AutoModelForCausalLM.from_pretrained(id, { dtype: opt("--dtype", "q4") });
console.log(`${opt("--path") || id} (${opt("--dtype", "q4")}) ready in ${(performance.now() - t0).toFixed(0)} ms`);

async function uncached(q) {
  const inputs = tok.apply_chat_template([{ role: "system", content: system }, { role: "user", content: q }],
    { tools, add_generation_prompt: true, return_dict: true });
  const n = inputs.input_ids.dims[1];
  const t1 = performance.now();
  const out = await model.generate({ ...inputs, max_new_tokens: 128, do_sample: false });
  return { text: tok.decode(out.slice(null, [n, null])[0], { skip_special_tokens: false }), ms: performance.now() - t1 };
}
let complete = uncached;
if (args.includes("--cache")) {
  const runner = await createRunner({ tf, tok, model, system, tools });
  console.log(`prefix ${runner.prefixTokens} tokens cached in ${runner.prefixMs.toFixed(0)} ms`);
  complete = runner.complete;
}

let pass = 0, totalMs = 0;
for (const c of cases) {
  const { text, ms } = await complete(c.q);
  totalMs += ms;
  const calls = spec.parse(text);
  const ok = caseMatches(c, calls);
  if (ok) pass++;
  const callStr = calls.map((k) => `${k.name}(${JSON.stringify(k.arguments)})`).join(" ; ") || "(none)";
  console.log(`${ok ? "PASS" : "FAIL"} ${ms.toFixed(0).padStart(5)}ms  ${c.q}\n      -> ${callStr}`);
  if (verbose || (!calls.length && c.want.length)) console.log("      raw:", JSON.stringify(text));
}
const promptTokens = tok.apply_chat_template([{ role: "system", content: system }, { role: "user", content: "x" }],
  { tools, add_generation_prompt: true, return_dict: true }).input_ids.dims[1];
console.log(`\n${pass}/${cases.length} passed; mean latency ${(totalMs / cases.length).toFixed(0)} ms (prompt ~${promptTokens} tokens)`);
