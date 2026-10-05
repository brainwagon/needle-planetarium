---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-350M
library_name: transformers.js
pipeline_tag: text-generation
tags:
  - tool-calling
  - function-calling
  - on-device
  - webgpu
  - onnx
  - astronomy
  - lora
---

# LFM2.5-350M — Planetarium (ONNX)

A LoRA fine-tune of [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M)
on natural-language planetarium commands ("show me Saturn from Tokyo",
"rewind at an hour per second", "hide the grid"), merged and exported to
4-bit ONNX for [Transformers.js](https://huggingface.co/docs/transformers.js)
on WebGPU.

It powers the demo at **https://mvandewettering.com/needle-planetarium/**, which
runs the model entirely in the browser. Source, tool schemas, training-data
generator and evaluation harness: **https://github.com/brainwagon/needle-planetarium**.

**This is a modified version of LFM2.5-350M**: the weights were changed by
fine-tuning and the model was converted to ONNX. It is distributed under the
[LFM Open License v1.0](LICENSE) — note its commercial-use threshold (§5).

## Tools

Fifteen tools, in [`tools.json`](https://github.com/brainwagon/needle-planetarium/blob/main/site/data/tools.json):
`show_object`, `follow_object`, `set_location`, `set_date_time`, `reset_to_now`,
`set_time_speed`, `zoom`, `set_view`, `look_toward`, `show_overlay`,
`hide_overlay`, `show_path`, `object_info`, `rise_set_times`, `list_visible`.
Calls use LFM's own syntax: `<|tool_call_start|>[show_object(name="Saturn")]<|tool_call_end|>`.

## Results

Exact-match accuracy with all 15 tools in the prompt, compared with Cactus
Needle 3 fine-tuned on the same 6,000 queries:

| Test set | LFM2.5-350M base | Needle 3 fine-tuned | **This model** |
|---|---|---|---|
| `cases4.json` — 60 new commands, written after training | 35/60 (58%) | 50/60 (83%) | **53/60 (88%)** |
| `cases3.json` — 40 commands held out from the templates | 26/40 (65%) | 31/40 (78%) | **38/40 (95%)** |
| `cases2.json` — 36 commands written before the generator | 25/36 (69%) | 33/36 (92%) | **35/36 (97%)** |

`cases3` and `cases2` have informed the training templates; `cases4` was
written afterwards and checked against the training queries for overlap.
On ±2 swings: the untuned model flips several borderline cases between two
q4 exports of the same weights.

**Speed** (Chrome, RTX 4060, WebGPU, `model_q4`): the 1,434-token system + tool
prompt is prefilled once in ~2.3 s and its cache reused, after which a command
takes **~160 ms** (115–245 ms). Without WebGPU, the WASM backend needs the
`q4f32` variant (not included here; `GatherBlockQuantized` is WebGPU-only) and
takes ~3 s per command.

## Training

- 6,000 synthetic examples from templates (`train/gen_data.py`), each with all
  15 tools in the prompt, ~8% off-topic refusals and ~15% two-call requests;
  300 held out for validation.
- LoRA rank 32 on all linear layers, lr 2e-4 cosine, 3 epochs, batch 16
  (4 × 4 accumulation), loss on the reply only (`train/lfm_train.py`);
  RTX 4060 (8 GB), 2.5 hours. Validation loss reached ~0 — the templates are
  easy to fit, which is why the hand-written test sets matter.
- Merged (`train/lfm_merge.py`) and exported with Liquid's
  [onnx-export](https://github.com/Liquid4All/onnx-export) at commit `5ae7589`
  (later commits emit onnxruntime-genai folders that Transformers.js can't load).

## Usage

```js
import { AutoTokenizer, AutoModelForCausalLM } from "@huggingface/transformers";
const id = "mvandewettering/lfm2.5-350m-planetarium-ONNX";
const tok = await AutoTokenizer.from_pretrained(id);
const model = await AutoModelForCausalLM.from_pretrained(id, { dtype: "q4", device: "webgpu" });
const inputs = tok.apply_chat_template(
  [{ role: "system", content: "You control a planetarium app. Call tools to carry out the user's request.\ndate: 2026-10-04 Sun 21:00" },
   { role: "user", content: "show me Saturn from Tokyo" }],
  { tools: TOOLS.map((f) => ({ type: "function", function: f })), add_generation_prompt: true, return_dict: true });
const out = await model.generate({ ...inputs, max_new_tokens: 128, do_sample: false });
```

The demo's [`site/js/lfm.js`](https://github.com/brainwagon/needle-planetarium/blob/main/site/js/lfm.js)
adds the prompt cache and a parser for the calls.

## Limitations

- Trained with exactly this system prompt and these 15 tool schemas; other
  tools are untested.
- Template-generated training data: phrasing far from the templates can still
  fail (e.g. "resume real time" → `reset_to_now`).
- It knows nothing about the sky itself: it maps words to tool calls, and the
  app resolves names to coordinates.
- No calibrated confidence: unlike Needle, nothing estimates how sure it is.
