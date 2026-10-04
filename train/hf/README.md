---
license: apache-2.0
base_model: Cactus-Compute/needle3
library_name: cactus-needle
pipeline_tag: text-generation
tags:
  - tool-calling
  - function-calling
  - on-device
  - webassembly
  - astronomy
  - lora
---

# Needle 3 — Planetarium

A LoRA fine-tune of [Cactus-Compute/needle3](https://huggingface.co/Cactus-Compute/needle3)
on natural-language planetarium commands ("show me Saturn from Tokyo",
"rewind at an hour per second", "hide the grid"), merged and exported as a
4-bit `.cact` archive for the Needle engine.

It powers the demo at **https://mvandewettering.com/needle-planetarium/**, which
runs the model entirely in the browser through Needle's WebAssembly engine.
Source, tool schemas, training-data generator and evaluation harness:
**https://github.com/brainwagon/needle-planetarium**.

## Tools

Fifteen tools, in [`tools.json`](https://github.com/brainwagon/needle-planetarium/blob/main/site/data/tools.json):
`show_object`, `follow_object`, `set_location`, `set_date_time`, `reset_to_now`,
`set_time_speed`, `zoom`, `set_view`, `look_toward`, `show_overlay`,
`hide_overlay`, `show_path`, `object_info`, `rise_set_times`, `list_visible`.

## Results

Exact-match accuracy on 40 hand-written commands held out from the training
templates (`spike/cases3.json`) and 36 earlier ones (`spike/cases2.json`),
run through the shipped WASM engine with all 15 tools declared. The base model
narrows them to 5 by embedding retrieval; this fine-tune has no embedding
readout (see Limitations), so it sees all 15:

| Test set | Base Needle 3 | This fine-tune |
|---|---|---|
| `cases3.json` — 40 commands with phrasings unseen in training | 20/40 (50%) | **26/40 (65%)** |
| `cases2.json` — 36 commands written before the generator | 18/36 (50%) | **31/36 (86%)** |
| Trainer's own validation split (600 template examples) | — | 538/600 (90%) |

Mean latency in Node's WASM runtime is ~0.6 s per command for both.

Remaining misses are mostly synonyms with no literal evidence for an enum
value ("freeze time" → `pause`), which the engine's grounding gate withholds
into `suppressed_calls` even when the call is right, and phrasings far from
the templates ("closer", "normal speed").

## Training

- 6,000 synthetic examples from templates (`train/gen_data.py`), each shown
  with a 5-tool subset that includes near-miss distractors, ~8% off-topic
  refusals and ~15% two-call requests. Arguments are always spans of the request.
- `needle finetune --epochs 3 --batch-size 8 --max-len 768`, LoRA rank 16 on
  the attention projections, base frozen; RTX 4060 (8 GB).
- `needle build --lora ...` merges the adapter and exports at 4 bits.

## Limitations

- Local fine-tunes don't train Needle's calibration head, and `needle build`
  drops it. The engine still reports a `confidence`, but it is the raw decode
  probability, not calibrated.
- The dropped head also removes the embedding readout: `needle_embed` returns
  an error, and with more than five tools the engine can't do tool retrieval,
  so the model sees every declared tool. It was trained on 5-tool subsets.
- Training data is template-generated, so phrasing far from the templates
  may still fail.
- Like the base model, it knows nothing about the sky itself: it maps words to
  tool calls, and the app resolves names to coordinates.

## Usage

```python
import needle
agent = needle.Needle(weights="needle3-planetarium.cact", tools=TOOLS)
agent.complete("draw Mars's path over 120 days")
```

or load the `.cact` into the WASM engine (`needle_load`) as the demo does.
