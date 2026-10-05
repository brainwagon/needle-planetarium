# Needle Planetarium

A planetarium you drive with plain English (typed or spoken), powered by small
tool-calling models running **entirely in the browser**:
[Cactus Needle 3](https://huggingface.co/Cactus-Compute/needle3) through its
WebAssembly engine, and — where WebGPU is available — a fine-tuned
[LFM2.5-350M](https://huggingface.co/mvandewettering/lfm2.5-350m-planetarium-ONNX)
through Transformers.js. A side panel, *Inside the Needle*, shows what the model heard, which
tool it called and how sure it was, where the command landed in the
tool-embedding space, and which words mattered.

## How it fits together

```
 voice ──► Whistle (speech→text, WASM) ─┐
 typed text ────────────────────────────┴─► Needle 3 (WASM) or LFM2.5 (WebGPU), each in a Web Worker
                                              │  one JSON tool call, e.g.
                                              │  show_object(name: "Saturn")
                                              ▼
       tools.js: name → coordinates (star catalog, gazetteer, ephemeris)
                                              ▼
                     sky.js: canvas planetarium (astronomy-engine)
```

Neither model knows much about the sky, by design: it maps words onto a tool
and copies argument values out of the request. Everything it doesn't know
(where Saturn is tonight, where Tokyo is, when the Moon rises) lives in the
tools.

## Layout

| Path | What |
|---|---|
| `site/` | The static site deployed to GitHub Pages (`.github/workflows/pages.yml`) |
| `site/js/needle-worker.js` | One Needle engine per Web Worker; weights cached with the Cache API |
| `site/js/lfm-worker.js`, `site/js/lfm.js` | LFM2.5 on WebGPU; `lfm.js` (call parser + prompt-cache runner) is shared with the Node harness |
| `site/js/sky.js` | Stereographic alt-az sky renderer |
| `site/js/tools.js` | Object/place resolvers and one executor per tool |
| `site/js/inspector.js` | The *Inside the Needle* panel |
| `site/data/tools.json` | The 15 tool schemas Needle sees |
| `site/tools/` | Scripts that build `sky.json` and `places.json` from the raw catalogs |
| `spike/` | Node harnesses that score models against test cases (`run.cjs` for Needle, `run-tf.mjs` for Transformers.js models); `spike/webgpu/` is a standalone WebGPU benchmark page |
| `train/gen_data.py` | Synthetic fine-tuning data for the planetarium tools |
| `train/lfm_train.py`, `train/lfm_merge.py` | LoRA fine-tune of LFM2.5-350M (PyTorch) and the merge before ONNX export |

## Running locally

```sh
cd site && python3 -m http.server 8765
# open http://127.0.0.1:8765/   (?q=show%20me%20Saturn runs a command on load)
```

The weights load from Hugging Face, pinned to a revision, and are cached by the
browser after the first visit: Needle 3 (35 MB) and Whistle (17 MB) first, then
the fine-tuned Needle (63 MB) and, if the browser has WebGPU and isn't in data
saver mode, the fine-tuned LFM2.5 (289 MB), which becomes the default once ready.

## Results

Exact match with all 15 tools in the prompt. `cases4.json` (60) was written
after all training and checked against the training queries; `cases3.json` (40)
and `cases2.json` (36) have informed the training templates, so read them as
optimistic.

| Model | cases4 | cases3 | cases2 | per command |
|---|---|---|---|---|
| Needle 3 base | 29 | 20 | 18 | ~0.8 s (WASM) |
| Needle 3 fine-tuned, 15-tool data | 50 | 31 | 33 | ~0.55 s (WASM) |
| LFM2.5-350M base | 35 | 26 | 25 | — |
| **LFM2.5-350M fine-tuned** | **53** | **38** | **35** | **~0.16 s** (WebGPU, RTX 4060) |

Needle's remaining misses are mostly its grounding gate: an enum value with no
literal evidence in the request ("closer" → zoom `in`) is withheld even when
right. LFM has no such gate. Its speed comes from prefilling the 1,434-token
system + tool prompt once and reusing the cache (`site/js/lfm.js`); without
WebGPU it needs ~3 s per command, so the site falls back to Needle.

## Evaluating

```sh
# fetch the engine + weights for Node (git-ignored)
mkdir -p vendor/needle && cd vendor/needle
for f in wasm/needle.js wasm/needle.wasm needle3.cact; do
  curl -sLO https://huggingface.co/Cactus-Compute/needle3/resolve/main/$f; done
cd ../..
node spike/run.cjs --tools tools2.json --cases cases4.json                 # base Needle
node spike/run.cjs --tools tools2.json --cases cases4.json --model X.cact  # any .cact

cd spike && npm install                                                    # Transformers.js
node run-tf.mjs --model lfm --cases cases4.json --cache                    # LFM2.5 base (from HF)
node run-tf.mjs --model lfm --cases cases4.json --cache --path DIR         # a local ONNX export
```

`spike/webgpu/` benchmarks a local export in a real browser:
`python3 spike/webgpu/serve.py 8767` from the repo root, then open
`http://localhost:8767/spike/webgpu/?auto=1`; results are appended to
`spike/webgpu/results.log`.

## Fine-tuning

Training data for both models comes from the same generator; `--all-tools`
shows every example with all 15 tools, as both fine-tuned models run.

**Needle 3** (JAX):

```sh
uv venv .venv && uv pip install "cactus-needle[train,gpu]"
.venv/bin/python train/gen_data.py --n 6000 --all-tools --out train/data15.jsonl
.venv/bin/needle finetune train/data15.jsonl --epochs 3 --batch-size 4 --max-len 1280 \
    --checkpoint checkpoints/needle3.safetensors --checkpoint-dir train/checkpoints15 \
    --out train/checkpoints15/planetarium15_lora.safetensors
.venv/bin/needle build checkpoints/needle3.safetensors \
    --lora train/checkpoints15/planetarium15_lora.safetensors --out train/needle3-planetarium15.cact
```

`needle finetune` truncates long examples from the end without warning, so
`--max-len` must cover the longest (15-tool examples are up to 1,233 tokens).
Batch 4 at 1280 fits in an 8 GB GPU; about 2 hours on an RTX 4060.

**LFM2.5-350M** (PyTorch), then ONNX via Liquid's exporter pinned to commit
`5ae7589` (later commits produce onnxruntime-genai folders that Transformers.js
can't load):

```sh
uv venv .venv-torch --python 3.12 && uv pip install --python .venv-torch/bin/python torch transformers peft accelerate
.venv/bin/python train/gen_data.py --n 6000 --out train/data.jsonl   # lfm_train.py adds all 15 tools itself
.venv-torch/bin/python train/lfm_train.py              # --trial first: 40 steps, prints speed and memory
.venv-torch/bin/python train/lfm_merge.py              # -> train/lfm-merged/
git clone https://github.com/Liquid4All/onnx-export vendor/onnx-export && git -C vendor/onnx-export checkout 5ae7589
cd vendor/onnx-export && uv run lfm2-export ../../train/lfm-merged --precision q4 --output-dir ../../train/lfm-onnx
```

About 2.5 hours on an RTX 4060 (the `causal_conv1d` kernel isn't installed, so
LFM's convolutions use the slower reference path).

Published at
[mvandewettering/needle3-planetarium](https://huggingface.co/mvandewettering/needle3-planetarium)
and [mvandewettering/lfm2.5-350m-planetarium-ONNX](https://huggingface.co/mvandewettering/lfm2.5-350m-planetarium-ONNX).
A local Needle fine-tune loses the calibration head, so its score is a raw
decode probability, and the embedding readout, so it sees all 15 tools rather
than a retrieved five. The site uses base Needle for the tool-space map.

## Credits

- Model and engine: [Cactus Needle](https://github.com/cactus-compute/needle), Apache-2.0 (`site/engine/LICENSE-needle`)
- [Liquid LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M), fine-tuned here, under the LFM Open License v1.0 (copy in the model repo; note its commercial-use threshold)
- [Transformers.js](https://github.com/huggingface/transformers.js), Apache-2.0
- Star, constellation and Messier data: [d3-celestial](https://github.com/ofrohn/d3-celestial), BSD-3-Clause, © Olaf Frohn
- Ephemerides: [Astronomy Engine](https://github.com/cosinekitty/astronomy), MIT
- Places: [GeoNames](https://www.geonames.org/), CC-BY 4.0
