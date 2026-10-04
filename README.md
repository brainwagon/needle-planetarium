# Needle Planetarium

A planetarium you drive with plain English (typed or spoken), powered by
[Cactus Needle 3](https://huggingface.co/Cactus-Compute/needle3), a small
tool-calling model, running **entirely in the browser** via its WebAssembly
engine. A side panel, *Inside the Needle*, shows what the model heard, which
tool it called and how sure it was, where the command landed in the
tool-embedding space, and which words mattered.

## How it fits together

```
 voice ──► Whistle (speech→text, WASM) ─┐
 typed text ────────────────────────────┴─► Needle 3 (WASM, in a Web Worker)
                                              │  one JSON tool call, e.g.
                                              │  show_object(name: "Saturn")
                                              ▼
       tools.js: name → coordinates (star catalog, gazetteer, ephemeris)
                                              ▼
                     sky.js: canvas planetarium (astronomy-engine)
```

Needle has almost no world knowledge, by design: it maps words onto a tool
and copies argument values out of the request. Everything it doesn't know
(where Saturn is tonight, where Tokyo is, when the Moon rises) lives in the
tools.

## Layout

| Path | What |
|---|---|
| `site/` | The static site deployed to GitHub Pages (`.github/workflows/pages.yml`) |
| `site/js/needle-worker.js` | One Needle engine per Web Worker; weights cached with the Cache API |
| `site/js/sky.js` | Stereographic alt-az sky renderer |
| `site/js/tools.js` | Object/place resolvers and one executor per tool |
| `site/js/inspector.js` | The *Inside the Needle* panel |
| `site/data/tools.json` | The 15 tool schemas Needle sees |
| `site/tools/` | Scripts that build `sky.json` and `places.json` from the raw catalogs |
| `spike/` | Node harness that scores a tool set and model against test cases |
| `train/gen_data.py` | Synthetic fine-tuning data for the planetarium tools |

## Running locally

```sh
cd site && python3 -m http.server 8765
# open http://127.0.0.1:8765/   (?q=show%20me%20Saturn runs a command on load)
```

The base weights (35 MB) and Whistle (17 MB) load from Hugging Face, pinned
to a revision, and are cached by the browser after the first visit.

## Evaluating

```sh
# fetch the engine + weights for Node (git-ignored)
mkdir -p vendor/needle && cd vendor/needle
for f in wasm/needle.js wasm/needle.wasm needle3.cact; do
  curl -sLO https://huggingface.co/Cactus-Compute/needle3/resolve/main/$f; done
cd ../..
node spike/run.cjs --tools tools2.json --cases cases3.json                 # base model
node spike/run.cjs --tools tools2.json --cases cases3.json --model X.cact  # any .cact
```

`cases3.json` is hand-written and held out from the training templates.

## Fine-tuning

```sh
uv venv .venv && uv pip install "cactus-needle[train,gpu]"
.venv/bin/python train/gen_data.py --n 6000
.venv/bin/needle finetune train/data.jsonl --epochs 3 --batch-size 8 --max-len 768 \
    --checkpoint-dir train/checkpoints --out train/checkpoints/planetarium_lora.safetensors
.venv/bin/needle build train/checkpoints/needle3.safetensors \
    --lora train/checkpoints/planetarium_lora.safetensors --out site/models/needle3-planetarium.cact
```

Batch 8 at length 768 fits in an 8 GB GPU (batch 16 at the default 1024 does not).

## Credits

- Model and engine: [Cactus Needle](https://github.com/cactus-compute/needle), Apache-2.0 (`site/engine/LICENSE-needle`)
- Star, constellation and Messier data: [d3-celestial](https://github.com/ofrohn/d3-celestial), BSD-3-Clause, © Olaf Frohn
- Ephemerides: [Astronomy Engine](https://github.com/cosinekitty/astronomy), MIT
- Places: [GeoNames](https://www.geonames.org/), CC-BY 4.0
