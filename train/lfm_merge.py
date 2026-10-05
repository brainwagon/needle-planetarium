"""Fold the LoRA adapter from lfm_train.py into LFM2.5-350M and save a plain
Hugging Face model directory, ready for Liquid's ONNX exporter:

    .venv-torch/bin/python train/lfm_merge.py
    cd vendor/onnx-export && uv run lfm2-export ../../train/lfm-merged --precision q4 \
        --output-dir ../../train/lfm-onnx
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE = "LiquidAI/LFM2.5-350M"

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=str(ROOT / "train" / "lfm-lora"))
ap.add_argument("--out", default=str(ROOT / "train" / "lfm-merged"))
args = ap.parse_args()

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
model.save_pretrained(args.out)
AutoTokenizer.from_pretrained(BASE).save_pretrained(args.out)
print(f"merged model saved to {args.out}")
