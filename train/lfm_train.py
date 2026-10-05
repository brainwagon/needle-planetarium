"""LoRA fine-tune of LiquidAI/LFM2.5-350M on the planetarium data from gen_data.py.

Unlike the Needle tune, every example shows all 15 tools: that is how the app runs
the model (there is no retrieval step), so training matches inference.

    .venv-torch/bin/python train/lfm_train.py --trial      # 40 steps, prints speed/memory
    .venv-torch/bin/python train/lfm_train.py              # full run -> train/lfm-lora/
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
BASE = "LiquidAI/LFM2.5-350M"
# Same system text spike/run-tf.mjs uses, followed by the date line.
SYSTEM = "You control a planetarium app. Call tools to carry out the user's request."
TOOLS = json.loads((ROOT / "spike" / "tools2.json").read_text())
REFUSAL = "Sorry, I can only control the planetarium."


def pyval(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(str(v))


def render_calls(answers):
    """LFM's tool-call syntax: <|tool_call_start|>[name(k="v", n=3), ...]<|tool_call_end|>"""
    if not answers:
        return REFUSAL
    calls = ", ".join(f"{a['name']}({', '.join(f'{k}={pyval(v)}' for k, v in a['arguments'].items())})"
                      for a in answers)
    return f"<|tool_call_start|>[{calls}]<|tool_call_end|>"


def messages_for(row):
    date = row["system"].split(";")[0]                    # "date: 2026-03-19 Thu 02:16"
    return [{"role": "system", "content": f"{SYSTEM}\n{date}"},
            {"role": "user", "content": row["query"]}]


def encode(tok, row):
    """Token ids with labels masked to the assistant's reply."""
    tools = [{"type": "function", "function": t} for t in TOOLS]
    msgs = messages_for(row)
    prompt = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
    reply = render_calls(row["answers"]) + "<|im_end|>"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(reply, add_special_tokens=False)["input_ids"]
    return p_ids + r_ids, [-100] * len(p_ids) + r_ids


def batches(examples, size, shuffle):
    order = list(range(len(examples)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), size):
        chunk = [examples[j] for j in order[i:i + size]]
        n = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), n), 0, dtype=torch.long)
        lab = torch.full((len(chunk), n), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for k, (a, b) in enumerate(chunk):
            ids[k, :len(a)] = torch.tensor(a)
            lab[k, :len(b)] = torch.tensor(b)
            att[k, :len(a)] = 1
        yield ids, lab, att


@torch.no_grad()
def val_loss(model, val, bs):
    model.eval()
    tot = n = 0
    for ids, lab, att in batches(val, bs, False):
        out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda())
        tot += out.loss.item() * len(ids)
        n += len(ids)
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "train" / "data.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "train" / "lfm-lora"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--no-checkpointing", action="store_true", help="faster, if it fits in memory")
    ap.add_argument("--trial", action="store_true", help="40 steps, report speed and memory, save nothing")
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(BASE)
    rows = [json.loads(l) for l in open(args.data)]
    random.shuffle(rows)
    data = [encode(tok, r) for r in rows]
    val, train = data[:args.val], data[args.val:]
    print(f"{len(train)} train / {len(val)} val; mean length {sum(len(a) for a, _ in data) / len(data):.0f} tokens")
    print("sample reply:", tok.decode([t for t in train[0][1] if t != -100]))

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train) / (args.batch_size * args.accum))
    total = 40 if args.trial else steps_per_epoch * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total), total)
    model.train()

    step, t0, running = 0, time.time(), []
    for epoch in range(args.epochs):
        for i, (ids, lab, att) in enumerate(batches(train, args.batch_size, True)):
            loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss / args.accum
            loss.backward()
            running.append(loss.item() * args.accum)
            if (i + 1) % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 20 == 0:
                el = time.time() - t0
                print(f"step {step}/{total}  loss {sum(running) / len(running):.4f}  "
                      f"{el / step:.2f} s/step  eta {(total - step) * el / step / 60:.0f} min  "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
                running = []
            if step >= total:
                break
        if args.trial:
            print(f"trial: {(time.time() - t0) / step:.2f} s/step -> full run of "
                  f"{steps_per_epoch * args.epochs} steps ~{steps_per_epoch * args.epochs * (time.time() - t0) / step / 60:.0f} min")
            return
        print(f"epoch {epoch + 1}/{args.epochs}  val loss {val_loss(model, val, args.batch_size):.4f}", flush=True)
        model.save_pretrained(args.out)
    print(f"adapter saved to {args.out}; {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
