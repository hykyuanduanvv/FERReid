"""LoRA fine-tuning of a multimodal LLM as the "same person?" annotator, on labeled SOURCE pairs only
(scripts/make_vlm_source_pairs.py). The target domain is never seen; scripts/diag_vlm.py --adapter evaluates the
result on the target question sets.

The objective matches the score of scripts/diag_vlm.py: at the first answer position, the log-probabilities of
"yes" and "no" (all first-token variants) form a two-way softmax, trained with cross-entropy against the label.
LoRA on the language model's attention and MLP projections; the vision encoder stays frozen.

  python scripts/finetune_vlm.py --pairs experiments/vlm_train_msmt/train_pairs.csv \\
      --model /root/autodl-tmp/basic-models/Qwen3-VL-8B-Instruct --out experiments/vlm_train_msmt/lora
"""
import argparse
import csv
import math
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import numpy as np
import torch

from scripts.diag_vlm import QUESTIONS, first_token_ids, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--question", default="default", choices=sorted(QUESTIONS))
    p.add_argument("--max_per_domain", type=int, default=6000)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    from transformers import AutoProcessor, AutoModelForImageTextToText, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    rows = list(csv.DictReader(open(a.pairs)))
    by_dom = {}
    for r in rows:
        by_dom.setdefault(r["domain"], []).append(r)
    rows = []
    for d, rs in by_dom.items():
        rs = [rs[i] for i in rng.permutation(len(rs))[:a.max_per_domain]]
        print("{}: {} pairs ({} positive)".format(d, len(rs), sum(int(r["same"]) for r in rs)))
        rows += rs
    rows = [rows[i] for i in rng.permutation(len(rows))]

    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "left"  # the answer position is the last one in every row
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    yes_ids = torch.tensor(first_token_ids(proc.tokenizer, ["Yes", "yes"]), device="cuda")
    no_ids = torch.tensor(first_token_ids(proc.tokenizer, ["No", "no"]), device="cuda")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                             {"type": "text", "text": QUESTIONS[a.question]}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    steps = int(math.ceil(len(rows) * a.epochs / a.batch))
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, steps // 20), steps)
    model.train()
    t0, run_loss, run_acc = time.time(), [], []
    for step in range(steps):
        batch = [rows[(step * a.batch + k) % len(rows)] for k in range(a.batch)]
        imgs = []
        for r in batch:
            imgs += [load(r["path_a"], a.height), load(r["path_b"], a.height)]
        inputs = proc(text=[text] * len(batch), images=imgs, return_tensors="pt", padding=True).to("cuda")
        logits = model(**inputs, logits_to_keep=1).logits[:, -1].float()
        lp = torch.log_softmax(logits, -1)
        two = torch.stack([torch.logsumexp(lp[:, no_ids], 1), torch.logsumexp(lp[:, yes_ids], 1)], 1)
        y = torch.tensor([int(r["same"]) for r in batch], device="cuda")
        loss = torch.nn.functional.cross_entropy(two, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run_loss.append(float(loss)); run_acc.append(float((two.argmax(1) == y).float().mean()))
        if (step + 1) % 25 == 0 or step + 1 == steps:
            print("step {}/{} loss {:.4f} acc {:.3f} | {:.1f}s/it".format(
                step + 1, steps, np.mean(run_loss[-25:]), np.mean(run_acc[-25:]), (time.time() - t0) / (step + 1)),
                flush=True)
    model.save_pretrained(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
