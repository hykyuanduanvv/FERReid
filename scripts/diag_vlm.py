"""Can a multimodal LLM verify "same person?" better than the ReID model? (decides whether an MLLM gets a role,
e.g. as a pre-filter before the human annotator)

Reads verify_pairs.csv of scripts/diag_retrieval.py (proposed cross-camera pairs with their true label and
ReID similarity), asks the model "Are these two images the same person?" for each pair and scores the answer
by log P(yes) - log P(no) of the next token. Reports, per domain and on the hardest half of the pairs (ReID
similarity closest to tau): accuracy and ROC-AUC of the MLLM vs the ReID similarity (accuracy at tau).

  python scripts/diag_vlm.py --pairs experiments/diag_base/verify_pairs.csv --model Qwen/Qwen2.5-VL-7B-Instruct \
      --out experiments/diag_base/vlm_qwen7b.csv

Needs a transformers version with AutoModelForImageTextToText and a chat-template processor (Qwen2-VL /
Qwen2.5-VL, InternVL3-hf, ...). Written against Qwen2.5-VL; not run in the CPU test environment.
"""
import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import numpy as np
import torch
from PIL import Image

QUESTION = ("These are two pedestrian images from different surveillance cameras. Are they the same person? "
            "Judge by clothing, body shape, accessories and other identity cues, not by pose, lighting or "
            "background. Answer yes or no.")
QUESTIONS = {
    "default": QUESTION,
    "strict": ("These are two pedestrian images from different surveillance cameras. Compare the upper-body clothing "
               "(colour, pattern, sleeves), the lower-body clothing, the shoes, bags or other carried items, hair and "
               "body shape one by one; ignore pose, viewpoint, lighting, resolution and background. Many different "
               "people wear similar clothes, so answer yes only if every visible cue matches and nothing contradicts. "
               "Are they the same person? Answer yes or no."),
}


def first_token_ids(tokenizer, words):
    ids = set()
    for w in words:
        for variant in (w, " " + w):
            t = tokenizer.encode(variant, add_special_tokens=False)
            if t:
                ids.add(t[0])
    return sorted(ids)


def load(path, height):
    img = Image.open(path).convert("RGB")
    if img.height < height:  # small pedestrian crops: upscale so the vision encoder sees enough patches
        img = img.resize((max(28, round(img.width * height / img.height)), height), Image.BICUBIC)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--out", default="")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--max_pairs", type=int, default=0)
    p.add_argument("--question", default="default", choices=sorted(QUESTIONS))
    p.add_argument("--adapter", default="", help="LoRA adapter of scripts/finetune_vlm.py")
    a = p.parse_args()
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from scripts.diag_retrieval import auc

    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).eval()
    yes_ids = first_token_ids(proc.tokenizer, ["Yes", "yes"])
    no_ids = first_token_ids(proc.tokenizer, ["No", "no"])
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": QUESTIONS[a.question]}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    rows = list(csv.DictReader(open(a.pairs)))
    if a.max_pairs:
        rows = rows[:a.max_pairs]
    for n, r in enumerate(rows):
        inputs = proc(text=[text], images=[load(r["path_a"], a.height), load(r["path_b"], a.height)],
                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        r["vlm_score"] = float(torch.logsumexp(lp[yes_ids], 0) - torch.logsumexp(lp[no_ids], 0))
        if n % 50 == 0:
            print("{}/{}".format(n, len(rows)), flush=True)

    out = a.out or os.path.splitext(a.pairs)[0] + "_vlm.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n{:<10} {:<6} {:>5} {:>9} {:>9} {:>9} {:>9}".format("domain", "subset", "n", "vlm_acc", "vlm_auc",
                                                                 "reid_acc", "reid_auc"))
    for dom in sorted({r["domain"] for r in rows}):
        R = [r for r in rows if r["domain"] == dom]
        y = np.array([int(r["same"]) for r in R], bool)
        v = np.array([r["vlm_score"] for r in R], float)
        s = np.array([float(r["sim"]) for r in R]); tau = np.array([float(r["tau"]) for r in R])
        hard = np.abs(s - tau) <= np.median(np.abs(s - tau))
        kinds = sorted({r.get("kind", "") for r in R} - {""})  # scripts/make_vlm_pairs.py question sets
        subsets = [("all", np.ones_like(y)), ("hard", hard)] + [
            (k, np.array([r.get("kind") == k for r in R])) for k in kinds]
        for subset, m in subsets:
            print("{:<10} {:<8} {:>5} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}".format(
                dom, subset, int(m.sum()), float(((v[m] > 0) == y[m]).mean()), auc(v[m], y[m]),
                float(((s[m] > tau[m]) == y[m]).mean()), auc(s[m], y[m])))
    print("wrote", out)


if __name__ == "__main__":
    main()
