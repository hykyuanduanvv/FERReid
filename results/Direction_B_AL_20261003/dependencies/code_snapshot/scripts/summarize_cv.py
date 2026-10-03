"""Summarize a cross-validation run made by scripts/run_cv.sh.

For each fold (experiments/<exp>/val_<domain>/trainer_state.json): training curves
(loss terms and feature std, averaged over windows) and the validation curve.

Usage: python scripts/summarize_cv.py <exp_name> [window]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    exp = sys.argv[1]
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    exp_dir = os.path.join(ROOT, "experiments", exp)
    best = {}
    for fold in sorted(d for d in os.listdir(exp_dir) if d.startswith("val_")):
        path = os.path.join(exp_dir, fold, "trainer_state.json")
        if not os.path.isfile(path):
            print("== {}: no trainer_state.json (not finished?)\n".format(fold))
            continue
        hist = json.load(open(path))["log_history"]
        train = [h for h in hist if "loss" in h and "step" in h]
        val = [h for h in hist if any(k.startswith("val_") and k.endswith("_rank1") for k in h)]
        name = fold[len("val_"):]

        print("== fold {}  (train logs: {}, val points: {})".format(fold, len(train), len(val)))
        print("  {:>9} {:>8} {:>8} {:>8} {:>8} {:>7}".format("steps", "loss", "icl", "id", "ot", "std"))
        steps = np.array([h["step"] for h in train])
        for lo in range(0, int(steps.max()) if len(steps) else 0, window):
            sel = [h for h in train if lo < h["step"] <= lo + window]
            if not sel:
                continue
            m = lambda k: np.mean([h[k] for h in sel if k in h])
            print("  {:>4}-{:<4} {:>8.3f} {:>8.3f} {:>8.4f} {:>8.4f} {:>7.4f}".format(
                lo + 1, lo + window, m("loss"), m("icl_loss"), m("id_loss"), m("ot_loss"), m("std")))
        if val:
            print("  val ({}):  step: rank1 / mAP".format(name))
            for h in val:
                print("    {:>5}: {:6.2f} / {:6.2f}".format(
                    h["step"], h["val_{}_rank1".format(name)], h["val_{}_mAP".format(name)]))
            b = max(val, key=lambda h: h["val_{}_mAP".format(name)])
            best[name] = (b["step"], b["val_{}_rank1".format(name)], b["val_{}_mAP".format(name)])
        print()
    if best:
        print("best validation mAP per fold:")
        for name, (step, r1, mAP) in best.items():
            print("  {:<11} step {:>5}: rank1 {:6.2f}  mAP {:6.2f}".format(name, step, r1, mAP))


if __name__ == "__main__":
    main()
