"""E0 / Q1 (task list v3): evaluate the saved prompts of a run on the identity-split test set.

For each saved prompt of a run directory (per-round snapshots prompts/rounds_*/roundNN.pt; a run without them:
its final prompt prompts/<domain>_<strategy>_seed<s>.pt counts as its last round) this computes Rank-1 / mAP on
the full test set, the development half and the final half (data_manifests/test_split_dev_final.json). All three
go to the csv; only the development half is printed, plus a check that the full-test mAP equals the run's own
log (proves the prompt / model / evaluation are the same as in training); the final half is never printed.

  python scripts/eval_rounds_1008.py --run_dir <exp dir> --domain cuhk03 --checkpoint <base ckpt> \
      --manifest data_manifests/test_split_dev_final.json --out <csv>
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import glob
import importlib.util
import json
import re
import tempfile

import numpy as np
import torch
import transformers

from adapters.active.image_store import features, retrieval_metrics
from adapters.args_reid import ReIDTrainingArguments
from adapters.baseline_model import load_checkpoint_model


def _eval_active():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_active.py")
    s = importlib.util.spec_from_file_location("eval_active_1008", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def logged_map(run_dir):
    """round -> mAP printed by the run (run.log)."""
    out = {}
    pat = re.compile(r"\[\S+ r(\d+)\].*mAP ([\d.]+) R1")
    p = os.path.join(run_dir, "run.log")
    if os.path.exists(p):
        for line in open(p, errors="ignore"):
            m = pat.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def prompts_of(run_dir, last_round):
    snaps = sorted(glob.glob(os.path.join(run_dir, "prompts", "rounds_*", "round*.pt")))
    if snaps:
        return [(int(re.search(r"round(\d+)\.pt$", p).group(1)), p) for p in snaps], "per-round"
    final = [p for p in glob.glob(os.path.join(run_dir, "prompts", "*.pt"))]
    if len(final) != 1:
        raise FileNotFoundError("{}: no per-round snapshots and {} final prompts".format(run_dir, len(final)))
    return [(last_round, final[0])], "final-only"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", default="")  # comma-separated subset; default all saved
    o = ap.parse_args()
    ea = _eval_active()
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, ea.ActiveArguments))
    args, a = parser.parse_args_into_dataclasses(["--output_dir", tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp")),
                                                  "--checkpoint", o.checkpoint, "--fp16", "True", "--report_to", "none",
                                                  "--eval_num_workers", "8"])
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_checkpoint_model(args, device, o.checkpoint)
    split = ea.load_split(o.domain, device, a.cache_max, args.eval_num_workers)
    man = json.load(open(o.manifest))["domains"][o.domain]
    dev = np.array(man["dev_pids"])
    qd, gd = np.isin(split.q_pids, dev), np.isin(split.g_pids, dev)
    log = logged_map(o.run_dir)
    last = max(log) if log else 0
    items, kind = prompts_of(o.run_dir, last)
    if o.rounds:
        keep = {int(x) for x in o.rounds.split(",")}
        items = [it for it in items if it[0] in keep]
    rows = []
    for r, p in items:
        prompt = torch.load(p, map_location="cpu")["prompt"].to(device)
        qf, gf = features(model, split.query, prompt), features(model, split.gallery, prompt)
        m = lambda qm, gm: retrieval_metrics(qf[torch.as_tensor(qm, device=qf.device)], gf[torch.as_tensor(gm, device=gf.device)],
                                             split.q_pids[qm], split.g_pids[gm], split.q_cams[qm], split.g_cams[gm])
        full_r1, full_map = retrieval_metrics(qf, gf, split.q_pids, split.g_pids, split.q_cams, split.g_cams)
        dev_r1, dev_map = m(qd, gd)
        fin_r1, fin_map = m(~qd, ~gd)
        lg = log.get(r)
        ok = "n/a" if lg is None else ("yes" if abs(lg - full_map) < 0.01 else "NO (delta {:+.3f})".format(full_map - lg))
        rows.append({"run": os.path.basename(o.run_dir.rstrip("/")), "domain": o.domain, "round": r, "source": kind,
                     "dev_mAP": dev_map, "dev_R1": dev_r1, "final_mAP": fin_map, "final_R1": fin_r1,
                     "full_mAP": full_map, "full_R1": full_r1, "logged_full_mAP": lg, "matches_log": ok})
        print("{} r{:02d} dev mAP {:.2f} R1 {:.2f} | full-test matches log: {}".format(
            rows[-1]["run"], r, dev_map, dev_r1, ok), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(o.out)), exist_ok=True)
    with open(o.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
