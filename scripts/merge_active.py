"""Merge the active.csv of several runs (e.g. one task per seed) and rebuild summary.csv / paired.csv.

  python scripts/merge_active.py experiments/rep_cuhk03_s* --out experiments/rep_cuhk03 --paired_ref repair_random
Rows are de-duplicated on (domain, strategy, seed, round), so the base row (round 0) is kept once per domain.
"""
import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from scripts.eval_active import paired, summarize, write_csv


def _num(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--paired_ref", default="random")
    a = ap.parse_args()
    rows, seen = [], set()
    for run in a.runs:
        path = os.path.join(run, "active.csv")
        if not os.path.isfile(path):
            print("skip {} (no active.csv)".format(run))
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                r = {k: _num(v) for k, v in r.items() if v != ""}
                key = (r["domain"], r["strategy"], r.get("seed"), r["round"])
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, "active.csv"), rows)
    write_csv(os.path.join(a.out, "summary.csv"), summarize(rows))
    write_csv(os.path.join(a.out, "paired.csv"), paired(rows, a.paired_ref))
    print("{} rows from {} runs -> {}".format(len(rows), len(a.runs), a.out))


if __name__ == "__main__":
    main()
