"""E1d (task list v14, section 2c), offline. Fixed before the MSMT17 data: on the round-2 clustering, take the rule's
first 2B questions (CUHK03 B = 121, MSMT17 B = 298: one round's budget), re-rank them by recip_r2 from low to high
(least mutual 20-NN between the two units in round 2 first) and keep B. Count "same person and still apart at round
R" among the B kept: E1d vs the rule's own first B vs the ceiling min(B, positives in the 2B window).
The hypothesis came from CUHK03 (02:28), so only MSMT17 decides: E1d finds >= 20% more than the rule with every
available seed in the same direction = support; fewer = refute. CUHK03 is reported only.

  python scripts/e1d_window_1008.py <e1a out_dir> <B> <run> [<run> ...] --out <json>
"""
import argparse
import importlib.util
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("e1a_dir")
    ap.add_argument("B", type=int)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    e1b = _mod("e1b_predict_1008", os.path.join(here, "e1b_predict_1008.py"))
    e1c = _mod("e1c_rule_candidates_1008", os.path.join(here, "e1c_rule_candidates_1008.py"))
    res = {"B": o.B, "window": 2 * o.B, "round": 2, "runs": {}}
    for run in o.runs:
        rows = e1c.run_one(o.e1a_dir, run, 2 * o.B, e1b, rq=2)  # rows in rule order, at most 2B
        rule = sum(r["y"] for r in rows[:o.B])
        # recip_r2 is stored negated (higher = less mutual); low recip first = highest stored value first
        e1d = sum(r["y"] for r in sorted(rows, key=lambda r: -r["recip_r2"])[:o.B])
        pos = sum(r["y"] for r in rows)
        res["runs"][run] = {"window_size": len(rows), "same_person_in_window": int(sum(r["same"] for r in rows)),
                            "positives_in_window": int(pos), "rule_topB": int(rule), "e1d_topB": int(e1d),
                            "ceiling": int(min(o.B, pos)),
                            "e1d_vs_rule": (e1d - rule) / rule if rule else None}
        print(run, json.dumps(res["runs"][run]), flush=True)
    rs = list(res["runs"].values())
    tot_rule, tot_e1d = sum(r["rule_topB"] for r in rs), sum(r["e1d_topB"] for r in rs)
    signs = [r["e1d_topB"] - r["rule_topB"] for r in rs]
    res["total"] = {"rule_topB": tot_rule, "e1d_topB": tot_e1d, "ceiling": sum(r["ceiling"] for r in rs),
                    "relative": (tot_e1d - tot_rule) / tot_rule if tot_rule else None,
                    "per_seed_diff": signs, "same_direction": all(s > 0 for s in signs) or all(s < 0 for s in signs)}
    rel = res["total"]["relative"]
    res["call_if_msmt17"] = ("support" if rel is not None and rel >= 0.2 and all(s > 0 for s in signs)
                             else "refute" if rel is not None and rel < 0 else "undecidable")
    print("TOTAL", json.dumps(res["total"]), "call(if MSMT17):", res["call_if_msmt17"])
    json.dump(res, open(o.out, "w"), indent=1)


if __name__ == "__main__":
    main()
