"""Read the validation curves of a plan stage and record the decisions the next stages use.

Score of a run = mean of its last 3 validation mAPs (val_mean_mAP, the held-out *source* domain;
never a target domain). Differences below ~1 mAP are within run-to-run noise at this scale
(two seeds of the same DINOv2 configuration differed by 0.4-0.8 mAP on the CUHK03 validation set).

  python scripts/pick_best.py stage1   # LOSS_ARGS := loss of the better VPT run  -> plans/chosen.sh
  python scripts/pick_best.py stage2   # convergence report of the 12k runs; STEPS stays 12000 unless
                                       # the curve is still clearly rising (then a warning)
  python scripts/pick_best.py stage3   # deltas vs the stage-1 reference; options with delta >= +1.0
                                       # are written to plans/final.sh (FINAL_ARGS) -- review it!
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
CHOSEN = ROOT / "plans" / "chosen.sh"
FINAL = ROOT / "plans" / "final.sh"
KEEP_DELTA = 1.0


def curve(name):
    path = EXP / name / "trainer_state.json"
    if not path.exists():
        return None
    hist = json.loads(path.read_text())["log_history"]
    # Final-step evaluation can be logged twice; score distinct evaluation steps.
    points = {h["step"]: h["val_mean_mAP"] for h in hist if "val_mean_mAP" in h}
    return sorted(points.items())


def score(name):
    c = curve(name)
    if not c:
        return None
    last = [m for _, m in c[-3:]]
    return sum(last) / len(last)


def write_chosen(**kv):
    lines = CHOSEN.read_text().splitlines() if CHOSEN.exists() else []
    lines = [l for l in lines if not any(l.startswith("export {}=".format(k)) for k in kv)]
    lines += ['export {}="{}"'.format(k, v) for k, v in kv.items()]
    CHOSEN.write_text("\n".join(lines) + "\n")
    print("wrote", CHOSEN.relative_to(ROOT), kv)


def table(names):
    print("{:<22} {:>9} {:>7} {:>9}".format("run", "last3 mAP", "best", "evals"))
    for n in names:
        c = curve(n)
        if not c:
            print("{:<22} {:>9}".format(n, "missing"))
            continue
        print("{:<22} {:>9.2f} {:>7.2f} {:>9}".format(n, score(n), max(m for _, m in c), len(c)))


def stage1():
    runs = ["s1_vpt_triplet", "s1_vpt_bot", "s1_plain_bot", "s1_plain_full"]
    table(runs)
    a, b = score("s1_vpt_triplet"), score("s1_vpt_bot")
    if a is None or b is None:
        sys.exit("both s1_vpt_triplet and s1_vpt_bot are needed")
    use_bot = b >= a
    print("\nVPT: BNNeck+CE {} triplet only ({:+.2f} mAP)".format(">=" if use_bot else "<", b - a))
    write_chosen(LOSS_ARGS="$LOSS_BOT" if use_bot else "$LOSS_TRIPLET",
                 LOSS_REF="s1_vpt_bot" if use_bot else "s1_vpt_triplet")
    full, lora = score("s1_plain_full"), score("s1_plain_bot")
    if full is not None and lora is not None:
        print("full fine-tuning vs LoRA (plain, BNNeck+CE): {:+.2f} mAP".format(full - lora))


def stage2():
    runs = ["s2_vpt_long", "s2_plain_full_long"]
    table(runs)
    for n in runs:
        c = curve(n)
        if not c:
            continue
        q = max(1, len(c) // 4)
        first, last = sum(m for _, m in c[-2 * q:-q]) / q, sum(m for _, m in c[-q:]) / q
        best_step = max(c, key=lambda x: x[1])[0]
        print("{}: last-quarter mean {:.2f} vs previous quarter {:.2f} ({:+.2f}); best at step {}".format(
            n, last, first, last - first, best_step))
        if last - first > KEEP_DELTA:
            print("  WARNING: still rising -- consider more steps (edit STEPS in plans/chosen.sh)")
    write_chosen(STEPS="12000", SCHEDULE="--lr_scheduler_type cosine --warmup_steps 500")


def stage3():
    ref = None
    if CHOSEN.exists():
        m = re.search(r'export LOSS_REF="([^"]+)"', CHOSEN.read_text())
        ref = m.group(1) if m else None
    if ref is None:
        sys.exit("run 'pick_best.py stage1' first (reference run unknown)")
    tasks = (ROOT / "plans" / "stage3_sweep.tasks").read_text().splitlines()
    options = {}
    for line in tasks:
        if "|" in line and not line.strip().startswith("#"):
            name, cmd = (s.strip() for s in line.split("|", 1))
            options[name] = cmd.split("$LOSS_ARGS", 1)[1].strip() if "$LOSS_ARGS" in cmd else ""
    base = score(ref)
    if base is None:
        sys.exit("reference run {} has no validation curve".format(ref))
    print("reference {}: {:.2f}\n".format(ref, base))
    print("{:<16} {:>9} {:>8}  options".format("run", "last3", "delta"))
    keep = []
    for name, opt in options.items():
        s = score(name)
        if s is None:
            print("{:<16} {:>9}".format(name, "missing"))
            continue
        flag = " <- keep" if s - base >= KEEP_DELTA else ""
        print("{:<16} {:>9.2f} {:>+8.2f}  {}{}".format(name, s, s - base, opt, flag))
        if flag:
            keep.append(opt)
    # options that change the same flag cannot be combined blindly: keep the first of each flag
    seen, final = set(), []
    for opt in keep:
        flags = set(re.findall(r"--\w+", opt))
        if flags & seen:
            print("  (not combined, conflicts with an earlier kept option):", opt)
            continue
        seen |= flags
        final.append(opt)
    FINAL.write_text('# written by scripts/pick_best.py stage3 -- review before stage 4.\n'
                     '# Each option improved the tuning-fold validation by >= {} mAP on its own;\n'
                     '# the combination itself was not validated.\nexport FINAL_ARGS="{}"\n'.format(KEEP_DELTA, " ".join(final)))
    print("\nwrote plans/final.sh: FINAL_ARGS=\"{}\"".format(" ".join(final)))


if __name__ == "__main__":
    {"stage1": stage1, "stage2": stage2, "stage3": stage3}[sys.argv[1] if len(sys.argv) > 1 else "stage1"]()
