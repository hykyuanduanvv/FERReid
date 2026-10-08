#!/bin/bash
# writes experiments/PROGRESS.md now and every hour until the queue log has ALLDONE (NZDT times)
cd /data1/yangbin/dz/code/exp_1008_eps
PY=/data1/yangbin/dz/venvs/ferreid/bin/python
while true; do
  {
  echo "# PROGRESS exp_1008_eps — $(TZ=Pacific/Auckland date "+%F %T") NZDT"
  echo; echo "Clock start 17:06:27; no claims after 22:36:27; hard stop 23:06:27 (NZDT)."; echo
  echo "## GPUs"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -4 | sed "s/^/- /"
  echo; echo "## Running"; for p in experiments/*/.pid; do [ -e "$p" ] && echo "- $(basename $(dirname $p)): $(grep -c "mAP" $(dirname $p)/run.log 2>/dev/null) rounds logged"; done
  echo; echo "## Counts"
  echo "- jobs: $(grep -vc "^#" experiments/jobs_1008.txt); done: $(ls experiments/*/.done 2>/dev/null | wc -l); failed: $(grep -c "exit [1-9]" experiments/exit_codes_1008.txt); cancelled: $(ls experiments/*/CANCELLED.txt 2>/dev/null | wc -l)"
  echo; echo "## Finished (last round)"; echo
  $PY - <<"PYEOF"
import csv, glob
print("| job | strategy | seed | R | mAP | R1 | purity | pw P/R | split_ids | member q/no | merge q/yes |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for d in sorted(glob.glob("experiments/*/.done")):
    d = d[:-6]
    try:
        rows = [r for r in csv.DictReader(open(d + "/active.csv")) if r.get("round") not in ("", None)]
    except Exception:
        continue
    L = rows[-1]
    f = lambda k, p=1: ("{:.%df}" % p).format(float(L[k])) if L.get(k) not in (None, "", "nan") else "-"
    s = lambda k: sum(int(float(r[k])) for r in rows if r.get(k) not in (None, "", "nan"))
    mq = "{}/{}".format(s("n_member_q"), s("n_member_no")) if "n_member_q" in L else "-"
    gq = "{}/{}".format(s("n_merge_q"), s("n_merge_yes")) if "n_merge_q" in L else "-"
    print("| {} | {} | {} | {} | {} | {} | {} | {}/{} | {} | {} | {} |".format(d.split("/")[-1], L["strategy"], L["seed"], L["round"],
          f("mAP"), f("rank1"), f("purity", 2), f("pw_prec", 2), f("pw_rec", 2), L.get("split_ids", "-"), mq, gq))
PYEOF
  echo; echo "## Failures"; grep "exit [1-9]" experiments/exit_codes_1008.txt | sed "s/^/- /"
  } > experiments/PROGRESS.md.tmp && mv experiments/PROGRESS.md.tmp experiments/PROGRESS.md
  grep -q ALLDONE experiments/exit_codes_1008.txt 2>/dev/null && break
  sleep 3600
done
