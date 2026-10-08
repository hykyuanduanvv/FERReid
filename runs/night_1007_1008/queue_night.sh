#!/bin/bash
# Second phase of the 10-08 night (task list v7; v11 GPU routing): GPUs 0-3, one worker each. A GPU's worker starts only after L20's
# worker on that GPU has run out of jobs ("gpu N: no jobs left" in exp_1008_eps/experiments/exit_codes_l20.txt).
# Jobs: experiments/jobs_night.txt  name|domain|strategy|budget_schedule|seed|rounds|needs_file|extra|gpus
#   priority = file order, re-read before every claim. gpus (e.g. "0,1"): only those GPUs may claim the job
#   (task list v11 routing; empty = any). A job whose needs_file is missing is passed over only in favour of a
#   later job that also has a needs_file (same tier); a job without dependencies is never claimed past a
#   waiting dependent job.
# No claims after 10:00 NZDT; at 10:30 NZDT running jobs are stopped and marked CANCELLED.txt ("window end").
set -u
cd /data1/yangbin/dz/code/exp_1008_night && source configs/local.sh
PY=/data1/yangbin/dz/venvs/ferreid/bin/python
CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp PYTHONUNBUFFERED=1
COMMON="--fp16 True --report_to none --eval_num_workers 8 --n_seeds 1 --pseudo True --warm_start True --eval_rounds all --steps 400 --oracle_all False --cannot_use none"
J=experiments/jobs_night.txt
LOG=experiments/exit_codes_night.txt
L20LOG=/data1/yangbin/dz/code/exp_1008_eps/experiments/exit_codes_l20.txt
LAST_CLAIM=$(TZ=Pacific/Auckland date -d "2026-10-08 10:00:00" +%s)
HARD=$(TZ=Pacific/Auckland date -d "2026-10-08 10:30:00" +%s)
now() { TZ=Pacific/Auckland date +%T; }
gpu_free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 1500 ]; }

next_job() {  # gpu -> first claimable job line, or nothing
  local g=$1 waiting=0
  while IFS="|" read -r name dom strat sched seed rounds needs extra gpus; do
    [ -z "$name" ] && continue
    [ -e "experiments/$name" ] && continue
    if [ -n "$gpus" ] && ! echo ",$gpus," | grep -q ",$g,"; then continue; fi
    if [ -n "$needs" ] && [ ! -e "$needs" ]; then waiting=1; continue; fi
    if [ -z "$needs" ] && [ $waiting -eq 1 ]; then return; fi
    echo "$name|$dom|$strat|$sched|$seed|$rounds|$needs|$extra"; return
  done < <(grep -v "^#" $J)
}

worker() {
  local gpu=$1 line name dom strat sched seed rounds needs extra out
  until grep -q "gpu $gpu: no jobs left" $L20LOG 2>/dev/null; do sleep 60; done
  echo "gpu $gpu released by L20, worker starts $(now)" >> $LOG
  while true; do
    if [ "$(date +%s)" -gt "$LAST_CLAIM" ]; then echo "gpu $gpu stops claiming (10:00) $(now)" >> $LOG; return; fi
    line=$(next_job $gpu)
    if [ -z "$line" ]; then
      # nothing claimable now: stop when no job is left at all, else wait for a dependency
      left=$(grep -v "^#" $J | while IFS="|" read -r n d s b e r nf x gp; do [ -n "$n" ] && [ ! -e "experiments/$n" ] && { [ -z "$gp" ] || echo ",$gp," | grep -q ",$gpu,"; } && echo x; done | wc -l)
      [ "$left" -eq 0 ] && { echo "gpu $gpu: no jobs left $(now)" >> $LOG; return; }
      sleep 60; continue
    fi
    IFS='|' read -r name dom strat sched seed rounds needs extra <<< "$line"
    while ! gpu_free $gpu; do sleep 30; done
    out=experiments/$name
    mkdir "$out" 2>/dev/null || continue
    echo "$name start gpu $gpu $(now)" >> $LOG
    CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/eval_active.py --output_dir "$out" --checkpoint $CK/base_$dom/checkpoint-12000 \
      --domains $dom --strategies $strat --budget_schedule $sched --base_seed $seed --rounds $rounds $COMMON $extra > "$out/run.log" 2>&1 &
    echo $! > "$out/.pid"
    wait $!
    rc=$?
    [ $rc -eq 0 ] && touch "$out/.done"
    rm -f "$out/.pid"
    echo "$name exit $rc gpu $gpu $(now)" >> $LOG
  done
}

watchdog() {
  while [ "$(date +%s)" -lt "$HARD" ]; do sleep 30; done
  for p in experiments/*/.pid; do
    [ -e "$p" ] || continue
    d=$(dirname "$p")
    kill "$(cat "$p")" 2>/dev/null
    echo "stopped at the window end (10:30 NZDT, task list v7 section 0); not finished" > "$d/CANCELLED.txt"
    echo "$(basename $d) CANCELLED (window end) $(now)" >> $LOG
  done
}

echo "queue start $(now)" >> $LOG
watchdog &
WD=$!
W=()
for g in 0 1 2 3; do worker $g & W+=($!); done
wait "${W[@]}"
kill $WD 2>/dev/null
echo "ALLDONE $(now)" >> $LOG
