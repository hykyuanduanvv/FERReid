#!/bin/bash
# 10-08 night queue (task order: 2026-10-07_夜间6小时验证/任务令_6小时验证.md). GPUs 0-3, one worker each.
# Jobs: experiments/jobs_1008.txt (name|domain|strategy|budget_schedule|seed|extra), priority = file order, re-read
# before every claim (unclaimed lines may be reordered with an atomic mv). A job is claimed by mkdir of its output dir.
# Clock starts at the smoke test (experiments/START_TIME.txt, 17:06:27 NZDT): no claims after +5h30 (22:36:27 NZDT);
# at +6h (23:06:27 NZDT) running jobs are killed and marked CANCELLED.txt. Times in the logs: NZDT.
set -u
cd /data1/yangbin/dz/code/exp_1008_eps && source configs/local.sh
PY=/data1/yangbin/dz/venvs/ferreid/bin/python
CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=/data1/yangbin/dz/tmp/cache TORCH_HOME=/data1/yangbin/dz/tmp/torch HF_HOME=/data1/yangbin/dz/tmp/hf
COMMON="--fp16 True --report_to none --eval_num_workers 8 --n_seeds 1 --pseudo True --warm_start True --rounds 5 --eval_rounds all --steps 400 --oracle_all False --cannot_use none"
J=experiments/jobs_1008.txt
LOG=experiments/exit_codes_1008.txt
LAST_CLAIM=$(TZ=Pacific/Auckland date -d "2026-10-07 22:36:27" +%s)
HARD=$(TZ=Pacific/Auckland date -d "2026-10-07 23:06:27" +%s)
now() { TZ=Pacific/Auckland date +%T; }
gpu_free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 1500 ]; }

next_job() {  # first line of the job file whose output dir does not exist yet
  grep -v '^#' $J | while IFS='|' read -r name rest; do
    [ -z "$name" ] && continue
    [ -e "experiments/$name" ] || { echo "$name|$rest"; break; }
  done
}

worker() {
  local gpu=$1 line name dom strat sched seed extra out
  while true; do
    if [ "$(date +%s)" -gt "$LAST_CLAIM" ]; then echo "gpu $gpu stops claiming $(now)" >> $LOG; return; fi
    line=$(next_job)
    if [ -z "$line" ]; then echo "gpu $gpu: no jobs left $(now)" >> $LOG; return; fi
    IFS='|' read -r name dom strat sched seed extra <<< "$line"
    while ! gpu_free $gpu; do sleep 30; done
    out=experiments/$name
    mkdir "$out" 2>/dev/null || continue
    echo "$name start gpu $gpu $(now)" >> $LOG
    CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/eval_active.py --output_dir "$out" --checkpoint $CK/base_$dom/checkpoint-12000 \
      --domains $dom --strategies $strat --budget_schedule $sched --base_seed $seed $COMMON $extra > "$out/run.log" 2>&1 &
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
    echo "stopped at the 6-hour limit ($(now) NZDT, task order section 0); not finished" > "$d/CANCELLED.txt"
    echo "$(basename $d) CANCELLED (6h limit) $(now)" >> $LOG
  done
}

echo "queue start $(now)" >> $LOG
watchdog &
WD=$!
W=()
for g in 0 1 2 3; do worker $g & W+=($!); sleep 20; done
wait "${W[@]}"
kill $WD 2>/dev/null
echo "ALLDONE $(now)" >> $LOG
