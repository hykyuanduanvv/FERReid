#!/bin/bash
# queue4: priority = leftover t_* (train longer) -> oracle decomposition (o_*) -> random questions x 3 "no" usages (p_*)
# -> the rest of queue3 (u_* = unclaimed queue3 jobs, renamed). GPUs 2-5 claim until 08:20 CST (13:20 NZDT),
# GPUs 0-1 without cutoff. Unclaimed queue3 jobs are claimed here first so queue3's workers find nothing left.
set -u
cd /data1/yangbin/dz/code/exp_1007_loop && source configs/local.sh
PY=/data1/yangbin/dz/venvs/ferreid/bin/python
CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp
COMMON="--fp16 True --report_to none --eval_num_workers 8 --n_seeds 1 --pseudo True --warm_start True --rounds 5 --eval_rounds all --steps 400 --oracle_all False --cannot_use none"
LAST_CLAIM=$(date -d "2026-10-07 08:20:00" +%s)
J=experiments/queue4_jobs.txt; : > $J.left_t; : > $J.left_rest
while IFS='|' read -r name rest; do
  [ -z "$name" ] && continue
  if mkdir "experiments/$name" 2>/dev/null; then
    echo "superseded by queue4_1007.sh (u_$name)" > "experiments/$name/SUPERSEDED.txt"
    case $name in t_*) echo "u_$name|$rest" >> $J.left_t ;; *) echo "u_$name|$rest" >> $J.left_rest ;; esac
  fi
done < experiments/queue3_long.txt
{
cat $J.left_t
cat <<'L'
o_cuhk03_oracle_merge|cuhk03|oracle_merge|0,0,0,0,0|0|
o_cuhk03_oracle_purify|cuhk03|oracle_purify|0,0,0,0,0|0|
o_msmt17_oracle_merge|msmt17|oracle_merge|0,0,0,0,0|0|
o_msmt17_oracle_purify|msmt17|oracle_purify|0,0,0,0,0|0|
p_cuhk03_random_yesonly|cuhk03|random|121,121,121,121,121|0|--cannot_use none
p_cuhk03_random_no_cluster|cuhk03|random|121,121,121,121,121|0|--cannot_use cluster
p_cuhk03_random_no_train|cuhk03|random|121,121,121,121,121|0|--cannot_use train
p_msmt17_random_yesonly|msmt17|random|298,298,298,298,298|0|--cannot_use none
p_msmt17_random_no_cluster|msmt17|random|298,298,298,298,298|0|--cannot_use cluster
p_msmt17_random_no_train|msmt17|random|298,298,298,298,298|0|--cannot_use train
L
cat $J.left_rest
} > $J
gpu_free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 1500 ]; }
worker() { # gpu deadline(0 = none)
  local gpu=$1 last=$2 name dom strat sched seed extra out
  while IFS='|' read -r name dom strat sched seed extra; do
    [ -z "$name" ] && continue
    [ -e "experiments/$name" ] && continue
    while ! gpu_free $gpu; do sleep 30; done
    if [ "$last" -gt 0 ] && [ "$(date +%s)" -gt "$last" ]; then echo "gpu $gpu stops claiming $(date +%T)" >> experiments/exit_codes_queue4.txt; return; fi
    out=experiments/$name
    mkdir "$out" 2>/dev/null || continue
    echo "$name start gpu $gpu $(date +%T)" >> experiments/exit_codes_queue4.txt
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/eval_active.py --output_dir "$out" --checkpoint $CK/base_$dom/checkpoint-12000 \
      --domains $dom --strategies $strat --budget_schedule $sched --base_seed $seed $COMMON $extra > "$out/run.log" 2>&1 && touch "$out/.done"
    echo "$name exit $? gpu $gpu $(date +%T)" >> experiments/exit_codes_queue4.txt
  done < $J
}
for g in 2 3 4 5; do worker $g $LAST_CLAIM & done
for g in 0 1; do worker $g 0 & done
wait
echo ALLDONE >> experiments/exit_codes_queue4.txt
