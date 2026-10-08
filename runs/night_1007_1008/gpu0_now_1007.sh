#!/bin/bash
# One pass over the priority t_* jobs on GPU 0 (idle while the MSMT17 oracle runs on GPU 1); same claim rule as queue3.
set -u
cd /data1/yangbin/dz/code/exp_1007_loop && source configs/local.sh
PY=/data1/yangbin/dz/venvs/ferreid/bin/python; CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp
COMMON="--fp16 True --report_to none --eval_num_workers 8 --n_seeds 1 --pseudo True --warm_start True --rounds 5 --eval_rounds all --steps 400 --oracle_all False --cannot_use none"
grep "^t_" experiments/queue3_short.txt | while IFS="|" read -r name dom strat sched seed extra; do
  [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")" -lt 1500 ] || break
  mkdir "experiments/$name" 2>/dev/null || continue
  echo "$name start gpu 0 $(date +%T)" >> experiments/exit_codes_queue3.txt
  CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_active.py --output_dir experiments/$name --checkpoint $CK/base_$dom/checkpoint-12000 \
    --domains $dom --strategies $strat --budget_schedule $sched --base_seed $seed $COMMON $extra > experiments/$name/run.log 2>&1 && touch experiments/$name/.done
  echo "$name exit $? gpu 0 $(date +%T)" >> experiments/exit_codes_queue3.txt
done
