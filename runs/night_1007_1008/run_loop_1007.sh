#!/bin/bash
# 5-round oracle loop, yes answers only, per-round budget = C/5 (C = round-0 pseudo clusters), seed 0, GPUs 0-1.
set -u
cd /data1/yangbin/dz/code/exp_1007_loop && source configs/local.sh
PY=/data1/yangbin/dz/venvs/ferreid/bin/python
CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp
COMMON="--fp16 True --report_to none --eval_num_workers 8 --n_seeds 1 --base_seed 0 --pseudo True --warm_start True --rounds 5 --eval_rounds all --steps 400 --oracle_all False --cannot_use none"
job() { # gpu domain strategy schedule
  local name=loop_$2_${3/:/_}
  local out=experiments/$name
  if [ -e "$out/.done" ]; then echo "skip $name"; return; fi
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/eval_active.py --output_dir "$out" --checkpoint $CK/base_$2/checkpoint-12000 \
    --domains $2 --strategies $3 --budget_schedule $4 $COMMON > "$out/run.log" 2>&1 && touch "$out/.done"
  echo "$name exit $? $(date +%T)" >> experiments/exit_codes.txt
}
C3=121,121,121,121,121; M=298,298,298,298,298
( job 0 cuhk03 off:ac $C3; job 0 cuhk03 none $C3; job 0 msmt17 off:ac $M; job 0 msmt17 none $M ) &
( job 1 cuhk03 off:rule $C3; job 1 cuhk03 off:cos $C3; job 1 msmt17 off:rule $M; job 1 msmt17 off:cos $M ) &
wait
echo ALLDONE >> experiments/exit_codes.txt
