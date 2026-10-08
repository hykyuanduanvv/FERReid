#!/bin/bash
# Queued after run_loop_1007.sh: full-label upper bound (strategy oracle: true identities every round), 5 rounds,
# same training settings, GPU 0 = cuhk03, GPU 1 = msmt17.
set -u
cd /data1/yangbin/dz/code/exp_1007_loop && source configs/local.sh
while ! grep -q ALLDONE experiments/exit_codes.txt 2>/dev/null; do sleep 60; done
PY=/data1/yangbin/dz/venvs/ferreid/bin/python
CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp
COMMON="--fp16 True --report_to none --eval_num_workers 8 --n_seeds 1 --base_seed 0 --pseudo True --warm_start True --rounds 5 --eval_rounds all --steps 400 --oracle_all False --cannot_use none"
job() { # gpu domain
  local out=experiments/loop_$2_oracle
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/eval_active.py --output_dir "$out" --checkpoint $CK/base_$2/checkpoint-12000 \
    --domains $2 --strategies oracle $COMMON > "$out/run.log" 2>&1 && touch "$out/.done"
  echo "loop_$2_oracle exit $? $(date +%T)" >> experiments/exit_codes_oracle.txt
}
job 0 cuhk03 & job 1 msmt17 & wait
echo ALLDONE >> experiments/exit_codes_oracle.txt
