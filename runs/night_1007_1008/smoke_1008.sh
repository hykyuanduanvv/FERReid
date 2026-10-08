#!/bin/bash
cd /data1/yangbin/dz/code/exp_1008_eps && source configs/local.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp PYTHONUNBUFFERED=1
CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006
out=experiments/smoke_none_c3; mkdir "$out" || exit 1
echo "smoke start $(TZ=Pacific/Auckland date +%F_%T)" > experiments/START_TIME.txt
CUDA_VISIBLE_DEVICES=0 $PYTHON -u scripts/eval_active.py --output_dir $out --checkpoint $CK/base_cuhk03/checkpoint-12000 \
  --domains cuhk03 --strategies none --budget_schedule 0 --base_seed 0 \
  --fp16 True --report_to none --eval_num_workers 8 --n_seeds 1 --pseudo True --warm_start True --rounds 1 --eval_rounds all --steps 400 --oracle_all False --cannot_use none \
  > $out/run.log 2>&1 && touch $out/.done
echo "smoke exit $? $(TZ=Pacific/Auckland date +%T)" >> experiments/START_TIME.txt
