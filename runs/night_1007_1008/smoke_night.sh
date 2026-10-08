#!/bin/bash
# CODE smoke (task list v3): shares one of GPUs 0-3 with L20 (lowest memory), runs three short checks in sequence.
cd /data1/yangbin/dz/code/exp_1008_night && source configs/local.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp PYTHONUNBUFFERED=1
CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006/base_cuhk03/checkpoint-12000
G=$(nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
C="--fp16 True --report_to none --eval_num_workers 8 --n_seeds 1 --pseudo True --warm_start True --rounds 2 --eval_rounds all --steps 400 --oracle_all False --cannot_use none --base_seed 0 --domains cuhk03 --checkpoint $CK"
L=experiments/smoke_night/log.txt; mkdir -p experiments/smoke_night
run() { n=$1; shift; echo "$n start gpu $G $(TZ=Pacific/Auckland date +%T)" >> $L; CUDA_VISIBLE_DEVICES=$G $PYTHON -u scripts/eval_active.py --output_dir experiments/smoke_night/$n $C "$@" > experiments/smoke_night/$n.log 2>&1; echo "$n exit $? $(TZ=Pacific/Auckland date +%T)" >> $L; }
run s1_rule_switch_off --strategies off:rule --budget_schedule 121,121
run s2_rule_switch_on --strategies off:rule --budget_schedule 121,121 --select_respect_cannot_use True
run s3_e2_random_k15 --strategies oracle_merge_subset --subset random --budget_schedule 100,0 --pseudo_k1 15 --pseudo_eps 0.6
echo ALLDONE >> $L
