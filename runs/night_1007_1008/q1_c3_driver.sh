#!/bin/bash
# Q1-C3 (task list v6): dev-half per-round evaluation of every finished CUHK03 L20 group (inference only, shares the
# lowest-memory GPU of 0-3 with L20). Exits only when all 13 CUHK03 groups (jobs_l20.txt) have an eval csv.
cd /data1/yangbin/dz/code/exp_1008_night && source configs/local.sh
export OMP_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp PYTHONUNBUFFERED=1
E=/data1/yangbin/dz/experiments/exp_1008_eps; CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006/base_cuhk03/checkpoint-12000
O=experiments/q1_c3; mkdir -p $O
while true; do
  left=0
  for d in $E/l20_c3_*; do
    n=$(basename $d); [ -f $O/$n.csv ] && continue
    left=$((left+1)); [ -f $d/.done ] || continue
    G=$(nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
    echo "$n start gpu $G $(date -u -d +13hours +%T)" >> $O/log.txt
    CUDA_VISIBLE_DEVICES=$G $PYTHON scripts/eval_rounds_1008.py --run_dir $d --domain cuhk03 --checkpoint $CK \
      --manifest data_manifests/test_split_dev_final.json --out $O/$n.csv > $O/$n.log 2>&1
    echo "$n exit $? $(date -u -d +13hours +%T)" >> $O/log.txt
  done
  [ $left -eq 0 ] && [ $(ls $O/l20_c3_*.csv 2>/dev/null | wc -l) -ge 13 ] && { echo "ALLDONE $(date -u -d +13hours +%T)" >> $O/log.txt; break; }
  sleep 300
done
