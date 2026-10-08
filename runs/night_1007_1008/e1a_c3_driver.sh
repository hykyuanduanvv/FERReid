#!/bin/bash
# E1a-C3 (task list v7): self-healing analysis of every CUHK03 no-question L20 run with per-round snapshots, as soon
# as the run is done (inference / clustering only, lowest-memory GPU of 0-3). Exits after the 5 runs.
cd /data1/yangbin/dz/code/exp_1008_night && source configs/local.sh
export OMP_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp PYTHONUNBUFFERED=1
E=/data1/yangbin/dz/experiments/exp_1008_eps; CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006/base_cuhk03/checkpoint-12000
O=experiments/e1a_c3; mkdir -p $O
RUNS="l20_c3_k15_none_s1:15 l20_c3_k15_none_s2:15 l20_c3_k15_none_s0:15 l20_c3_k30_none_s1:30 l20_c3_k30_none_s2:30"
while true; do
  left=0
  for rk in $RUNS; do
    r=${rk%:*}; k=${rk#*:}
    [ -f $O/${r}_summary.json ] && continue
    left=$((left+1)); [ -f $E/$r/.done ] || continue
    G=$(nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
    echo "$r start gpu $G $(date -u -d +13hours +%T)" >> $O/log.txt
    CUDA_VISIBLE_DEVICES=$G $PYTHON scripts/e1a_selfheal_1008.py --run_dir $E/$r --domain cuhk03 --k1 $k --rounds 20 \
      --checkpoint $CK --out_dir $O > $O/$r.log 2>&1
    echo "$r exit $? $(date -u -d +13hours +%T)" >> $O/log.txt
  done
  [ $left -eq 0 ] && { echo "ALLDONE $(date -u -d +13hours +%T)" >> $O/log.txt; break; }
  sleep 120
done
