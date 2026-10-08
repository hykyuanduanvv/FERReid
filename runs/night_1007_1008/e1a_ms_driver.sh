#!/bin/bash
# E1a-MS (task list v7 priority 4): self-healing analysis of the MSMT17 no-question runs with per-round snapshots
# (l20_ms_k30_none_s2, n_ms_k30_none_s1; both run in exp_1008_night), as soon as each is done. GPU inference.
cd /data1/yangbin/dz/code/exp_1008_night && source configs/local.sh
export OMP_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp PYTHONUNBUFFERED=1
E=/data1/yangbin/dz/code/exp_1008_night/experiments; CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006/base_msmt17/checkpoint-12000
O=experiments/e1a_ms; mkdir -p $O
RUNS="l20_ms_k30_none_s2 n_ms_k30_none_s1"
while true; do
  left=0
  for r in $RUNS; do
    [ -f $O/${r}_summary.json ] && continue
    left=$((left+1)); [ -f $E/$r/.done ] || continue
    G=$(nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
    echo "$r start gpu $G $(date -u -d +13hours +%T)" >> $O/log.txt
    CUDA_VISIBLE_DEVICES=$G $PYTHON scripts/e1a_selfheal_1008.py --run_dir $E/$r --domain msmt17 --k1 30 --rounds 20 \
      --checkpoint $CK --out_dir $O > $O/$r.log 2>&1
    echo "$r exit $? $(date -u -d +13hours +%T)" >> $O/log.txt
  done
  [ $left -eq 0 ] && { echo "ALLDONE $(date -u -d +13hours +%T)" >> $O/log.txt; break; }
  sleep 120
done
