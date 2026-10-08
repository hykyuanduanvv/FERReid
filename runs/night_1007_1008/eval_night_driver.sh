#!/bin/bash
# Dev-half evaluation (rounds 16-20) of the night runs as they finish: E2-C3 (n_c3_e2_*), Q1-MSMT17 (l20_ms_* run
# here, n_ms_k30_none_s1), E2-MS (n_ms_e2_*). Inference only, on the lowest-memory GPU of 0-3, and only when that
# GPU uses < 12 GB (never next to an MSMT17 training run, which takes ~23 GB). Runs until 10:30 NZDT.
cd /data1/yangbin/dz/code/exp_1008_night && source configs/local.sh
export OMP_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp PYTHONUNBUFFERED=1
E=experiments; O=experiments/eval_night; mkdir -p $O
CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006
END=$(TZ=Pacific/Auckland date -d "2026-10-08 10:30:00" +%s)
while [ "$(date +%s)" -lt "$END" ]; do
  for d in $E/n_c3_e2_* $E/l20_ms_* $E/n_ms_k30_none_s1 $E/n_ms_e2_*; do
    [ -d "$d" ] || continue
    n=$(basename $d); [ -f $d/.done ] || continue; [ -f $O/$n.csv ] && continue
    case $n in n_c3_*) dom=cuhk03 ;; *) dom=msmt17 ;; esac
    line=$(nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1)
    G=${line%%,*}; M=$(echo ${line#*,} | tr -d " ")
    [ "$M" -lt 12000 ] || { sleep 60; continue 2; }
    echo "$n start gpu $G $(date -u -d +13hours +%T)" >> $O/log.txt
    CUDA_VISIBLE_DEVICES=$G $PYTHON scripts/eval_rounds_1008.py --run_dir $d --domain $dom --checkpoint $CK/base_$dom/checkpoint-12000 \
      --manifest data_manifests/test_split_dev_final.json --out $O/$n.csv --rounds 16,17,18,19,20 > $O/$n.log 2>&1
    echo "$n exit $? $(date -u -d +13hours +%T)" >> $O/log.txt
  done
  sleep 120
done
