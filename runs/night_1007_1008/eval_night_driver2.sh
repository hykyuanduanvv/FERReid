#!/bin/bash
# Dev-half evaluation (rounds 16-20; 5-round runs: their last round) of the night runs as they finish. Inference only
# (< 8 GB: query cached ~1.1 GB, gallery streamed from host). App 05:45: never next to an MSMT17 training run or on
# an idle GPU (the queue may claim it at once, cf. the 03:48 OOM). So it only uses a GPU whose current queue job is a
# CUHK03 one (from exit_codes_night.txt) and that has < 12 GB in use. Runs until 10:30 NZDT.
cd /data1/yangbin/dz/code/exp_1008_night && source configs/local.sh
export OMP_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp PYTHONUNBUFFERED=1
E=experiments; O=experiments/eval_night; mkdir -p $O
L=$E/exit_codes_night.txt
CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006
END=$(TZ=Pacific/Auckland date -d "2026-10-08 10:30:00" +%s)
cuhk_gpu() {  # a GPU whose last started queue job is CUHK03 and still running, < 12 GB used
  for g in 0 1 2 3; do
    job=$(grep -E " start gpu $g " $L | tail -1 | awk '{print $1}')
    [ -n "$job" ] || continue
    grep -qE "^$job exit " $L && continue
    case $job in n_c3_*|n_bug_c3_*) ;; *) continue ;; esac
    m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "$m" -gt 3000 ] && [ "$m" -lt 12000 ] && { echo $g; return; }
  done
}
while [ "$(date +%s)" -lt "$END" ]; do
  for d in $E/n_c3_e2_* $E/l20_ms_* $E/n_ms_k30_none_s1 $E/n_ms_e2_* $E/n_bug_c3_r20_* $E/n_bug_ms_* $E/n_ms_k40_rule_r20_* $E/n_c3_k*_r20_*; do
    [ -d "$d" ] || continue
    n=$(basename $d); [ -f $d/.done ] || continue; [ -f $O/$n.csv ] && continue
    case $n in n_c3_*|n_bug_c3_*) dom=cuhk03 ;; *) dom=msmt17 ;; esac
    case $n in n_bug_ms_*) R=5 ;; *) R=16,17,18,19,20 ;; esac
    G=$(cuhk_gpu)
    [ -n "$G" ] || { sleep 60; continue 2; }
    echo "$n start gpu $G $(date -u -d +13hours +%T)" >> $O/log.txt
    CUDA_VISIBLE_DEVICES=$G $PYTHON scripts/eval_rounds_1008.py --run_dir $d --domain $dom --checkpoint $CK/base_$dom/checkpoint-12000 \
      --manifest data_manifests/test_split_dev_final.json --out $O/$n.csv --rounds $R > $O/$n.log 2>&1
    echo "$n exit $? $(date -u -d +13hours +%T)" >> $O/log.txt
  done
  sleep 120
done
