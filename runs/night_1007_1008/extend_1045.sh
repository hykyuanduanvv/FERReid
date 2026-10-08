#!/bin/bash
# Post-window extension (task list v16; user 10:38): inference only, at most 2 of GPUs 0-3 (GPU 1 and GPU 2; GPU 0
# holds another user's process), everything stopped at 11:45 NZDT. Lane A on GPU 1, lane B on GPU 2, each serial.
cd /data1/yangbin/dz/code/exp_1008_night && source configs/local.sh
export OMP_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp PYTHONUNBUFFERED=1
E=experiments; O=experiments/eval_night; LOG=experiments/extend_1045.log
CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006
EPS=/data1/yangbin/dz/experiments/exp_1008_eps
END=$(TZ=Pacific/Auckland date -d "2026-10-08 11:45:00" +%s)
now() { date -u -d +13hours +%T; }
ev() {  # gpu run_dir domain rounds out_name
  [ "$(date +%s)" -lt "$END" ] || { echo "$5 SKIPPED (11:45 limit) $(now)" >> $LOG; return; }
  echo "$5 start gpu $1 $(now)" >> $LOG
  CUDA_VISIBLE_DEVICES=$1 timeout $(( END - $(date +%s) )) $PYTHON scripts/eval_rounds_1008.py --run_dir $2 --domain $3 \
    --checkpoint $CK/base_$3/checkpoint-12000 --manifest data_manifests/test_split_dev_final.json --out $O/$5.csv --rounds $4 > $O/$5.log 2>&1
  echo "$5 exit $? $(now)" >> $LOG
}
laneA() {
  ev 1 $E/n_ms_k30_none_s1 msmt17 16,17,18,19,20 n_ms_k30_none_s1
  ev 1 $E/n_ms_k40_rule_r20_s2 msmt17 16,17,18,19,20 n_ms_k40_rule_r20_s2
}
laneB() {
  for n in n_c3_k10_none_r20_s1 n_c3_k10_none_r20_s2 n_c3_k20_none_r20_s1 n_c3_k20_none_r20_s2 n_c3_k20_rule_r20_s1 n_c3_k20_rule_r20_s2; do
    ev 2 $E/$n cuhk03 16,17,18,19,20 $n
  done
  # App 10:5x: E1a-MS seed 1 (> 8 GB, alone on GPU 2) before the BUG-MS baseline
  if [ "$(date +%s)" -lt "$END" ]; then
    echo "e1a_ms n_ms_k30_none_s1 start gpu 2 $(now)" >> $LOG
    CUDA_VISIBLE_DEVICES=2 timeout $(( END - $(date +%s) )) $PYTHON scripts/e1a_selfheal_1008.py --run_dir $E/n_ms_k30_none_s1 --domain msmt17 --k1 30 \
      --rounds 20 --checkpoint $CK/base_msmt17/checkpoint-12000 --out_dir $E/e1a_ms > $E/e1a_ms/n_ms_k30_none_s1.log 2>&1
    echo "e1a_ms n_ms_k30_none_s1 exit $? $(now)" >> $LOG
  else
    echo "e1a_ms n_ms_k30_none_s1 SKIPPED (11:45 limit) $(now)" >> $LOG
  fi
  for s in 0 1 2; do ev 2 $EPS/e_ms_eps000_s$s msmt17 5 e_ms_eps000_s$s; done
}
echo "extension start $(now)" >> $LOG
laneA & A=$!
laneB & B=$!
wait $A $B
echo "ALLDONE $(now)" >> $LOG
