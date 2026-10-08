#!/bin/bash
# App 05:45 [urgent]: (a) after the R-MS rerun (n_ms_k30_none_s1, GPU 0) GPU 0 is held (hold_e1a_ms_gpu0) and runs
# E1a-MS of that run alone (~12 GB, never next to training), then the hold is released; (b) at 08:30 NZDT, if the
# E1a-MS persist list of seed 1 is not there, n_ms_e2_{persist,random}_r1_s1 are cancelled before start.
cd /data1/yangbin/dz/code/exp_1008_night && source configs/local.sh
export OMP_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp PYTHONUNBUFFERED=1
L=experiments/exit_codes_night.txt
O=experiments/e1a_ms
PF=$O/n_ms_k30_none_s1_persist.json
now() { date -u -d +13hours +%T; }
cutoff() {  # (b), in the background
  local t=$(TZ=Pacific/Auckland date -d "2026-10-08 08:30:00" +%s)
  while [ "$(date +%s)" -lt "$t" ]; do sleep 30; done
  [ -f $PF ] && return
  for n in n_ms_e2_persist_r1_s1 n_ms_e2_random_r1_s1; do
    if mkdir experiments/$n 2>/dev/null; then
      echo "cancelled before start at 08:30 NZDT (App 05:45): the E1a-MS persist list of n_ms_k30_none_s1 was not ready" > experiments/$n/CANCELLED.txt
      echo "$n CANCELLED (08:30 dependency cutoff) $(now)" >> $L
    fi
  done
}
cutoff &
# (a)
until grep -qE "^n_ms_k30_none_s1 exit [0-9]+ gpu 0" $L; do sleep 20; done
rc=$(grep -E "^n_ms_k30_none_s1 exit [0-9]+ gpu 0" $L | tail -1 | awk '{print $3}')
if [ "$rc" = "0" ]; then
  until [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 1500 ]; do sleep 10; done
  echo "n_ms_k30_none_s1 start gpu 0 (alone, hold) $(now)" >> $O/log.txt
  CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/e1a_selfheal_1008.py --run_dir experiments/n_ms_k30_none_s1 --domain msmt17 --k1 30 --rounds 20 \
    --checkpoint /data1/yangbin/dz/experiments/exp_1006_2_20261006/base_msmt17/checkpoint-12000 --out_dir $O > $O/n_ms_k30_none_s1.log 2>&1
  echo "n_ms_k30_none_s1 exit $? $(now)" >> $O/log.txt
else
  echo "R-MS rerun failed again (exit $rc), no E1a-MS $(now)" >> $O/log.txt
fi
mkdir -p experiments/hold_e1a_ms_gpu0 && echo "released $(now)" > experiments/hold_e1a_ms_gpu0/RELEASED.txt
wait
