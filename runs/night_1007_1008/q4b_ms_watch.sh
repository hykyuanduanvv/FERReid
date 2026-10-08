#!/bin/bash
# Q4b-MSMT17 (task list v6 deployment protocol; App 02:25: alone on a route-A GPU, not next to training): waits for
# GPU 1 to finish l20_ms_k30_rule_s1 (its worker then pauses on hold_q4b_ms), runs Q4b on GPU 1, then releases the hold.
cd /data1/yangbin/dz/code/exp_1008_night
L=experiments/exit_codes_night.txt
until grep -q "l20_ms_k30_rule_s1 exit" $L; do sleep 20; done
until [ "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")" -lt 1500 ]; do sleep 10; done
echo "q4b_ms start gpu 1 $(date -u -d +13hours +%T)" >> experiments/q4/log.txt
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp /data1/yangbin/dz/venvs/ferreid/bin/python -u scripts/q4_probe_k1_1008.py \
  /data1/yangbin/dz/experiments/exp_1006_2_20261006/round0/msmt17/msmt17_epoch0.npz experiments/q4/q4b_deploy_msmt17.json \
  --device cuda --mode deploy --grid 15,20,25,30,40,50 > experiments/q4/q4b_deploy_msmt17.log 2>&1
echo "q4b_ms exit $? $(date -u -d +13hours +%T)" >> experiments/q4/log.txt
mkdir -p experiments/hold_q4b_ms && echo "released $(date -u -d +13hours +%T)" > experiments/hold_q4b_ms/RELEASED.txt
