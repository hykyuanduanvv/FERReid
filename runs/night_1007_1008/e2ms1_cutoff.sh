#!/bin/bash
# App 05:5x (v15): E2-MS seed 1 needs ~95 min (n_ms_e2_persist_r1_s2 took 87 min). Any of its jobs still unclaimed at
# 08:55 NZDT cannot finish by 10:30, so it is cancelled before start (complements the 08:30 dependency check).
cd /data1/yangbin/dz/code/exp_1008_night/experiments
t=$(TZ=Pacific/Auckland date -d "2026-10-08 08:55:00" +%s)
while [ "$(date +%s)" -lt "$t" ]; do sleep 30; done
for n in n_ms_e2_persist_r1_s1 n_ms_e2_random_r1_s1; do
  if mkdir $n 2>/dev/null; then
    echo "cancelled before start at 08:55 NZDT (App, v15): not claimed yet, a 20-round MSMT17 run (~95 min) could not finish by 10:30" > $n/CANCELLED.txt
    echo "$n CANCELLED (08:55 time cutoff) $(date -u -d +13hours +%T)" >> exit_codes_night.txt
  fi
done
