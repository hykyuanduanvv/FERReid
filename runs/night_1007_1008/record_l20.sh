#!/bin/bash
cd /data1/yangbin/dz/code/exp_1008_eps
while true; do
  /data1/yangbin/dz/venvs/ferreid/bin/python record_l20.py 2>> experiments/record_l20.err
  grep -q ALLDONE experiments/exit_codes_l20.txt 2>/dev/null && break
  sleep 600
done
