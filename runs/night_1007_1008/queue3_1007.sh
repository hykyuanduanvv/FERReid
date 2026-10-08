#!/bin/bash
# queue3 = queue2 with the "train longer" CUHK03 jobs first (t_*). Job lines may carry extra args in a 6th field. GPUs 2-5: after their current job, only short CUHK03 jobs that can end
# before the cutoff (09:00 CST = 14:00 NZDT, user: only two GPUs after that), claimed no later than cutoff - 45 min.
# GPUs 0-1: after run_oracle_1007.sh, every remaining job, no cutoff.
set -u
cd /data1/yangbin/dz/code/exp_1007_loop && source configs/local.sh
PY=/data1/yangbin/dz/venvs/ferreid/bin/python
CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp
COMMON="--fp16 True --report_to none --eval_num_workers 8 --n_seeds 1 --pseudo True --warm_start True --rounds 5 --eval_rounds all --steps 400 --oracle_all False --cannot_use none"
CUTOFF=$(date -d "2026-10-07 09:00:00" +%s); LAST_CLAIM=$((CUTOFF - 45*60))
# 1) stop queue_1007.sh workers from claiming anything new
while IFS='|' read -r name rest; do
  [ -z "$name" ] && continue
  mkdir "experiments/$name" 2>/dev/null && echo "superseded by queue3_1007.sh (r_ names)" > "experiments/$name/SUPERSEDED.txt"
done < experiments/queue_jobs.txt
SHORT=experiments/queue3_short.txt; LONG=experiments/queue3_long.txt
cat > $SHORT <<'J'
t_cuhk03_none_r15|cuhk03|none|0,0,0,0,0,0,0,0,0,0,0,0,0,0,0|0|--rounds 15
t_cuhk03_off_ac_r15|cuhk03|off:ac|121,121,121,121,121,0,0,0,0,0,0,0,0,0,0|0|--rounds 15
t_cuhk03_oracle_r15|cuhk03|oracle|0,0,0,0,0,0,0,0,0,0,0,0,0,0,0|0|--rounds 15
t_cuhk03_none_steps1000|cuhk03|none|0,0,0,0,0|0|--steps 1000
r_cuhk03_off_ac_even_s1|cuhk03|off:ac|121,121,121,121,121|1
r_cuhk03_off_rule_even_s1|cuhk03|off:rule|121,121,121,121,121|1
r_cuhk03_off_cos_even_s1|cuhk03|off:cos|121,121,121,121,121|1
r_cuhk03_none_s1|cuhk03|none|121,121,121,121,121|1
r_cuhk03_off_ac_front_s0|cuhk03|off:ac|605,0,0,0,0|0
r_cuhk03_off_rule_front_s0|cuhk03|off:rule|605,0,0,0,0|0
J
cat > $LONG <<'J'
t_cuhk03_none_r15|cuhk03|none|0,0,0,0,0,0,0,0,0,0,0,0,0,0,0|0|--rounds 15
t_cuhk03_off_ac_r15|cuhk03|off:ac|121,121,121,121,121,0,0,0,0,0,0,0,0,0,0|0|--rounds 15
t_cuhk03_oracle_r15|cuhk03|oracle|0,0,0,0,0,0,0,0,0,0,0,0,0,0,0|0|--rounds 15
t_cuhk03_none_steps1000|cuhk03|none|0,0,0,0,0|0|--steps 1000
r_cuhk03_off_ac_even_s1|cuhk03|off:ac|121,121,121,121,121|1
r_cuhk03_off_rule_even_s1|cuhk03|off:rule|121,121,121,121,121|1
r_cuhk03_off_cos_even_s1|cuhk03|off:cos|121,121,121,121,121|1
r_cuhk03_none_s1|cuhk03|none|121,121,121,121,121|1
r_cuhk03_off_ac_front_s0|cuhk03|off:ac|605,0,0,0,0|0
r_cuhk03_off_rule_front_s0|cuhk03|off:rule|605,0,0,0,0|0
r_msmt17_off_ac_front_s0|msmt17|off:ac|1490,0,0,0,0|0
r_msmt17_off_rule_front_s0|msmt17|off:rule|1490,0,0,0,0|0
r_market1501_off_ac_even_s0|market1501|off:ac|108,108,108,108,108|0
r_market1501_off_rule_even_s0|market1501|off:rule|108,108,108,108,108|0
r_market1501_off_cos_even_s0|market1501|off:cos|108,108,108,108,108|0
r_market1501_none_s0|market1501|none|108,108,108,108,108|0
r_market1501_oracle_s0|market1501|oracle|108,108,108,108,108|0
r_msmt17_off_ac_even_s1|msmt17|off:ac|298,298,298,298,298|1
r_msmt17_off_rule_even_s1|msmt17|off:rule|298,298,298,298,298|1
r_msmt17_off_cos_even_s1|msmt17|off:cos|298,298,298,298,298|1
r_msmt17_none_s1|msmt17|none|298,298,298,298,298|1
J
gpu_free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 1500 ]; }
worker() { # gpu jobs-file deadline(0 = none)
  local gpu=$1 file=$2 last=$3 name dom strat sched seed out
  while IFS='|' read -r name dom strat sched seed extra; do
    [ -z "$name" ] && continue
    [ -e "experiments/$name" ] && continue
    while ! gpu_free $gpu; do sleep 30; done
    if [ "$last" -gt 0 ] && [ "$(date +%s)" -gt "$last" ]; then echo "gpu $gpu stops claiming $(date +%T)" >> experiments/exit_codes_queue3.txt; return; fi
    out=experiments/$name
    mkdir "$out" 2>/dev/null || continue
    echo "$name start gpu $gpu $(date +%T)" >> experiments/exit_codes_queue3.txt
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/eval_active.py --output_dir "$out" --checkpoint $CK/base_$dom/checkpoint-12000 \
      --domains $dom --strategies $strat --budget_schedule $sched --base_seed $seed $COMMON $extra > "$out/run.log" 2>&1 && touch "$out/.done"
    echo "$name exit $? gpu $gpu $(date +%T)" >> experiments/exit_codes_queue3.txt
  done < $file
}
for g in 2 3 4 5; do worker $g $SHORT $LAST_CLAIM & done
for g in 0 1; do ( while ! grep -q ALLDONE experiments/exit_codes_oracle.txt 2>/dev/null; do sleep 60; done; worker $g $LONG 0 ) & done
wait
echo ALLDONE >> experiments/exit_codes_queue3.txt
