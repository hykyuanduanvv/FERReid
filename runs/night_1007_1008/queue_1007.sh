#!/bin/bash
# Shared job queue for GPUs 0-5. Workers on GPUs 2-5 start now; workers on 0-1 join after the queued oracle runs.
# A job is claimed atomically by mkdir of its output directory; finished jobs get .done.
set -u
cd /data1/yangbin/dz/code/exp_1007_loop && source configs/local.sh
PY=/data1/yangbin/dz/venvs/ferreid/bin/python
CK=/data1/yangbin/dz/experiments/exp_1006_2_20261006
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 FERREID_CPU_THREADS=8 TMPDIR=/data1/yangbin/dz/tmp
COMMON="--fp16 True --report_to none --eval_num_workers 8 --n_seeds 1 --pseudo True --warm_start True --rounds 5 --eval_rounds all --steps 400 --oracle_all False --cannot_use none"
JOBS=experiments/queue_jobs.txt
# name | domain | strategy | budget schedule | seed      (priority order)
cat > $JOBS <<'J'
q_msmt17_off_ac_even_s0|msmt17|off:ac|298,298,298,298,298|0
q_msmt17_off_rule_even_s0|msmt17|off:rule|298,298,298,298,298|0
q_msmt17_off_cos_even_s0|msmt17|off:cos|298,298,298,298,298|0
q_msmt17_none_s0|msmt17|none|298,298,298,298,298|0
q_cuhk03_off_ac_even_s1|cuhk03|off:ac|121,121,121,121,121|1
q_cuhk03_off_rule_even_s1|cuhk03|off:rule|121,121,121,121,121|1
q_cuhk03_off_cos_even_s1|cuhk03|off:cos|121,121,121,121,121|1
q_cuhk03_none_s1|cuhk03|none|121,121,121,121,121|1
q_cuhk03_off_ac_front_s0|cuhk03|off:ac|605,0,0,0,0|0
q_cuhk03_off_rule_front_s0|cuhk03|off:rule|605,0,0,0,0|0
q_msmt17_off_ac_front_s0|msmt17|off:ac|1490,0,0,0,0|0
q_msmt17_off_rule_front_s0|msmt17|off:rule|1490,0,0,0,0|0
q_market1501_off_ac_even_s0|market1501|off:ac|108,108,108,108,108|0
q_market1501_off_rule_even_s0|market1501|off:rule|108,108,108,108,108|0
q_market1501_off_cos_even_s0|market1501|off:cos|108,108,108,108,108|0
q_market1501_none_s0|market1501|none|108,108,108,108,108|0
q_market1501_oracle_s0|market1501|oracle|108,108,108,108,108|0
q_msmt17_off_ac_even_s1|msmt17|off:ac|298,298,298,298,298|1
q_msmt17_off_rule_even_s1|msmt17|off:rule|298,298,298,298,298|1
q_msmt17_off_cos_even_s1|msmt17|off:cos|298,298,298,298,298|1
q_msmt17_none_s1|msmt17|none|298,298,298,298,298|1
J
worker() {
  local gpu=$1 name dom strat sched seed out
  while IFS='|' read -r name dom strat sched seed; do
    [ -z "$name" ] && continue
    out=experiments/$name
    mkdir "$out" 2>/dev/null || continue
    echo "$name start gpu $gpu $(date +%T)" >> experiments/exit_codes_queue.txt
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/eval_active.py --output_dir "$out" --checkpoint $CK/base_$dom/checkpoint-12000 \
      --domains $dom --strategies $strat --budget_schedule $sched --base_seed $seed $COMMON > "$out/run.log" 2>&1 && touch "$out/.done"
    echo "$name exit $? gpu $gpu $(date +%T)" >> experiments/exit_codes_queue.txt
  done < $JOBS
}
# the old schedule must skip its MSMT17 jobs (now in this queue)
for s in off_ac none off_rule off_cos; do
  d=experiments/loop_msmt17_$s; mkdir -p $d
  [ -e $d/run.log ] || { touch $d/.done; echo "moved to q_msmt17_${s}_*_s0 (queue_1007.sh)" > $d/MOVED.txt; }
done
for g in 2 3 4 5; do worker $g & done
( while ! grep -q ALLDONE experiments/exit_codes_oracle.txt 2>/dev/null; do sleep 60; done; worker 0 ) &
( while ! grep -q ALLDONE experiments/exit_codes_oracle.txt 2>/dev/null; do sleep 60; done; worker 1 ) &
wait
echo ALLDONE >> experiments/exit_codes_queue.txt
