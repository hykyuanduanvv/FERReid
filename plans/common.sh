# Shared settings / helpers for plans/*.tasks (sourced by scripts/launch_tasks.py before every task).
# Every experiment writes to experiments/<name>/ and refuses to overwrite a finished one.

PY=${PYTHON:-python}

# ---------------------------------------------------------------- base-model training
# 64 identities x 2 images, Adam lr 1e-4, fp16. Later arguments on a task line override these.
TRAIN_COMMON="--backbone dinov2_b14 --per_device_train_batch_size 64 --num_icl_samples 64 \
  --learning_rate 1e-4 --weight_decay 0.0 --max_grad_norm 0 --logging_steps 10 \
  --eval_strategy steps --eval_steps 1000 --context_k 16 --context_method random --selection_unit image --eval_seeds 1 \
  --save_strategy steps --save_total_limit 1 --save_safetensors False \
  --fp16 True --dataloader_num_workers 12 --dataloader_drop_last True --seed 42 --report_to none"
LOSS_TRIPLET=""                                       # hardest triplet (margin 0.1) + 0.01 WPA
LOSS_BOT="--ce_loss_weight 1.0 --bnneck True"         # + identity CE (label smoothing 0.1) + BNNeck
# decisions of the earlier stage-1/2 runs may be kept in plans/chosen.sh (not in git)
[ -f plans/chosen.sh ] && source plans/chosen.sh
LOSS_ARGS=${LOSS_ARGS-$LOSS_BOT}
STEPS=${STEPS:-12000}
SCHEDULE=${SCHEDULE:-"--lr_scheduler_type cosine --warmup_steps 500"}

# leave-one-out folds: target = market1501 | msmt17 | cuhk03; sources = train splits of the other three
# (CUHK-SYSU is always a source). fold <target> -> the training arguments of that fold
fold_sources() {
  case $1 in
    market1501) echo msmt17,cuhk03,cuhksysu ;;
    msmt17)     echo market1501,cuhk03,cuhksysu ;;
    cuhk03)     echo market1501,msmt17,cuhksysu ;;
    *) echo "unknown target $1" >&2; return 2 ;;
  esac
}
fold() { echo "--source_domains $(fold_sources "$1") --target_domains $1 --source_all_images False --val_domains none --eval_strategy no"; }

# multi-domain tokens of the base model (method A): m tokens per layer for every source domain
CLIP="--backbone clip_b16 --per_device_train_batch_size 32 --instances_per_id 4 --cross_camera_instances True --ot_loss_weight 0"
PASSB="--backbone pass_vitb --per_device_train_batch_size 32 --instances_per_id 4 --cross_camera_instances True --ot_loss_weight 0"
MD="--model_type vpt --source_domain_tokens ${MD_TOKENS:-8}"

# train <name> <steps> [extra args...]
train() {
  local name=$1 steps=$2; shift 2
  local out="experiments/$name"
  if [ -e "$out/trainer_state.json" ]; then echo "refusing to overwrite finished run $out" >&2; return 2; fi
  mkdir -p "$out"
  # shellcheck disable=SC2086
  "$PY" scripts/train_reid.py --output_dir "$out" --max_steps "$steps" --save_steps "$steps" $TRAIN_COMMON "$@" \
    2>&1 | tee "$out/train.log"
}

# ckpt <run name> -> its last checkpoint directory
ckpt() { ls -d "experiments/$1"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1; }

# ---------------------------------------------------------------- fallback queue (pilot watchdog)
# experiments/_launch/fallback.queue, one item per line:  <name> | <prerequisite run or -> | <command>
# run_fallback_queue (one worker per GPU, started by watchdog/watchdog.sh) takes items one after another until the
# queue is empty; an item waits until its prerequisite's final checkpoint exists (training still running elsewhere).
FALLBACK=experiments/_launch/fallback.queue
trained() { [ "$1" = "-" ] || ls experiments/"$1"/checkpoint-*/trainer_state.json >/dev/null 2>&1; }

run_fallback_queue() {
  local waited=0 line name need cmd picked
  while true; do
    picked=""
    exec 9>"$FALLBACK.lock"
    flock 9
    if [ -s "$FALLBACK" ]; then
      while IFS= read -r line; do
        need=$(echo "$line" | cut -d'|' -f2 | xargs)
        if [ -z "$picked" ] && trained "$need"; then picked=$line; fi
      done < "$FALLBACK"
      if [ -n "$picked" ]; then
        grep -vxF "$picked" "$FALLBACK" > "$FALLBACK.tmp"; mv "$FALLBACK.tmp" "$FALLBACK"
      fi
    fi
    local left=$( [ -s "$FALLBACK" ] && wc -l < "$FALLBACK" || echo 0 )
    flock -u 9
    if [ -z "$picked" ]; then
      [ "$left" -eq 0 ] && { echo "fallback queue empty"; return 0; }
      [ $waited -ge 180 ] && { echo "fallback: $left items still wait for checkpoints; giving up this slot"; return 0; }
      sleep 60; waited=$((waited + 1)); continue
    fi
    name=$(echo "$picked" | cut -d'|' -f1 | xargs)
    cmd=$(echo "$picked" | cut -d'|' -f3-)
    echo "[$(date +%T)] fallback item $name on GPU $CUDA_VISIBLE_DEVICES: $cmd"
    ( eval "$cmd" ) > "experiments/_launch/fb_$name.log" 2>&1
    echo $? > "experiments/_launch/fb_$name.status"
    echo "[$(date +%T)] fallback item $name exit $(cat experiments/_launch/fb_$name.status)"
  done
}

# gpu_wait: block until this task's GPU (CUDA_VISIBLE_DEVICES) is ours: a per-GPU lock (held by this task's shell
# until it exits, so two launchers never start on one GPU together) and less than 1.5 GB in use (runs started
# without the lock, e.g. by an older launcher, still finishing there)
gpu_wait() {
  # any GPU (not only the one the launcher assigned): the first whose per-GPU lock is free and that has less than
  # 1.5 GB in use; the lock is held (fd 8) until this task's shell exits and CUDA_VISIBLE_DEVICES is set to it
  local n g used
  n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  while true; do
    for g in $(seq 0 $((n - 1))); do
      exec 8>"/tmp/ferreid_gpu$g.lock"
      if flock -n 8; then
        used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
        if [ "$used" -lt 1500 ]; then export CUDA_VISIBLE_DEVICES=$g; echo "[gpu_wait] using GPU $g"; return 0; fi
        flock -u 8
      fi
      exec 8>&-
    done
    sleep 30
  done
}

# ---------------------------------------------------------------- active module
# active <name> <checkpoint> [extra args...]: scripts/eval_active.py
active() {
  local name=$1 ck=$2; shift 2
  local out="experiments/$name"
  if [ -e "$out/summary.csv" ] && [ -e "$out/.done" ]; then echo "refusing to overwrite $out" >&2; return 2; fi
  [ -d "$ck" ] || { echo "missing checkpoint $ck" >&2; return 2; }
  mkdir -p "$out"
  "$PY" scripts/eval_active.py --output_dir "$out" --checkpoint "$ck" --fp16 True --report_to none \
    --eval_num_workers 12 "$@" 2>&1 | tee "$out/active.log" && touch "$out/.done"
}

# diag <name> <checkpoint> [extra args...]: scripts/diag_retrieval.py
diag() {
  local name=$1 ck=$2; shift 2
  local out="experiments/$name"
  if [ -e "$out/diag_retrieval.json" ]; then echo "refusing to overwrite $out" >&2; return 2; fi
  [ -d "$ck" ] || { echo "missing checkpoint $ck" >&2; return 2; }
  mkdir -p "$out"
  "$PY" scripts/diag_retrieval.py --output_dir "$out" --checkpoint "$ck" --fp16 True --report_to none \
    --eval_num_workers 12 "$@" 2>&1 | tee "$out/diag.log"
}

# in-context baseline: evaluate <name> <checkpoint> [extra args...] (scripts/eval_context.py, VICP)
evaluate() {
  local name=$1 ck=$2; shift 2
  local out="experiments/$name"
  if [ -e "$out/context_eval.csv" ]; then echo "refusing to overwrite $out/context_eval.csv" >&2; return 2; fi
  mkdir -p "$out"
  "$PY" scripts/eval_context.py --output_dir "$out" --checkpoint "$ck" \
    --num_icl_samples 64 --fp16 True --report_to none --eval_num_workers 12 "$@" 2>&1 | tee "$out/eval.log"
}

# strategies of the main comparison (pair strategies + the ID-level anchor protocol)
STRATS=${STRATS:-"cover,uncertain,balanced,confident,random,anchor:random,anchor:facility_camera"}
# active-module settings of the main runs: 5 rounds x 50 yes/no answers (250 in all; was 1,000)
ACT=${ACT:-"--rounds 5 --budget 50 --eval_rounds 1,3,5"}
[ -f plans/active_chosen.sh ] && source plans/active_chosen.sh   # lr / steps chosen by plans/tune.tasks
ACT_TUNE=${ACT_TUNE:-""}

# ---------------------------------------------------------------- cluster repair (docs/CLUSTER_REPAIR.md)
# pseudo labels of the whole pool + merge / split questions; 5 rounds x 50 answers as in the pair setting, the
# domain tokens continue across rounds (cluster-then-train loop). REP_TUNE: chosen by plans/repair_tune.tasks
REP=${REP:-"--pseudo True --warm_start True --rounds 5 --budget 50 --eval_rounds 1,3,5 --steps 400 --paired_ref repair_random --oracle_all False"}
REP_STRATS=${REP_STRATS:-"none,repair,repair_unc,repair_random,random,cover,disagree"}
[ -f plans/repair_chosen.sh ] && source plans/repair_chosen.sh
REP_TUNE=${REP_TUNE:-""}

# sim <name> <checkpoint> [extra args...]: scripts/sim_selection.py (offline screening, no training)
sim() {
  local name=$1 ck=$2; shift 2
  local out="experiments/$name"
  if [ -e "$out/sim_summary.csv" ] && [ -e "$out/.done" ]; then echo "refusing to overwrite $out" >&2; return 2; fi
  [ -d "$ck" ] || { echo "missing checkpoint $ck" >&2; return 2; }
  mkdir -p "$out"
  "$PY" scripts/sim_selection.py --output_dir "$out" --checkpoint "$ck" --fp16 True --report_to none \
    --eval_num_workers 12 "$@" 2>&1 | tee "$out/sim.log" && touch "$out/.done"
}
