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

# small targets (VIPeR / GRID / i-LIDS): sources = every image of Market + MSMT17 + CUHK03 (no validation)
SMALL_SRC="--source_domains market1501,msmt17,cuhk03 --source_all_images True --val_domains none"

# large targets: leave-one-out over Market1501 / MSMT17 / CUHK03 (+ CUHK-SYSU when its data is present),
# sources = train splits. large_sources market1501 -> msmt17,cuhk03[,cuhksysu]
large_sources() {
  local t=$1 out="" d
  for d in market1501 msmt17 cuhk03 cuhksysu; do
    [ "$d" = "$t" ] && continue
    [ "$d" = cuhksysu ] && [ ! -d "${FERREID_DATA_ROOT:-/root/autodl-tmp/reid-data}/cuhksysu" ] && continue
    out="${out:+$out,}$d"
  done
  echo "$out"
}

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

# base checkpoints used by the active plans (override in configs/local.sh, e.g. with an existing VPT run)
BASE_SMALL=${BASE_SMALL:-$(ckpt base_vpt_small)}
STRATS=${STRATS:-"cover,uncertain,balanced,confident,random,anchor:random,anchor:facility_camera"}
