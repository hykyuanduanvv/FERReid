# Shared settings / helpers for plans/*.tasks (sourced by scripts/launch_tasks.py before every task).
# Every experiment writes to experiments/<name>/ and refuses to overwrite an existing trainer_state.json.

PY=${PYTHON:-python}

# ---------------------------------------------------------------- tuning fold (stages 1-3)
# Protocol-2 fold "target = CUHK-SYSU" has sources Market + MSMT17 + CUHK03. Hyper-parameters are
# chosen inside it by leaving one *source* out: train on the train splits of Market + MSMT17,
# validate on CUHK03 (an unseen domain). No target domain is looked at while tuning, and CUHK-SYSU
# is not needed for stages 1-3.
TUNE_DATA="--source_domains market1501,msmt17 --source_all_images False --val_domains cuhk03 --val_max_ids 500"

# ---------------------------------------------------------------- losses compared in stage 1
LOSS_TRIPLET=""                                       # historical: hardest triplet (margin 0.1) + 0.01 WPA
LOSS_BOT="--ce_loss_weight 1.0 --bnneck True"         # + identity CE (label smoothing 0.1) + BNNeck
FULL_FT="--train_backbone full --backbone_lr_mult 0.1"  # full fine-tuning, encoder lr = 0.1 x lr

# ---------------------------------------------------------------- common training arguments
# 64 identities x 2 images, Adam lr 1e-4, fp16, validation every 250 steps (label-free image selection,
# random, k=16; only VICP uses the context). Later arguments on a task line override these.
TRAIN_COMMON="--backbone dinov2_b14 --per_device_train_batch_size 64 --num_icl_samples 64 \
  --learning_rate 1e-4 --lr_scheduler_type constant --weight_decay 0.0 --max_grad_norm 0 \
  --logging_steps 10 --eval_strategy steps --eval_steps 250 \
  --context_k 16 --context_method random --selection_unit image --eval_seeds 1 \
  --save_strategy steps --save_total_limit 1 --save_safetensors False \
  --fp16 True --dataloader_num_workers 12 --dataloader_drop_last True --seed 42 --report_to none"

# stage-1/2 decisions are written here by scripts/pick_best.py (LOSS_ARGS, STEPS, SCHEDULE)
[ -f plans/chosen.sh ] && source plans/chosen.sh
# Empty LOSS_ARGS intentionally selects triplet-only; default only when unset.
LOSS_ARGS=${LOSS_ARGS-$LOSS_BOT}
STEPS=${STEPS:-12000}
SCHEDULE=${SCHEDULE:-"--lr_scheduler_type cosine --warmup_steps 500"}
# stage-3 decisions (pedestrian-specific options kept after the sweep) go into plans/final.sh
[ -f plans/final.sh ] && source plans/final.sh
FINAL_ARGS=${FINAL_ARGS:-""}

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

# evaluate <name> <checkpoint> [extra args...]   (architecture / source data are read from the checkpoint)
evaluate() {
  local name=$1 ckpt=$2; shift 2
  local out="experiments/$name"
  if [ -e "$out/context_eval.csv" ]; then echo "refusing to overwrite $out/context_eval.csv" >&2; return 2; fi
  mkdir -p "$out"
  "$PY" scripts/eval_context.py --output_dir "$out" --checkpoint "$ckpt" \
    --num_icl_samples 64 --fp16 True --report_to none --eval_num_workers 12 "$@" 2>&1 | tee "$out/eval.log"
}

# protocol-2 sources for a target: p2_sources market1501 -> msmt17,cuhksysu,cuhk03
p2_sources() {
  local t=$1 out="" d
  for d in market1501 msmt17 cuhksysu cuhk03; do [ "$d" != "$t" ] && out="${out:+$out,}$d"; done
  echo "$out"
}

# ---------------------------------------------------------------- direction A (plans/dirA.tasks)
A_RES="--prompt_mode residual"                                  # prompt = base + gate * delta(context)
A_EMA="--prompt_mode residual --ctx_center ema"                 # delta only sees what changes with the context
A_EPI="--episode_context_ids 16 --episode_context_ids_min 4 --unique_ids_per_batch True"  # context people != retrieved people, k in [4,16]
A_CAM="--pseudo_domains camera_pair --pseudo_min_ids 64 --unique_ids_per_batch True"  # (dataset, camera pair) pseudo-domains; >= 64 ids so a batch has no repeated person

# gain <name> <checkpoint> [extra args...]: direction-A decision test (scripts/context_gain.py) on the small
# target domains; the stage-1 VPT run with the same loss is the no-context reference when it exists.
gain() {
  local name=$1 ckpt=$2; shift 2
  local out="experiments/$name" ref="experiments/${LOSS_REF:-s1_vpt_bot}/checkpoint-3000"
  if [ -e "$out/context_gain.csv" ]; then echo "refusing to overwrite $out/context_gain.csv" >&2; return 2; fi
  mkdir -p "$out"
  local refarg=""; [ -d "$ref" ] && refarg="--reference_checkpoint $ref"
  # shellcheck disable=SC2086
  "$PY" scripts/context_gain.py --output_dir "$out" --checkpoint "$ckpt" $refarg \
    --domains viper,grid,ilids --eval_splits 3 --ks 4,16 --n_draws 5 --n_cross 2 --selection_unit image \
    --num_icl_samples 64 --fp16 True --report_to none "$@" 2>&1 | tee "$out/gain.log"
}

# ---------------------------------------------------------------- direction B (plans/select_*.tasks)
# best direction-A configuration / run, set after plans/dirA.tasks (override in plans/chosen.sh)
A_BEST_ARGS=${A_BEST_ARGS:-"$A_EMA $A_EPI $A_CAM"}
A_BEST_RUN=${A_BEST_RUN:-a5_all}
SELECTORS_ALL="random,first,dedup,pairable,typical,kcenter,camera_balanced,style_cover,hard_negative,facility,facility_camera"

# selsweep <name> <checkpoint> <incontext|tuned> [extra args...]: all selectors on the small target domains
selsweep() {
  local name=$1 ckpt=$2 gen=$3; shift 3
  local out="experiments/$name"
  if [ -e "$out/selectors.csv" ]; then echo "refusing to overwrite $out/selectors.csv" >&2; return 2; fi
  mkdir -p "$out"
  "$PY" scripts/eval_selectors.py --output_dir "$out" --checkpoint "$ckpt" --generator "$gen" \
    --selectors "$SELECTORS_ALL" --num_icl_samples 64 --report_to none "$@" 2>&1 | tee "$out/selectors.log"
}

# ---------------------------------------------------------------- direction A, contrastive context loss
# (plans/dirA_contrast.tasks, docs/DIRECTION_A_CONTRAST.md). Warm start = a strong DINOv2 VPT trained with
# triplet+WPA (no BNNeck head); its prompt becomes base_prompt, only the context branch is trained.
VPT_WARM=${VPT_WARM:-experiments/s2_vpt_long/checkpoint-12000}
# loss written out (not $LOSS_ARGS): a BNNeck head absent from the VPT would be random *and* frozen
C_BASE="--model_type vicp --prompt_mode residual --delta_init_std 0 --init_from $VPT_WARM \
  --train_context_only True --bnneck False --ce_loss_weight 0 \
  --episode_context_ids 16 --unique_ids_per_batch True --batch_domain_mode single --eval_steps 100"
CTR="--ctx_contrast_weight 1.0 --ctx_contrast_margin 0.05"

# cgain <name> <checkpoint>: context_gain.py with the warm-start VPT as the no-context reference
cgain() {
  local name=$1 ckpt=$2; shift 2
  local out="experiments/$name"
  if [ -e "$out/context_gain.csv" ]; then echo "refusing to overwrite $out/context_gain.csv" >&2; return 2; fi
  mkdir -p "$out"
  # shellcheck disable=SC2086
  "$PY" scripts/context_gain.py --output_dir "$out" --checkpoint "$ckpt" --reference_checkpoint "$VPT_WARM" \
    --domains viper,grid,ilids --eval_splits 3 --ks 4,16 --n_draws 5 --n_cross 2 --selection_unit image \
    --num_icl_samples 64 --fp16 True --report_to none "$@" 2>&1 | tee "$out/gain.log"
}

# ---------------------------------------------------------------- direction A, prompt distillation
# (plans/dirA_distill.tasks, docs/DIRECTION_A_DISTILL.md): batches from style camera groups, teachers per group
KD_DIR=${KD_DIR:-experiments/groups_style_lr1e-4_s1000}
KD_GRP="--pseudo_domains camera_group --camera_groups $KD_DIR/groups.json --pseudo_min_ids 64"
KD_TEACH="--prompt_teacher $KD_DIR/teachers.pt"
