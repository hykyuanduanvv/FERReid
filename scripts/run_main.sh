#!/usr/bin/env bash
# Usage: bash scripts/run_main.sh vicp experiment_name [steps] [training_seed]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-vicp}
EXP=${2:-main_vicp}
STEPS=${3:-1000}
SEED=${4:-42}
[[ "$MODEL" == vicp || "$MODEL" == plain ]] || { echo 'model must be vicp or plain' >&2; exit 2; }
[[ "$EXP" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Use a simple experiment name without paths' >&2; exit 2; }
OUT="experiments/$EXP/val_cuhk03"
[[ ! -e "$OUT" ]] || { echo "Output already exists: $OUT. Choose a new experiment name." >&2; exit 2; }
mkdir -p "$OUT"
"${PYTHON:-python}" scripts/train_reid.py \
  --output_dir "$OUT" --model_type "$MODEL" \
  --source_domains market1501,msmt17 --val_domains cuhk03 --source_all_images True \
  --max_steps "$STEPS" --per_device_train_batch_size 64 --num_icl_samples 64 \
  --learning_rate 1e-4 --lr_scheduler_type constant --weight_decay 0.0 --max_grad_norm 0 \
  --logging_steps 10 --eval_strategy steps --eval_steps 100 \
  --context_k 16 --context_method random --selection_unit identity --eval_seeds 1 --val_max_ids 500 \
  --save_strategy steps --save_steps "$STEPS" --save_total_limit 1 --save_safetensors False \
  --fp16 True --dataloader_num_workers 16 --dataloader_drop_last True \
  --seed "$SEED" --report_to none 2>&1 | tee "$OUT/train.log"
