#!/bin/bash
# Leave-one-source-out cross-validation: each fold holds out one source domain for validation.
# Usage: bash scripts/run_cv.sh <exp_name> [max_steps]
set -euo pipefail
EXP=${1:?exp name}
STEPS=${2:-1000}
PY=${PYTHON:-python}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/experiments/$EXP
mkdir -p "$OUT"
cd "$ROOT"

run_fold() {  # $1 = sources, $2 = val domain
    local dir=$OUT/val_$2
    mkdir -p "$dir"
    echo "[$(date "+%F %T")] start fold val=$2 sources=$1" | tee -a "$OUT/progress.txt"
    "$PY" scripts/train_reid.py \
        --output_dir "$dir" \
        --source_domains "$1" --val_domains "$2" \
        --max_steps "$STEPS" --per_device_train_batch_size 64 --num_icl_samples 64 \
        --learning_rate 1e-4 --lr_scheduler_type constant --weight_decay 0.0 --max_grad_norm 0 \
        --logging_steps 10 \
        --eval_strategy steps --eval_steps 100 --context_k 16 --context_method random --selection_unit identity --eval_seeds 1 \
        --save_strategy steps --save_steps "$STEPS" --save_total_limit 1 --save_safetensors False \
        --fp16 True --dataloader_num_workers 16 --dataloader_drop_last True \
        --seed 42 --report_to none > "$dir/train.log" 2>&1
    echo "[$(date "+%F %T")] end fold val=$2 exit=$?" | tee -a "$OUT/progress.txt"
}

run_fold market1501,msmt17 cuhk03
run_fold market1501,cuhk03 msmt17
run_fold msmt17,cuhk03     market1501
echo ALL_DONE >> "$OUT/progress.txt"
