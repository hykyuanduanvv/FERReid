#!/bin/bash
# cv_0926 fold1 (Market+MSMT -> val CUHK03) rerun with 3000 steps; all other settings identical
set -euo pipefail
cd "$(dirname "$0")/.."
D=experiments/cv_fold1_3k/val_cuhk03
mkdir -p "$D"
"${PYTHON:-python}" scripts/train_reid.py --output_dir "$D" \
    --source_domains market1501,msmt17 --val_domains cuhk03 \
    --max_steps 3000 --per_device_train_batch_size 64 --num_icl_samples 64 \
    --learning_rate 1e-4 --lr_scheduler_type constant --weight_decay 0.0 --max_grad_norm 0 \
    --logging_steps 10 \
    --eval_strategy steps --eval_steps 100 --context_k 16 --context_method random --selection_unit identity --eval_seeds 1 \
    --save_strategy steps --save_steps 3000 --save_total_limit 1 --save_safetensors False \
    --fp16 True --dataloader_num_workers 16 --dataloader_drop_last True \
    --seed 42 --report_to none > "$D/train.log" 2>&1
