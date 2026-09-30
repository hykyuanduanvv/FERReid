#!/bin/bash
# V1: ICL questions built from the trained encoder (LoRA, no prompts); otherwise identical to cv_0926 fold1
set -euo pipefail
cd "$(dirname "$0")/.."
D=experiments/v1_iclfeat/val_cuhk03
mkdir -p "$D"
"${PYTHON:-python}" scripts/train_reid.py --output_dir "$D" \
    --model_type vicp --icl_feature trained \
    --source_domains market1501,msmt17 --val_domains cuhk03 \
    --max_steps 1000 --per_device_train_batch_size 64 --num_icl_samples 64 \
    --learning_rate 1e-4 --lr_scheduler_type constant --weight_decay 0.0 --max_grad_norm 0 \
    --logging_steps 10 \
    --eval_strategy steps --eval_steps 100 --context_k 16 --context_method random --selection_unit identity --eval_seeds 1 \
    --save_strategy steps --save_steps 1000 --save_total_limit 1 --save_safetensors False \
    --fp16 True --dataloader_num_workers 16 --dataloader_drop_last True \
    --seed 42 --report_to none > "$D/train.log" 2>&1
