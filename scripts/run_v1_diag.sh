#!/bin/bash
# reduced diagnostics for V1 (ICL questions from the trained encoder)
set -euo pipefail
cd "$(dirname "$0")/.."
D=experiments/v1_iclfeat/diag
mkdir -p "$D"
"${PYTHON:-python}" scripts/context_diagnostics.py --output_dir "$D" \
    --checkpoint experiments/v1_iclfeat/val_cuhk03/checkpoint-1000 --icl_feature trained \
    --domains viper,grid,ilids --ks 4,16 --n_contexts 4 --eval_splits 1 --selection_unit identity \
    --num_icl_samples 64 --fp16 True --report_to none > "$D/run.log" 2>&1
