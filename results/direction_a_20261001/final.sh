# written by scripts/pick_best.py stage3 -- review before stage 4.
# Each option improved the tuning-fold validation by >= 1.0 mAP on its own;
# the combination itself was not validated.
export FINAL_ARGS="--lora_layers 12 --learning_rate 3e-4 --lr_scheduler_type cosine --warmup_steps 300 --triplet_margin 0.3"
