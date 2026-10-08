# Copy to configs/local.sh and edit for your machine, then: source configs/local.sh
# (scripts/launch_tasks.py sources configs/local.sh automatically before every task)
export FERREID_DATA_ROOT=/data0/yb_data/dz_data/reid-data
export FERREID_WEIGHTS_DIR=/data1/yangbin/dz/weights       # vit_base_patch16_224.pth, dinov2_vitb14_pretrain.pth
export FERREID_DINOV2_REPO=$PWD/external/dinov2            # clone of facebookresearch/dinov2 (model code only)
export QWEN3_06B_DIR=/data1/yangbin/dz/models/Qwen3-0.6B       # only for the VICP baseline
export PYTHON=python
# Clone the pinned deep-person-reid dependency here (see docs/DEPLOYMENT.md).
export PYTHONPATH="$PWD/external/deep-person-reid${PYTHONPATH:+:$PYTHONPATH}"
