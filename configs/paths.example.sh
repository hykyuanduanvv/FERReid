# Copy to configs/local.sh and edit for your machine, then: source configs/local.sh
export FERREID_DATA_ROOT=/root/autodl-tmp/reid-data
export FERREID_WEIGHTS_DIR=/root/autodl-tmp/weights
export QWEN3_06B_DIR=/root/basic-models/Qwen3-0.6B
export PYTHON=python
# Clone the pinned deep-person-reid dependency here (see docs/DEPLOYMENT.md).
export PYTHONPATH="$PWD/external/deep-person-reid${PYTHONPATH:+:$PYTHONPATH}"
