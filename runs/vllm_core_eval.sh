#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=8

# -----------------------------------------------------------------------------
# Python venv setup with uv
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# W&B credentials and per-user settings come from .env (gitignored), never from
# this script: an API key committed here would live in the repo's history
# permanently. Sourcing is optional -- without it wandb simply does not
# authenticate, which is what WANDB_RUN=dummy expects anyway. set -a exports
# every assignment in the file, so .env needs no `export` of its own.
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

# evaluate the model: CORE metric, BPB on train/val, and draw samples
python -m scripts.core_eval_vllm \
       --base-url http://localhost:8000 \
       --model Qwen/Qwen3.5-0.8B-Base \
       --concurrency 1
