#!/bin/bash
set -e -o pipefail

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=8

# -----------------------------------------------------------------------------
# activate venv so that `python` uses the project's venv instead of system python
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


# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer
python -m scripts.tok_train

# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval


# -----------------------------------------------------------------------------
# Base model (pretraining)
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
	 --depth=22 \
	 --target-param-data-ratio=60 \
	 --device-batch-size=32 \
	 --run=$WANDB_RUN \
	 --core-metric-every -1 \
	 --save-every 5000

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens)
torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- \
	 --device-batch-size=16 \
	 --chatcore-every -1 \
	 --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# RL (optional)
# Since the context length of the default nanochat is limited, we apply reinforcement learning to ensure the output length fits within the context.

# torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl -- \
# 	 --device-batch-size=4 \
# 	 --run=$WANDB_RUN \
# 	 --max-new-tokens 1536 \
# 	 --top-k 0 \
# 	 --init-lr-frac 0.01 \
# 	 --save-every 10 \
# 	 --num-samples 8 \
# 	 --eval-every -1


# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
