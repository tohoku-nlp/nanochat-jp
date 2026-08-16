#!/bin/bash
set -eux

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=8

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# Load optional per-user settings from .env (gitignored). set -a exports every
# assignment in the file, so .env needs no `export` of its own.
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi


# -----------------------------------------------------------------------------
# Download the dataset.
python -m nanochat.dataset -n 58 --num-workers 16


BASE_DIR=`python -c "from nanochat.common import get_base_dir; print(str(get_base_dir()))"`
# Download the SFT dataset from Hugging Face
hf download tohoku-nlp/nanochat-jp-sft --repo-type dataset --local-dir $BASE_DIR/datasets/nanochat-jp-sft --include "v1/*"

# Download the RL dataset from Hugging Face
hf download tohoku-nlp/nanochat-jp-rl --repo-type dataset --local-dir $BASE_DIR/datasets/nanochat-jp-rl --include "v0/*"

# Download the eval bundle from Hugging Face
hf download tohoku-nlp/nanochat-jp-eval-bundle --repo-type dataset --local-dir $BASE_DIR/eval_bundle

