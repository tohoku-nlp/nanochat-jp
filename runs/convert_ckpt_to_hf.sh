#!/bin/bash
set -ex -o pipefail
INPUT_DIR=$1
OUTPUT_BASE_DIR=$2
STEP=$3

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Input directory does not exist: $INPUT_DIR" >&2
    exit 1
fi

if [[ ! -d "$OUTPUT_BASE_DIR" ]]; then
    echo "Output base directory does not exist: $OUTPUT_BASE_DIR" >&2
    exit 1
fi


if [[ -z "$STEP" ]]; then
    echo "Step number is not provided." >&2
    exit 1
fi


printf -v STEP_ZERO_PADDED "%06d" "$STEP"

if [[ ! -f "$INPUT_DIR/model_$STEP_ZERO_PADDED.pt" ]]; then
    echo "Model checkpoint does not exist: $INPUT_DIR/model_$STEP_ZERO_PADDED.pt" >&2
    exit 1
fi
OUTPUT_DIR="$OUTPUT_BASE_DIR/iter_$STEP_ZERO_PADDED"


python converter/convert_tokenizer_from_spm_to_hf.py

python converter/convert_nanochat_jp_checkpoints.py \
       --input_dir "$INPUT_DIR" \
       --output_dir "$OUTPUT_DIR" \
       --step "$STEP"
