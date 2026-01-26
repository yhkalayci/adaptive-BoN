#!/usr/bin/env bash
#SBATCH --job-name=pandora_exp_ws
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

set -euo pipefail

PYTHON_BIN="/home1/kalayci/env2/bin/python"
DATASET_ROOT="/scratch1/kalayci/pandora/adaptive-BoN/dataset"

: "${DATASET_NAME:?Set DATASET_NAME to alpaca or rlhf}"
: "${LLM_NAME:?Set LLM_NAME}"
: "${RM_NAME:?Set RM_NAME}"
: "${TRANSFORMATION:?Set TRANSFORMATION}"
: "${DISTRIBUTION:?Set DISTRIBUTION}"
: "${ALPHA:?Set ALPHA}"
: "${BATCH_SIZE:?Set BATCH_SIZE}"

if [[ "$DATASET_NAME" == "alpaca" ]]; then
  INPUT_FOLDER="$DATASET_ROOT/alpaca"
  OUTPUT_FOLDER="/scratch1/kalayci/pandora/adaptive-BoN/slurm_result_alpaca"
elif [[ "$DATASET_NAME" == "rlhf" ]]; then
  INPUT_FOLDER="$DATASET_ROOT/hh_rlhf"
  OUTPUT_FOLDER="/scratch1/kalayci/pandora/adaptive-BoN/slurm_result_rlhf"
else
  echo "Unknown DATASET_NAME: $DATASET_NAME" >&2
  exit 1
fi

mkdir -p "$OUTPUT_FOLDER"

echo "Dataset: $DATASET_NAME"
echo "Input: $INPUT_FOLDER"
echo "Output: $OUTPUT_FOLDER"
echo "Config: rm=$RM_NAME llm=$LLM_NAME transformation=$TRANSFORMATION distribution=$DISTRIBUTION alpha=$ALPHA batch_size=$BATCH_SIZE"

"$PYTHON_BIN" algorithm/experiment_with_target_acceptance_rate.py \
  --rm_name "$RM_NAME" \
  --llm_name "$LLM_NAME" \
  --input_folder "$INPUT_FOLDER" \
  --output_folder "$OUTPUT_FOLDER" \
  --transformation "$TRANSFORMATION" \
  --distribution "$DISTRIBUTION" \
  --alpha "$ALPHA" \
  --batch_size "$BATCH_SIZE"
