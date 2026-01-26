#!/usr/bin/env bash
set -euo pipefail

RMS=(mistral_rm fsfairx_rm)
LLMS=(gemma2_9b llama3.1_8b mistral_7b qwen2.5_7b)

SETTINGS=(
  "cdf shifted_exponential 0.99 1"
  "bradley_terry shifted_exponential 0.99 1"
  "bradley_terry lognormal 0.99 1"
  "bradley_terry shifted_exponential 0.99 2"
  "bradley_terry shifted_exponential 0.99 4"
  "bradley_terry shifted_exponential 0.99 8"
  "bradley_terry shifted_exponential 0.99 16"
  "bradley_terry shifted_exponential 0.9 1"
  "bradley_terry shifted_exponential 0.8 1"
  "bradley_terry shifted_exponential 0.7 1"
  "bradley_terry shifted_exponential 0.6 1"
)

run_dataset() {
  local dataset_name="$1"
  local input_folder="$2"
  local output_folder="$3"

  mkdir -p "$output_folder"

  for setting in "${SETTINGS[@]}"; do
    read -r transformation distribution alpha batch_size <<<"$setting"
    echo "Setting: transformation=$transformation distribution=$distribution alpha=$alpha batch_size=$batch_size"

    for rm in "${RMS[@]}"; do
      for llm in "${LLMS[@]}"; do
        echo "  Running: dataset=$dataset_name rm=$rm llm=$llm"
        python algorithm/experiment.py \
          --rm_name "$rm" \
          --llm_name "$llm" \
          --input_folder "$input_folder" \
          --output_folder "$output_folder" \
          --transformation "$transformation" \
          --distribution "$distribution" \
          --alpha "$alpha" \
          --batch_size "$batch_size"
      done
    done
  done
}

run_dataset "alpaca" "dataset/alpaca" "result_alpaca"
run_dataset "rlhf" "dataset/hh_rlhf" "result_rlhf"
