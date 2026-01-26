#!/usr/bin/env bash
set -euo pipefail

RMS=(mistral_rm fsfairx_rm)
LLMS=(gemma2_9b llama3.1_8b mistral_7b qwen2.5_7b)

SETTINGS=(
  "bradley_terry shifted_exponential 0.99 1"
  "bradley_terry lognormal 0.99 1"
  "cdf shifted_exponential 0.99 1"
  "cdf lognormal 0.99 1"
  "bradley_terry shifted_exponential 0.9 1"
  "bradley_terry shifted_exponential 0.8 1"
  "bradley_terry shifted_exponential 0.7 1"
  "bradley_terry shifted_exponential 0.6 1"
)

DATASETS=(alpaca rlhf)

for dataset in "${DATASETS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    read -r transformation distribution alpha batch_size <<<"$setting"
    for rm in "${RMS[@]}"; do
      for llm in "${LLMS[@]}"; do
        sbatch \
          --export=DATASET_NAME="$dataset",LLM_NAME="$llm",RM_NAME="$rm",TRANSFORMATION="$transformation",DISTRIBUTION="$distribution",ALPHA="$alpha",BATCH_SIZE="$batch_size" \
          slurm_experiment_ws.sh
      done
    done
  done
done
