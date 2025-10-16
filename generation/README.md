# Generation and Reward Computation

This directory contains the scripts for generating responses from a language model and computing rewards for those responses.

## 1. Generating Responses

The `generate_responses.py` script uses the `vllm` library to generate a specified number of responses for each prompt in a given dataset.

### Usage

```bash
python generation/generate_responses.py --llm [LLM_NAME] --gpu_index [GPU_INDEX] --sample_count [SAMPLE_COUNT] --input_file [INPUT_FILE]
```

-   `--llm`: The name of the language model to use. Choices are: `llama3.2_3b`, `llama3.1_8b`, `qwen3_4b`, `qwen2.5_7b`, `mistral_7b`, `gemma2_9b`.
-   `--gpu_index`: The index of the GPU to use.
-   `--sample_count`: The number of responses to generate for each prompt.
-   `--input_file`: The path to the input JSON file containing the prompts.

### Example

```bash
python generation/generate_responses.py --llm llama3.1_8b --gpu_index 0 --sample_count 100 --input_file dataset/alpaca_farm_100.json
```

This command will generate 100 responses for each prompt in the `alpaca_farm_100.json` dataset using the `llama3.1_8b` model on GPU 0. The output will be saved to a file named `llama3.1_8b_output.jsonl`.

## 2. Computing Rewards

The `compute_rewards.py` script uses a reward model to score the generated responses.

### Usage

```bash
python generation/compute_rewards.py [INPUT_FOLDER] [MODEL_NAME] [OUTPUT_FOLDER] --gpu_index [GPU_INDEX]
```

-   `INPUT_FOLDER`: The folder containing the `.jsonl` files with the generated responses.
-   `MODEL_NAME`: The name of the reward model to use. Choices are: `armo_rm`, `fsfairx_rm`, `mistral_rm`, `skywork_rm`.
-   `OUTPUT_FOLDER`: The folder where the output files with the scores will be saved.
-   `--gpu_index`: The index of the GPU to use.

### Example

```bash
python generation/compute_rewards.py . armo_rm rewarded_output --gpu_index 0
```

This command will use the `armo_rm` reward model to score the responses in all `.jsonl` files in the current directory and save the results to the `rewarded_output` directory.
