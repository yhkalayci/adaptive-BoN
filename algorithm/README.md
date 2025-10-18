# Algorithm Execution

This directory contains the scripts for running the experiments for our paper. The scripts implement and evaluate the adaptive inference-time optimization method described in the paper, which is inspired from Pandora's Box problem. The algorithms are designed to balance output quality against inference cost for large language model (LLM) generation.

## 1. Experiment with Cost

The `experiment_with_cost.py` script runs the Pandora's Box algorithm with a range of different costs.

### Usage

```bash
python algorithm/experiment_with_cost.py --llm_name [LLM_NAME] --rm_name [RM_NAME] --input_folder [INPUT_FOLDER] --output_folder [OUTPUT_FOLDER] --seed [SEED]
```

-   `--llm_name`: The name of the language model used for generation.
-   `--rm_name`: The name of the reward model used for scoring.
-   `--input_folder`: The folder containing the scored responses.
-   `--output_folder`: The folder where the results will be saved.
-   `--seed`: The random seed to use for the experiment.

### Example

```bash
python algorithm/experiment_with_cost.py --llm_name llama3.1_8b --rm_name armo_rm --input_folder rewarded_output --output_folder results --seed 42
```

This command will run the experiment with the `llama3.1_8b` model and `armo_rm` reward model, using the data in the `rewarded_output` folder. The results will be saved in the `results` folder.

## 2. Experiment with Target Acceptance Rate

The `experiment_with_target_acceptance_rate.py` script runs the Pandora's Box algorithm with a range of different target acceptance rates.

### Usage

```bash
python algorithm/experiment_with_target_acceptance_rate.py --llm_name [LLM_NAME] --rm_name [RM_NAME] --input_folder [INPUT_FOLDER] --output_folder [OUTPUT_FOLDER] --seed [SEED]
```

-   `--llm_name`: The name of the language model used for generation.
-   `--rm_name`: The name of the reward model used for scoring.
-   `--input_folder`: The folder containing the scored responses.
-   `--output_folder`: The folder where the results will be saved.
-   `--seed`: The random seed to use for the experiment.

### Example

```bash
python algorithm/experiment_with_target_acceptance_rate.py --llm_name llama3.1_8b --rm_name armo_rm --input_folder rewarded_output --output_folder results --seed 42
```

This command will run the experiment with the `llama3.1_8b` model and `armo_rm` reward model, using the data in the `rewarded_output` folder. The results will be saved in the `results` folder.
