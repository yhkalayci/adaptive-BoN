# Pandora's Box: An ML Experimentation Framework

This repository contains the code for a series of machine learning experiments that implement the Pandora's Box algorithm. The pipeline is designed to be modular, allowing for easy extension and modification.

## Experimental Pipeline

The project is divided into three main stages:

1.  **Response Generation**: In this stage, we use a large language model (LLM) to generate a set of responses for a given set of prompts.
2.  **Reward Computation**: After generating the responses, we use a reward model to assign a score to each response.
3.  **Algorithm Execution**: Finally, we run the Pandora's Box algorithm on the scored responses to evaluate its performance under different conditions.

## Setup and Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yhkalayci/adaptive-BoN.git
cd adaptive-BoN
pip install -r requirements.txt
```

## Usage

For detailed instructions on how to run each stage of the pipeline, please refer to the `README.md` files in the `generation` and `algorithm` directories.
