# Optimal Stopping VS Best-of-N For Inference Time Optimization

<div style="padding: 20px; text-align: center; color: white; font-family: Arial, sans-serif;">
  <p style="font-size: 24px; margin-bottom: 10px;">
    <a href="#" style="text-decoration: none;">Yusuf Hakan Kalayci</a><sup>1*</sup>, 
    <a href="#" style="text-decoration: none;">Vinod Raman</a><sup>2*</sup>, 
    <a href="#" style="text-decoration: none;">Shaddin Dughmi</a><sup>1</sup>
  </p>
  
  <p style="font-size: 16px; margin-bottom: 5px;">
    <sup>1</sup>University of Southern California
    <sup>2</sup>University of Michigan
    <sup>*</sup>Equal Contribution
  </p>
  
  <p style="font-size: 18px;">
    [<a href="https://arxiv.org/abs/2510.01394" style="color: #4a9eff; text-decoration: none;">Paper</a>] 
  </p>
</div>

## Experimental Pipeline

The experiment is divided into three main stages:

1.  **Response Generation**: In this stage, we use a large language model (LLM) to generate a set of responses for a given set of prompts.
2.  **Reward Computation**: After generating the responses, we use a reward model to assign a score to each response.
3.  **Algorithm Execution**: Finally, we run the Pandora's Box inspired Adaptive Best-of-N algorithm on the scored responses to evaluate its performance under different conditions.

## Dataset
During stages 1 and 2, we generate 960 different responses for each of 100 prompts sampled from the HH-RLHF and AlpacaFarm datasets. We then evaluate these responses using 5 different reward models. All generations and their corresponding rewards are stored in the `datasets/` folder.

## Setup and Installation

To get started, clone the repository and install the necessary dependencies:

```bash
# clone repo
git clone https://github.com/yhkalayci/adaptive-BoN.git
cd adaptive-BoN

# create env
conda create -n adaptivebon python=3.13 -y
conda activate adaptivebon

pip install -r requirements.txt
```

## Usage

For detailed instructions on how to run each stage of the pipeline, please refer to the `README.md` files in the `generation` and `algorithm` directories.

## Citation

If you use this work, please cite the following paper:
```bibtex
@misc{kalayci2025optimalstoppingvsbestofn,
      title={Optimal Stopping vs Best-of-$N$ for Inference Time Optimization}, 
      author={Yusuf Kalayci and Vinod Raman and Shaddin Dughmi},
      year={2025},
      eprint={2510.01394},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.01394}, 
}
```
