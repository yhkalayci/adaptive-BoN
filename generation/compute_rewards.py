# score_generations.py (16-bit Precision Only)

import argparse
import glob
import json
import os
import sys
import torch
from tqdm import tqdm

# Prerequisite Checks
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from importlib.metadata import version
except ImportError:
    print("Transformers library not found. Please install it with 'pip install transformers'")
    sys.exit(1)

# Reward Model Configuration (Verified)
REWARD_MODEL_CONFIG = {
    "armo_rm": {
        "path": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        "formatter": lambda p, r: [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
    },
    "fsfairx_rm": {
        "path": "sfairXC/FsfairX-LLaMA3-RM-v0.1",
        "formatter": lambda p, r: [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
    },
    "mistral_rm": {
        "path": "weqweasdas/RM-Mistral-7B",
        "formatter": lambda p, r: [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
    },
    "skywork_rm": {
        "path": "Skywork/Skywork-Reward-Llama-3.1-8B",
        "formatter": lambda p, r: [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
    }
}

def check_armo_rm_compatibility(model_name):
    """Checks for transformers version compatibility with ArmoRM."""
    if model_name == "armo_rm":
        current_version = version('transformers')
        if current_version > '4.40.0':
            print(f"Error: Your transformers version ({current_version}) is incompatible with 'armo_rm'.")
            print("Please downgrade with: pip install transformers==4.39.3")
            sys.exit(1)

def score_generations(args):
    """Scores all generations in a folder using a specified reward model."""
    check_armo_rm_compatibility(args.model_name)

    # Setup and Validation
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder not found at {args.input_folder}")
        return
    os.makedirs(args.output_folder, exist_ok=True)
    
    config = REWARD_MODEL_CONFIG[args.model_name]
    model_path = config["path"]
    formatter = config["formatter"]

    # Load Model and Tokenizer in 16-bit
    print(f"Loading reward model: {model_path} in 16-bit (bfloat16)...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded successfully.")

    # Process Files
    jsonl_files = glob.glob(os.path.join(args.input_folder, "*.jsonl"))
    for file_path in tqdm(jsonl_files, desc="Processing files"):
        basename = os.path.basename(file_path)
        output_filename = f"{os.path.splitext(basename)[0]}.{args.model_name}.jsonl"
        output_path = os.path.join(args.output_folder, output_filename)

        with open(file_path, 'r') as infile, open(output_path, 'w') as outfile:
            for line in tqdm(infile, desc=f"Scoring {basename}", leave=False):
                try:
                    data = json.loads(line)
                    prompt = data["prompt"]
                    generations = data["generations"]
                    all_rewards = []
                    
                    # Process generations in smaller chunks (mini-batches)
                    batch_iterator = range(0, len(generations), args.batch_size)
                    for i in tqdm(batch_iterator, desc="Scoring batches", leave=False):
                        batch_generations = generations[i : i + args.batch_size]

                        formatted_inputs = [
                            tokenizer.apply_chat_template(formatter(prompt, gen["text"]), tokenize=False)
                            for gen in batch_generations
                        ]

                        with torch.no_grad():
                            inputs = tokenizer(
                                formatted_inputs, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True,
                                max_length=4096
                            ).to("cuda")
                            
                            outputs = model(**inputs)
                            batch_rewards = outputs.logits.squeeze().cpu().float().tolist()
                        
                        if not isinstance(batch_rewards, list):
                            batch_rewards = [batch_rewards]
                        
                        all_rewards.extend(batch_rewards)
                    
                    reward_key = f"{args.model_name}_reward"
                    for i, gen in enumerate(generations):
                        gen[reward_key] = all_rewards[i]
                    
                    outfile.write(json.dumps(data) + '\n')

                except Exception as e:
                    print(f"An error occurred while processing a line in {basename}: {e}")

    print("All files processed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Compute rewards for LLM generations using a reward model.")
    parser.add_argument("input_folder", type=str, help="Folder containing the input .jsonl files.")
    parser.add_argument("model_name", type=str, choices=REWARD_MODEL_CONFIG.keys(), help="Short name of the reward model to use.")
    parser.add_argument("output_folder", type=str, help="Folder where the output files will be saved.")
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="The index of the GPU to use (e.g., 0, 1, 2)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for scoring to prevent OutOfMemoryError. Adjust based on your GPU."
    )
    
    args = parser.parse_args()

    # Set the visible GPU device
    print(f"Setting CUDA_VISIBLE_DEVICES to '{args.gpu_index}'")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    score_generations(args)

if __name__ == "__main__":
    main()