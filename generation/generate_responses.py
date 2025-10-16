# generate.py (Corrected Version)

import os
import json
import argparse
import logging
from vllm import LLM, SamplingParams

# --- Model Configuration ---
# (This section remains unchanged)
MODEL_CONFIG = {
    "llama3.2_3b": {
        "path": "meta-llama/Llama-3.2-3B-Instruct",
        "eos_token": "<|eot_id|>",
    },
    "llama3.1_8b": {
        "path": "meta-llama/Llama-3.1-8B-Instruct",
        "eos_token": "<|eot_id|>",
    },
    "qwen3_4b": {
        "path": "Qwen/Qwen3-4B-Instruct-2507",
        "eos_token": "<|im_end|>",
    },
    "qwen2.5_7b": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "eos_token": "<|im_end|>",
    },
    "mistral_7b": {
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
        "eos_token": "</s>",
    },
    "gemma2_9b": {
        "path": "google/gemma-2-9b-it",
        "eos_token": "<eos>",
    },
}

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    """Main function to run the generation script."""
    parser = argparse.ArgumentParser(
        description="Run LLM generation for multiple prompts using vLLM."
    )
    # (Parser arguments remain unchanged)
    parser.add_argument("--llm", type=str, required=True, choices=MODEL_CONFIG.keys(), help="Short name of the LLM to use for generation.")
    parser.add_argument("--gpu_index", type=int, required=True, help="The index of the GPU to use (e.g., 0 for GPU 0).")
    parser.add_argument("--sample_count", type=int, default=960, help="Number of responses to generate for each prompt.")
    parser.add_argument("--input_file", type=str, default="alpaca_farm_100.json", help="Path to the input JSON file.")
    args = parser.parse_args()

    # --- 1. Set Environment and Configuration ---
    # (This section remains unchanged)
    logging.info(f"Setting CUDA_VISIBLE_DEVICES to '{args.gpu_index}'")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    config = MODEL_CONFIG[args.llm]
    model_path = config["path"]
    eos_token = config["eos_token"]
    output_file = f"{args.llm}_output.jsonl"
    logging.info(f"Starting generation with model: {model_path}")
    logging.info(f"Number of samples per prompt: {args.sample_count}")
    logging.info(f"Output will be written to: {output_file}")

    # --- 2. Load Input Data ---
    # (This section remains unchanged)
    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded {len(data)} prompts from {args.input_file}")
    except FileNotFoundError:
        logging.error(f"Error: Input file '{args.input_file}' not found.")
        return

    # --- 3. Initialize vLLM Model and Sampling Parameters ---
    logging.info("Initializing vLLM engine... (This may take a few minutes)")
    llm = LLM(
        model=model_path, 
        trust_remote_code=True, 
        tensor_parallel_size=1,
        max_model_len=8192
    )
    tokenizer = llm.get_tokenizer()
    
    # Set sampling parameters as requested
    sampling_params = SamplingParams(
        n=args.sample_count,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        max_tokens=8000,
        stop=[eos_token],
        logprobs=1 
    )
    
    # --- 4. Generation Loop ---
    logging.info("Starting generation loop...")
    
    with open(output_file, 'a', encoding='utf-8') as f:
        for i, item in enumerate(data):
            prompt_text = item.get("prompt")
            if not prompt_text:
                logging.warning(f"Skipping item {i} due to missing 'prompt' key.")
                continue

            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            logging.info(f"Processing prompt {i+1}/{len(data)}...")
            
            outputs = llm.generate(formatted_prompt, sampling_params)
            
            processed_generations = []
            for completion_output in outputs[0].outputs:
                sequence_logprob = 0.0
                if completion_output.logprobs:
                    for i, token_id in enumerate(completion_output.token_ids):
                        step_logprobs = completion_output.logprobs[i]
                        # --- THIS IS THE CORRECTED LINE ---
                        # Access the .logprob attribute of the Logprob object
                        sequence_logprob += step_logprobs[token_id].logprob

                generation_data = {
                    "text": completion_output.text.strip(),
                    "log_likelihood": sequence_logprob
                }
                processed_generations.append(generation_data)

            output_record = {
                **item,
                "generations": processed_generations
            }
            
            f.write(json.dumps(output_record) + '\n')
            f.flush()
            logging.info(f"Completed and saved {args.sample_count} responses (with log likelihoods) for prompt {i+1}.")

    logging.info(f"All prompts processed. Results are saved in {output_file}.")


if __name__ == "__main__":
    main()