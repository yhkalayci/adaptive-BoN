import json


def read_data(input_folder, llm_name):
    """Read JSONL data for a given LLM."""
    file_name = f"{input_folder}/{llm_name}_output.merged_rm.jsonl"
    with open(file_name, "r") as f:
        return [json.loads(line) for line in f]


def write_results(output_file, payload):
    with open(output_file, "w") as f:
        f.write(json.dumps(payload, default=float) + "\n")


def print_cost_progress(label, result):
    print(
        "{} cost={:.6f} mean_wr={:.6f} median_wr={:.6f} mean_util={:.6f} median_util={:.6f} avg_open={:d}".format(
            label,
            result["cost"],
            result["mean"],
            result["median"],
            result["mean_utility"],
            result["median_utility"],
            result["sample_count"],
        )
    )


def print_fixed_n_progress(result):
    print(
        "fixed_n cost={:.6f} best_n={:d} mean_wr={:.6f} median_wr={:.6f} mean_util={:.6f} median_util={:.6f}".format(
            result["cost"],
            result["n"],
            result["mean"],
            result["median"],
            result["mean_utility"],
            result["median_utility"],
        )
    )
