"""Report token-count DMRL net utility against a train-selected fixed N."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pandora_token_alignment_report import (
    BOOTSTRAP_SEED, FOCUS_GENERATOR, FOCUS_REWARDS, MODELS,
    PRICES_PER_MILLION, REWARD_LABELS, SPLIT_SEEDS, draw_figures,
    evaluate_pair, focused_tables, write_csv,
)


def write_report(tables, output):
    lines = [
        "# Token-count DMRL alignment: adaptive versus train-selected fixed N", "",
        "The adaptive policy is the frozen `AdaptiveAlignment mean_costse2` rule. "
        "Each fixed best-of-N baseline chooses one integer N on training prompts to "
        "maximize mean net utility at that cost multiplier, then uses that same N for "
        "every held-out prompt. Each table uses Qwen 3.5 9B; the figures show all "
        "four generators for FSFairX and RM-Mistral.", "",
        "**Net utility (profit) = reward-based quality − generation cost.** Quality "
        "is sigmoid(selected reward minus the prompt's full-pool reward q99). Cost "
        "uses recorded output tokens at the illustrative price per million tokens. "
        "Reward scoring and input-token costs are excluded.", "",
        "For each price and reward model, values average five 50/50 held-out split "
        "results. Mean fixed N can be fractional because a separate integer N was "
        "selected in each split. Percent gains and direct cost savings average the "
        "splitwise percentages, so they need not equal ratios of the displayed means. "
        "The same eight response orders are used for adaptive and fixed N. Qwen was "
        "chosen after inspecting earlier results, so this focus is exploratory.", "",
    ]
    for reward in FOCUS_REWARDS:
        lines += [
            f"## {REWARD_LABELS[reward]}", "",
            "| $/M tokens | Fixed N | Fixed quality | Adaptive quality | Fixed cost ($) | Adaptive cost ($) | Fixed net utility | Adaptive net utility | Relative net utility gain | Direct cost saving |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in tables[reward]:
            lines.append("| " + " | ".join((
                f"{r['price_per_million']:g}", f"{r['fixed_n']:.1f}",
                f"{r['fixed_quality']:.4f}", f"{r['adaptive_quality']:.4f}",
                f"{r['fixed_cost_dollars']:.5f}", f"{r['adaptive_cost_dollars']:.5f}",
                f"{r['fixed_net_utility']:.4f}", f"{r['adaptive_net_utility']:.4f}",
                f"{r['relative_net_utility_gain_percent']:+.2f}%",
                f"{r['direct_cost_saving_percent']:+.1f}%")) + " |")
        lines += ["", f"![{REWARD_LABELS[reward]} net utility gain]({reward}_net_utility_gain.png)", ""]
    lines += [
        "Bands are pointwise 95% paired prompt-bootstrap intervals conditional on "
        "the fixed N selected from training, the frozen DMRL policy, and the "
        "exploratory generator choice. They are not simultaneous intervals.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite output")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    source = json.loads((args.study / "METHOD.json").read_text())
    prices = np.asarray(source["prices_per_million_output_tokens"])
    np.testing.assert_array_equal(prices[:len(PRICES_PER_MILLION)], PRICES_PER_MILLION)
    assert tuple(source["split_seeds"]) == tuple(SPLIT_SEEDS)
    args.output.mkdir(parents=True)
    weights = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        100, np.full(100, .01), size=args.bootstrap).astype(float)
    rows, bands, hashes = [], [], {}
    ids_reference = None
    for model in MODELS:
        for reward in FOCUS_REWARDS:
            path = args.study / f"{model}__{reward}.npz"
            with np.load(path) as data:
                ids = data["ids"]
                adaptive = data["adaptive"][:, :len(PRICES_PER_MILLION), :]
                fixed = data["fixed"]
            if ids_reference is None:
                ids_reference = ids
            else:
                np.testing.assert_array_equal(ids_reference, ids)
            if adaptive.shape != (100, 7, 3) or fixed.shape != (100, 960, 3):
                raise ValueError(f"Unexpected cache shape: {path}")
            if not np.isfinite(adaptive).all() or not np.isfinite(fixed).all():
                raise ValueError(f"Nonfinite cached values: {path}")
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            new_rows, new_bands = evaluate_pair(reward, model, adaptive, fixed, weights)
            rows.extend(new_rows)
            bands.extend(new_bands)
    tables = focused_tables(rows)
    write_csv(args.output / "split_results.csv", rows)
    write_csv(args.output / "net_utility_intervals.csv", bands)
    for reward in FOCUS_REWARDS:
        write_csv(args.output / f"table_{reward}.csv", tables[reward])
    draw_figures(bands, args.output)
    write_report(tables, args.output)
    method = dict(source_study=str(args.study.resolve()), source_cache_sha256=hashes,
                  generator_models=MODELS, reward_models=FOCUS_REWARDS,
                  table_generator=FOCUS_GENERATOR,
                  prices_per_million_output_tokens=PRICES_PER_MILLION.tolist(),
                  policy=source["policy"], split_seeds=SPLIT_SEEDS,
                  bootstrap_seed=BOOTSTRAP_SEED, bootstrap_repetitions=args.bootstrap,
                  fixed_baseline="one N selected to maximize training mean net utility per model/reward/split/price, then frozen on test",
                  metric="net utility = quality minus recorded output-token cost",
                  interval="pointwise paired prompt bootstrap conditional on training-selected N",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "METHOD.json").write_text(json.dumps(method, indent=2) + "\n")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
