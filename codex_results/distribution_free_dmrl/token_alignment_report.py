"""Complete the token-count alignment replay and make four comparison tables/figures.

The replay policy is the frozen AdaptiveAlignment mean_costse2 rule. Existing
partial caches may be reused only after their IDs, shapes, and a deterministic
one-prompt replay agree with the source records. All model and fixed-N choices
in the tables use training prompts for each split and price.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_refinement import interpolated_cost
from replay_token_reward_farm import (
    CAP, MODELS, PERMUTATIONS, PRICES_PER_MILLION, REPLAY_SEED, REWARDS,
    SPLIT_SEEDS, load_model, replay, summarize,
)


LABELS = {
    "gemma3_4b": "Gemma 3 4B",
    "granite42_8b": "Granite 4.2 8B",
    "ministral3_8b": "Ministral 3 8B",
    "qwen35_9b": "Qwen 3.5 9B",
}
REWARD_LABELS = {
    "armorm_llama3_8b": "ArmoRM Llama 3 8B",
    "fsfairx_llama3_rm": "FSFairX Llama 3 RM",
    "rm_mistral_7b": "RM Mistral 7B",
    "skywork_llama31_8b": "Skywork Llama 3.1 8B",
}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
BOOTSTRAP_SEED = 20260924


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_or_replay(cache: Path | None, destination: Path, ids, scores, lengths):
    if cache is not None and cache.exists():
        with np.load(cache) as data:
            adaptive, fixed = data["adaptive"], data["fixed"]
            np.testing.assert_array_equal(data["ids"], ids)
        if adaptive.shape != (100, len(PRICES_PER_MILLION), 3) or fixed.shape != (100, CAP, 3):
            raise ValueError(f"Unexpected cache dimensions: {cache}")
        if not np.isfinite(adaptive).all() or not np.isfinite(fixed).all():
            raise ValueError(f"Nonfinite cached values: {cache}")
        check_adaptive, check_fixed = replay(scores[:1], lengths[:1])
        np.testing.assert_allclose(adaptive[:1], check_adaptive, rtol=0, atol=1e-11)
        np.testing.assert_allclose(fixed[:1], check_fixed, rtol=0, atol=1e-11)
        if cache.resolve() != destination.resolve():
            shutil.copyfile(cache, destination)
        origin = "validated prior replay"
    else:
        adaptive, fixed = replay(scores, lengths)
        np.savez_compressed(destination, adaptive=adaptive, fixed=fixed, ids=ids)
        origin = "new replay"
    return adaptive, fixed, origin


def bootstrap_profit(rows_by_pair, bootstrap: int):
    """Pointwise 95% prompt intervals, shared across generators and splits.

    Fixed-N selections are frozen after training. The same resampled prompt
    weights are used for all five overlapping splits and all generators.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(100, np.full(100, .01), size=bootstrap).astype(float)
    prices = PRICES_PER_MILLION / 1e6
    intervals = []
    split_indices = [np.array_split(np.random.default_rng(seed).permutation(100), 2)
                     for seed in SPLIT_SEEDS]
    for reward in REWARDS:
        for model in MODELS:
            adaptive, fixed = rows_by_pair[reward, model]
            boot = np.zeros((bootstrap, len(prices)))
            split_point = np.zeros((len(SPLIT_SEEDS), len(prices)))
            for si, (train, test) in enumerate(split_indices):
                training_fixed = fixed[train].mean(axis=0)
                selected_n = np.argmax(
                    training_fixed[:, 0, None] - training_fixed[:, 1, None] * prices,
                    axis=0,
                )
                aq = adaptive[test, :, 0]
                ac = adaptive[test, :, 1]
                fq = fixed[test][:, selected_n, 0]
                fc = fixed[test][:, selected_n, 1]
                a_profit = aq - ac * prices
                f_profit = fq - fc * prices
                point_denominator = f_profit.mean(axis=0)
                if np.any(point_denominator <= 0):
                    raise ValueError("Relative profit requires a positive fixed-N profit")
                split_point[si] = 100 * (a_profit.mean(axis=0) - point_denominator) / point_denominator
                w = weights[:, test].copy()
                w /= w.sum(axis=1, keepdims=True)
                # Freeze the positive observed baseline denominator. At high
                # prices a few bootstrap baselines cross zero; dividing by
                # those resampled values makes percentage bands singular.
                boot += 100 * ((w @ a_profit) - (w @ f_profit)) / point_denominator
            boot /= len(SPLIT_SEEDS)
            mean = split_point.mean(axis=0)
            lo, hi = np.quantile(boot, [.025, .975], axis=0)
            for pi, price in enumerate(PRICES_PER_MILLION):
                intervals.append(dict(reward_model=reward, model=model,
                                      price_per_million=float(price),
                                      profit_improvement_percent=float(mean[pi]),
                                      ci_low=float(lo[pi]), ci_high=float(hi[pi]),
                                      confidence=.95, bootstrap_repetitions=bootstrap))
    return intervals


def selected_tables(rows_by_pair):
    """Choose the generator and fixed N on each split's training prompts."""
    tables = {reward: [] for reward in REWARDS}
    selection_rows = []
    for reward in REWARDS:
        for pi, price_per_million in enumerate(PRICES_PER_MILLION):
            price = price_per_million / 1e6
            chosen = []
            for seed in SPLIT_SEEDS:
                train, test = np.array_split(np.random.default_rng(seed).permutation(100), 2)
                train_profit = []
                for model in MODELS:
                    adaptive, _ = rows_by_pair[reward, model]
                    train_profit.append(float(np.mean(adaptive[train, pi, 0]
                                                      - price * adaptive[train, pi, 1])))
                model = MODELS[int(np.argmax(train_profit))]
                adaptive, fixed = rows_by_pair[reward, model]
                train_fixed = fixed[train].mean(axis=0)
                n = int(np.argmax(train_fixed[:, 0] - price * train_fixed[:, 1]))
                aq, at = adaptive[test, pi, :2].mean(axis=0)
                fq, ft = fixed[test, n, :2].mean(axis=0)
                matched_tokens = float(interpolated_cost(
                    fixed[test, :, 0].mean(axis=0),
                    fixed[test, :, 1].mean(axis=0), np.array([aq]))[0])
                ap, fp = aq - price * at, fq - price * ft
                row = dict(reward_model=reward, price_per_million=float(price_per_million),
                           split_seed=seed, selected_model=model, fixed_n=n + 1,
                           adaptive_utility=float(aq), fixed_utility=float(fq),
                           adaptive_output_tokens=float(at), fixed_output_tokens=float(ft),
                           adaptive_cost_dollars=float(price * at),
                           fixed_cost_dollars=float(price * ft),
                           adaptive_profit=float(ap), fixed_profit=float(fp),
                           profit_improvement_percent=float(100 * (ap - fp) / fp),
                           direct_cost_saving_percent=float(100 * (1 - at / ft)),
                           matched_quality_cost_saving_percent=float(100 * (1 - at / matched_tokens)))
                selection_rows.append(row)
                chosen.append(row)
            counts = {model: sum(r["selected_model"] == model for r in chosen) for model in MODELS}
            winner = max(MODELS, key=lambda model: (counts[model], -MODELS.index(model)))
            table_row = dict(reward_model=reward, price_per_million=float(price_per_million),
                             most_selected_model=winner,
                             model_selection_counts=json.dumps(counts, sort_keys=True))
            for key in ("fixed_n", "adaptive_utility", "fixed_utility",
                        "adaptive_output_tokens", "fixed_output_tokens",
                        "adaptive_cost_dollars", "fixed_cost_dollars",
                        "adaptive_profit", "fixed_profit", "profit_improvement_percent",
                        "direct_cost_saving_percent", "matched_quality_cost_saving_percent"):
                table_row[key] = float(np.mean([r[key] for r in chosen]))
            tables[reward].append(table_row)
    return tables, selection_rows


def draw_figures(intervals, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "savefig.dpi": 220})
    for reward in REWARDS:
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        for mi, model in enumerate(MODELS):
            group = sorted((r for r in intervals if r["reward_model"] == reward
                            and r["model"] == model), key=lambda r: r["price_per_million"])
            x = np.array([r["price_per_million"] for r in group])
            y = np.array([r["profit_improvement_percent"] for r in group])
            lo = np.array([r["ci_low"] for r in group])
            hi = np.array([r["ci_high"] for r in group])
            ax.plot(x, y, marker="o", linewidth=2, color=COLORS[mi], label=LABELS[model])
            ax.fill_between(x, lo, hi, color=COLORS[mi], alpha=.14, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks(PRICES_PER_MILLION, [f"{p:g}" for p in PRICES_PER_MILLION])
        if reward == "skywork_llama31_8b":
            # The fixed-N profit approaches zero at $10/M for some generators.
            # Compress the resulting large relative percentages while keeping
            # losses through -50% in the linear part of the scale.
            ax.set_yscale("symlog", linthresh=50)
            ax.set_yticks([-50, -40, -20, 0, 20, 50, 100, 200, 400],
                          ["-50", "-40", "-20", "0", "20", "50", "100", "200", "400"])
        ax.axhline(0, color="#555555", linewidth=.9)
        ax.grid(axis="y", color="#dce2e8", linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Generation price ($ per million output tokens)")
        ax.set_ylabel("Profit improvement over train-selected fixed N (%)")
        ax.set_title(REWARD_LABELS[reward], fontsize=15, pad=12)
        ax.legend(ncol=2, frameon=False, title="Generator", loc="best")
        fig.subplots_adjust(left=.11, right=.97, top=.90, bottom=.18)
        note = "Bands: pointwise 95% paired prompt bootstrap; five splits, eight replay orders per prompt."
        if reward == "skywork_llama31_8b":
            note += " Vertical scale is symmetric log (linear within ±50%)."
        fig.text(.11, .04, note,
                 fontsize=8.5, color="#444444")
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{reward}_profit_improvement.{extension}", facecolor="white")
        plt.close(fig)


def fmt_cost(value):
    return f"{value:.5f}" if value >= .00001 else f"{value:.2e}"


def write_markdown(tables, output):
    lines = ["# Token-count adaptive alignment replay", "",
             "Four generators, four reward models, 100 Alpaca prompts per generator, "
             "960 cached responses per prompt, eight seeded response orders, and five 50/50 splits.", "",
             "Utility is sigmoid(selected reward − that prompt's full-pool reward q99). "
             "Generation cost is recorded output tokens × the stated price; reward scoring cost is excluded. "
             "Profit is utility minus generation cost. Prices are illustrative rather than "
             "model-specific billed rates, and utility is a reward-model proxy rather than a human rating.", "",
             "Each table selects the generator with highest adaptive training profit within each split "
             "and price. It then selects the fixed N with highest training profit for that generator. "
             "The generator column reports the most frequent training selection; the numeric entries "
             "average the five held-out split results and can include other selected generators. "
             "Selection counts and unrounded values are in the CSV files.", "",
             "Direct cost saving compares paid generation costs and may reflect a utility difference. "
             "Matched-utility saving compares with a retrospective fixed-N mixture at the attained "
             "adaptive utility; that mixture is a diagnostic, not a deployable policy.", ""]
    for reward in REWARDS:
        lines += [f"## {REWARD_LABELS[reward]}", "",
                  "| $/M tokens | Most selected generator | Fixed N | Fixed utility | Adaptive utility | Fixed cost ($) | Adaptive cost ($) | Fixed profit | Adaptive profit | Profit gain | Direct cost saving | Matched-utility saving |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in tables[reward]:
            lines.append("| " + " | ".join((
                f"{r['price_per_million']:g}", LABELS[r["most_selected_model"]],
                f"{r['fixed_n']:.1f}", f"{r['fixed_utility']:.4f}",
                f"{r['adaptive_utility']:.4f}", fmt_cost(r["fixed_cost_dollars"]),
                fmt_cost(r["adaptive_cost_dollars"]), f"{r['fixed_profit']:.4f}",
                f"{r['adaptive_profit']:.4f}", f"{r['profit_improvement_percent']:+.2f}%",
                f"{r['direct_cost_saving_percent']:+.1f}%",
                f"{r['matched_quality_cost_saving_percent']:+.1f}%")) + " |")
        lines += ["", f"![{REWARD_LABELS[reward]} profit improvement]({reward}_profit_improvement.png)", ""]
    lines += ["Figure bands are pointwise 95% paired prompt-bootstrap percentile intervals "
              "for each generator's profit improvement against its own train-selected fixed N. "
              "They condition on the five splits, fixed-N choices, and the frozen adaptive rule; "
              "they are not simultaneous intervals or adjusted for policy development. "
              "Bootstrap profit differences are divided by the observed positive fixed-N profit "
              "to avoid singular resampled ratios near zero. Figures show all four generators, "
              "whereas tables use training-selected generators. Skywork uses a symmetric-log "
              "vertical scale because several fixed-N profits approach zero at $10/M tokens.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(
        "/scratch1/kalayci/alignment_generation_farm/reward_run/data/rewarded_records"))
    parser.add_argument("--reuse-cache", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--resume", action="store_true",
                        help="Refresh this script's output using its saved replay caches")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error("Refusing to overwrite an existing output directory")
    if args.resume and not args.output.is_dir():
        parser.error("--resume requires an existing output directory")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    rows_by_pair = {}
    signatures = set()
    ids_ref = None
    cache_manifest = {}
    pair_rows = []
    for model in MODELS:
        ids, scores, lengths, signature = load_model(args.input_root, model)
        if ids_ref is None:
            ids_ref = ids
        elif ids != ids_ref:
            raise ValueError("Prompt order differs across generators")
        signatures.add(signature)
        for ri, reward in enumerate(REWARDS):
            name = f"{model}__{reward}.npz"
            source_cache = None if args.reuse_cache is None else args.reuse_cache / name
            print("Replaying", model, reward, flush=True)
            adaptive, fixed, origin = load_or_replay(source_cache, args.output / name,
                                                      ids, scores[:, :, ri], lengths)
            rows_by_pair[reward, model] = adaptive, fixed
            cache_manifest[name] = dict(origin=origin,
                                        sha256=hashlib.sha256((args.output / name).read_bytes()).hexdigest())
            pair_rows.extend(summarize(model, reward, adaptive, fixed))
    if len(signatures) != 1:
        raise ValueError("Mixed reward configuration signatures across models")
    write_csv(args.output / "generator_split_results.csv", pair_rows)
    intervals = bootstrap_profit(rows_by_pair, args.bootstrap)
    write_csv(args.output / "profit_intervals.csv", intervals)
    tables, selections = selected_tables(rows_by_pair)
    write_csv(args.output / "training_selections.csv", selections)
    for reward in REWARDS:
        write_csv(args.output / f"table_{reward}.csv", tables[reward])
    draw_figures(intervals, args.output)
    write_markdown(tables, args.output)
    method = dict(input_root=str(args.input_root.resolve()),
                  reward_config_signature=signatures.pop(),
                  generator_models=MODELS, reward_models=REWARDS,
                  prompts_per_generator=100, responses_per_prompt=CAP,
                  output_token_field="generations[].output_tokens",
                  prices_per_million_output_tokens=PRICES_PER_MILLION.tolist(),
                  split_seeds=SPLIT_SEEDS, replay_seed=REPLAY_SEED,
                  permutations=PERMUTATIONS, bootstrap_seed=BOOTSTRAP_SEED,
                  bootstrap_repetitions=args.bootstrap,
                  policy="AdaptiveAlignment mean_costse2; min=4, q=.99, cost_adjustment=2",
                  utility="sigmoid(selected raw reward - prompt full-pool empirical q99)",
                  cost="recorded output tokens * price; reward scoring cost excluded",
                  price_interpretation="illustrative regimes, not generator-specific billed rates",
                  table_selection="generator and fixed N trained within each split/price; held-out evaluation",
                  figure_baseline="each generator compared with own train-selected fixed N",
                  intervals="pointwise paired prompt-bootstrap of profit differences divided by observed positive fixed-N profit; shared weights across splits and generators; conditional on training selections",
                  cache_manifest=cache_manifest,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "METHOD.json").write_text(json.dumps(method, indent=2) + "\n")
    print("Wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
