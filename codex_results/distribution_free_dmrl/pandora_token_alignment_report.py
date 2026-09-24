"""Replay the earlier alignment Pandora policy on token-counted records.

Net utility means reward-based quality minus paid generation cost. The fixed-N
baseline maximizes training mean net utility, freezes that N, and uses it on
held-out prompts. Tables focus on Qwen 3.5 9B; figures include four generators.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALIGNMENT_CODE = Path(__file__).resolve().parents[1] / "code" / "alignment_scripts"
sys.path.insert(0, str(ALIGNMENT_CODE))
from alignment_pandora_ucb import pandora_stop_many  # noqa: E402
from alignment_simplification_study import (  # noqa: E402
    VARIANT_BY_NAME, prefix_only_placeholder_prior,
)
from replay_token_reward_farm import (  # noqa: E402
    CAP, MODELS, PERMUTATIONS, REPLAY_SEED, REWARDS, SPLIT_SEEDS,
    load_model, sigmoid,
)


FOCUS_REWARDS = ("fsfairx_llama3_rm", "rm_mistral_7b")
FOCUS_GENERATOR = "qwen35_9b"
PRICES_PER_MILLION = np.array([.1, .2, .5, 1., 2., 5., 10.])
LABELS = {"gemma3_4b": "Gemma 3 4B", "granite42_8b": "Granite 4.2 8B",
          "ministral3_8b": "Ministral 3 8B", "qwen35_9b": "Qwen 3.5 9B"}
REWARD_LABELS = {"fsfairx_llama3_rm": "FSFairX Llama 3 RM",
                 "rm_mistral_7b": "RM Mistral 7B"}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
BOOTSTRAP_SEED = 20260925
POLICY = VARIANT_BY_NAME["local_exp_open5_conf06"]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay_policy(rewards, lengths, fixed):
    """Use the same prompt/repetition response orders as the saved fixed cache."""
    prices = PRICES_PER_MILLION / 1e6
    divisors = tuple(1 / price for price in prices)
    adaptive = np.zeros((len(rewards), len(prices), 3), dtype=float)
    prior = prefix_only_placeholder_prior()
    rng = np.random.default_rng(REPLAY_SEED)
    first_fixed = np.zeros((CAP, 3), dtype=float)
    for i in range(len(rewards)):
        reference = float(np.sort(rewards[i])[int(.99 * CAP)])
        for _ in range(PERMUTATIONS):
            order = rng.permutation(CAP)
            r, length = rewards[i, order], lengths[i, order]
            if i == 0:
                first_fixed += np.stack((sigmoid(np.maximum.accumulate(r) - reference),
                                         np.cumsum(length), np.arange(1, CAP + 1)), axis=1)
            stops = pandora_stop_many(r, length, divisors, 1.0, prior,
                                      POLICY.config, POLICY.min_open)
            for pi, divisor in enumerate(divisors):
                opened, best, paid_tokens = stops[divisor]
                adaptive[i, pi] += (float(sigmoid(best - reference)), paid_tokens, opened)
        if (i + 1) % 25 == 0:
            print("  prompts", i + 1, "/", len(rewards), flush=True)
    np.testing.assert_allclose(first_fixed / PERMUTATIONS, fixed[0], rtol=0, atol=1e-11)
    return adaptive / PERMUTATIONS


def evaluate_pair(reward, model, adaptive, fixed, weights):
    prices = PRICES_PER_MILLION / 1e6
    split_rows = []
    point = np.zeros((len(SPLIT_SEEDS), len(prices)))
    boot = np.zeros((len(weights), len(prices)))
    for si, seed in enumerate(SPLIT_SEEDS):
        train, test = np.array_split(np.random.default_rng(seed).permutation(len(fixed)), 2)
        training_curve = fixed[train].mean(axis=0)
        selected_n = np.argmax(training_curve[:, 0, None] -
                               training_curve[:, 1, None] * prices[None, :], axis=0)
        aq, at, calls = (adaptive[test, :, j] for j in range(3))
        fq = fixed[test][:, selected_n, 0]
        ft = fixed[test][:, selected_n, 1]
        adaptive_net = aq - at * prices
        fixed_net = fq - ft * prices
        a_mean, f_mean = adaptive_net.mean(axis=0), fixed_net.mean(axis=0)
        if np.any(f_mean <= 0):
            raise ValueError(f"Nonpositive fixed-N net utility: {model}, {reward}, {seed}")
        point[si] = 100 * (a_mean - f_mean) / f_mean
        w = weights[:, test].copy()
        w /= w.sum(axis=1, keepdims=True)
        a_boot, f_boot = w @ adaptive_net, w @ fixed_net
        if np.any(f_boot <= 0):
            raise ValueError(f"Nonpositive bootstrap fixed-N net utility: {model}, {reward}")
        boot += 100 * (a_boot - f_boot) / f_boot
        for pi, price in enumerate(PRICES_PER_MILLION):
            q_a, q_f = float(aq[:, pi].mean()), float(fq[:, pi].mean())
            tokens_a, tokens_f = float(at[:, pi].mean()), float(ft[:, pi].mean())
            row = dict(reward_model=reward, generator=model, split_seed=seed,
                       price_per_million=float(price), fixed_n=int(selected_n[pi]) + 1,
                       fixed_quality=q_f, adaptive_quality=q_a,
                       fixed_cost_dollars=float(prices[pi] * tokens_f),
                       adaptive_cost_dollars=float(prices[pi] * tokens_a),
                       fixed_net_utility=float(f_mean[pi]),
                       adaptive_net_utility=float(a_mean[pi]),
                       net_utility_gain=float(a_mean[pi] - f_mean[pi]),
                       relative_net_utility_gain_percent=float(point[si, pi]),
                       fixed_output_tokens=tokens_f, adaptive_output_tokens=tokens_a,
                       adaptive_calls=float(calls[:, pi].mean()),
                       direct_cost_saving_percent=float(100 * (1 - tokens_a / tokens_f)))
            split_rows.append(row)
    boot /= len(SPLIT_SEEDS)
    lower, upper = np.quantile(boot, [.025, .975], axis=0)
    intervals = [dict(reward_model=reward, generator=model,
                      price_per_million=float(price),
                      relative_net_utility_gain_percent=float(point[:, pi].mean()),
                      ci_low=float(lower[pi]), ci_high=float(upper[pi]),
                      confidence=.95, bootstrap_repetitions=len(weights))
                 for pi, price in enumerate(PRICES_PER_MILLION)]
    return split_rows, intervals


def focused_tables(rows):
    tables = {}
    for reward in FOCUS_REWARDS:
        tables[reward] = []
        for price in PRICES_PER_MILLION:
            group = [r for r in rows if r["reward_model"] == reward and
                     r["generator"] == FOCUS_GENERATOR and
                     r["price_per_million"] == float(price)]
            assert len(group) == len(SPLIT_SEEDS)
            row = dict(reward_model=reward, generator=FOCUS_GENERATOR,
                       price_per_million=float(price))
            for key in ("fixed_n", "fixed_quality", "adaptive_quality",
                        "fixed_cost_dollars", "adaptive_cost_dollars",
                        "fixed_net_utility", "adaptive_net_utility", "net_utility_gain",
                        "relative_net_utility_gain_percent", "fixed_output_tokens",
                        "adaptive_output_tokens", "adaptive_calls", "direct_cost_saving_percent"):
                row[key] = float(np.mean([r[key] for r in group]))
            tables[reward].append(row)
    return tables


def draw_figures(intervals, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "savefig.dpi": 220})
    for reward in FOCUS_REWARDS:
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        for mi, model in enumerate(MODELS):
            group = sorted((r for r in intervals if r["reward_model"] == reward and
                            r["generator"] == model), key=lambda r: r["price_per_million"])
            x = np.array([r["price_per_million"] for r in group])
            y = np.array([r["relative_net_utility_gain_percent"] for r in group])
            lo = np.array([r["ci_low"] for r in group])
            hi = np.array([r["ci_high"] for r in group])
            ax.plot(x, y, marker="o", linewidth=2, color=COLORS[mi], label=LABELS[model])
            ax.fill_between(x, lo, hi, color=COLORS[mi], alpha=.14, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks(PRICES_PER_MILLION, [f"{p:g}" for p in PRICES_PER_MILLION])
        ax.axhline(0, color="#555555", linewidth=.9)
        ax.grid(axis="y", color="#dce2e8", linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Generation price ($ per million output tokens)")
        ax.set_ylabel("Net utility gain over train-selected fixed N (%)")
        ax.set_title(REWARD_LABELS[reward], fontsize=15, pad=12)
        ax.legend(ncol=2, frameon=False, title="Generator", loc="best")
        fig.subplots_adjust(left=.11, right=.97, top=.90, bottom=.18)
        fig.text(.11, .04, "Bands: pointwise 95% paired prompt bootstrap; five 50/50 splits and eight replay orders.",
                 fontsize=8.5, color="#444444")
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{reward}_net_utility_gain.{extension}", facecolor="white")
        plt.close(fig)


def write_report(tables, output):
    lines = ["# Token-count Pandora alignment utility", "",
             "This run uses the earlier training-free `local_exp_open5_conf06` Pandora tail policy "
             "on the new token-counted responses. Qwen 3.5 9B is held fixed in both tables; "
             "the two figures show all four generators. Prices are $0.1–$10 per million "
             "recorded output tokens.", "",
             "**Correction:** the previous focused report labeled pre-cost reward quality as "
             "utility. Here **net utility = quality − generation cost**, matching the earlier "
             "alignment experiment's objective. The earlier 2–10% utility result used this "
             "Pandora policy, whereas the DMRL rule is a different algorithm.", "",
             "For every generator, reward model, split, and price, training prompts select the "
             "single integer fixed N that maximizes mean net utility. That N is frozen on "
             "held-out prompts. The same eight response orders are used for adaptive and fixed N. "
             "Values average five 50/50 split results. Percent gains are averages of splitwise "
             "relative net utility gains; they need not equal a ratio of displayed means.", "",
             "Quality is sigmoid(selected reward minus the prompt's full-pool reward q99). "
             "Generation cost uses actual recorded output tokens. Reward scoring and input-token "
             "costs are excluded. Prices are illustrative and quality is a reward-model proxy. "
             "Qwen was chosen after inspecting prior results, so its selection is exploratory.", ""]
    for reward in FOCUS_REWARDS:
        lines += [f"## {REWARD_LABELS[reward]}", "",
                  "| $/M tokens | Fixed N | Fixed quality | Adaptive quality | Fixed cost ($) | Adaptive cost ($) | Fixed net utility | Adaptive net utility | Relative net utility gain | Direct cost saving |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in tables[reward]:
            lines.append("| " + " | ".join((
                f"{r['price_per_million']:g}", f"{r['fixed_n']:.1f}",
                f"{r['fixed_quality']:.4f}", f"{r['adaptive_quality']:.4f}",
                f"{r['fixed_cost_dollars']:.5f}", f"{r['adaptive_cost_dollars']:.5f}",
                f"{r['fixed_net_utility']:.4f}", f"{r['adaptive_net_utility']:.4f}",
                f"{r['relative_net_utility_gain_percent']:+.2f}%",
                f"{r['direct_cost_saving_percent']:+.1f}%")) + " |")
        lines += ["", f"![{REWARD_LABELS[reward]} net utility gain]({reward}_net_utility_gain.png)", ""]
    lines += ["The pointwise 95% paired prompt-bootstrap bands condition on the frozen policy, "
              "the training-selected fixed Ns, and the exploratory Qwen choice. They are not "
              "simultaneous or corrected for development-time selection.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(
        "/scratch1/kalayci/alignment_generation_farm/reward_run/data/rewarded_records"))
    parser.add_argument("--fixed-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite output")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    args.output.mkdir(parents=True)
    weights = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        100, np.full(100, .01), size=args.bootstrap).astype(float)
    split_rows, intervals = [], []
    signatures = set()
    source_hashes = {}
    ids_reference = None
    for model in MODELS:
        ids, rewards, lengths, signature = load_model(args.input_root, model)
        signatures.add(signature)
        if ids_reference is None:
            ids_reference = ids
        elif ids != ids_reference:
            raise ValueError("Prompt order differs across generators")
        for reward in FOCUS_REWARDS:
            ri = REWARDS.index(reward)
            cache = args.fixed_study / f"{model}__{reward}.npz"
            with np.load(cache) as data:
                np.testing.assert_array_equal(ids, data["ids"])
                fixed = data["fixed"]
            source_hashes[cache.name] = hashlib.sha256(cache.read_bytes()).hexdigest()
            print("Replaying Pandora", model, reward, flush=True)
            adaptive = replay_policy(rewards[:, :, ri], lengths, fixed)
            np.savez_compressed(args.output / cache.name, adaptive=adaptive, fixed=fixed, ids=ids)
            rows, bands = evaluate_pair(reward, model, adaptive, fixed, weights)
            split_rows.extend(rows)
            intervals.extend(bands)
    if len(signatures) != 1:
        raise ValueError("Mixed reward configurations")
    tables = focused_tables(split_rows)
    write_csv(args.output / "split_results.csv", split_rows)
    write_csv(args.output / "net_utility_intervals.csv", intervals)
    for reward in FOCUS_REWARDS:
        write_csv(args.output / f"table_{reward}.csv", tables[reward])
    draw_figures(intervals, args.output)
    write_report(tables, args.output)
    method = dict(input_root=str(args.input_root.resolve()),
                  fixed_study=str(args.fixed_study.resolve()),
                  source_cache_sha256=source_hashes,
                  reward_config_signature=signatures.pop(),
                  generator_models=MODELS, reward_models=FOCUS_REWARDS,
                  table_generator=FOCUS_GENERATOR,
                  prices_per_million_output_tokens=PRICES_PER_MILLION.tolist(),
                  policy=POLICY.name, policy_config=POLICY.config.__dict__,
                  minimum_open=POLICY.min_open,
                  replay_seed=REPLAY_SEED, permutations=PERMUTATIONS,
                  split_seeds=SPLIT_SEEDS, bootstrap_seed=BOOTSTRAP_SEED,
                  bootstrap_repetitions=args.bootstrap,
                  metric="net utility = sigmoid(selected reward - prompt full-pool q99) - price * paid output tokens",
                  fixed_baseline="single integer N selected to maximize training mean net utility for each generator/reward/split/price and frozen on test",
                  interval="pointwise paired prompt bootstrap, fixed N held frozen, shared weights across generators/splits",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "METHOD.json").write_text(json.dumps(method, indent=2) + "\n")
    print("Wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
