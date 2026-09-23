# Server handoff: adaptive alignment

Start with [the standalone alignment algorithm](algorithm/ADAPTIVE_ALIGNMENT.md)
and [its implementation](algorithm/adaptive_alignment.py). This branch also
includes the development/replay scripts, saved positive and negative results,
and the five-generator figures. The manuscript is not included in this commit.
Coding policies and results have not been changed.

## Fetch and test

~~~sh
git fetch origin
git switch --track origin/dmrl-alignment-server-20260923
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r codex_results/distribution_free_dmrl/requirements-server.txt
python -m unittest discover -s codex_results/distribution_free_dmrl -v
~~~

Python 3.11 or 3.12 is recommended for the replay dependencies. The lightweight
requirements do not install torch or vLLM. Keep your GPU generation/scoring
environment separate; this handoff does not run generation or change those models.
The standalone stopping class itself needs only Python 3.10+.

## Replay the historical alignment experiment

Run from the repository root. Use a NEW output directory: runners refuse to
overwrite previous results. The tracked Alpaca caches contain the five
generators and both evaluated reward fields.

~~~sh
python codex_results/distribution_free_dmrl/two_reward_models.py \
  --data-dir dataset/alpaca --reward-key fsfairx_rm_reward \
  --length-key text_chars --length-unit characters \
  --replay-seed 20260923 --output server_runs/alpaca_fsfairx_characters

python codex_results/distribution_free_dmrl/two_reward_models.py \
  --data-dir dataset/alpaca --reward-key mistral_rm_reward \
  --length-key text_chars --length-unit characters \
  --replay-seed 20260923 --output server_runs/alpaca_mistral_characters
~~~

The explicit seed reproduces the manuscript's streams; do not rely on the
runner's older default. The primary method is mean_costse2. The earlier Mistral
artifact calls this same policy odds_mean_costse2.
The frozen runner also includes the previous rule and two smoothing ablations.
Each run uses eight orderings, cap 960 (or the smallest available pool), and
five 50/50 splits with seeds 71–75. Only the fixed-N baseline is train-selected.

## Replay newly generated, token-counted responses

Store one JSON object per prompt, optionally gzip-compressed:

~~~json
{"JSON_idx": 0, "generations": [
  {"text": "response text", "token_count": 123,
   "fsfairx_rm_reward": 1.2, "mistral_rm_reward": 0.7}
]}
~~~

The example shows one response; real input needs at least four per prompt,
at least two prompts, finite rewards, and positive integer token counts.
Use the same five filenames:

- gemma2_9b_output.merged_rm.jsonl.gz
- llama3.1_8b_output.merged_rm.jsonl.gz
- llama3.2_3b_output.merged_rm.jsonl.gz
- mistral_7b_output.merged_rm.jsonl.gz
- qwen2.5_7b_output.merged_rm.jsonl.gz

Then run, changing the data directory to your new cache location:

~~~sh
python codex_results/distribution_free_dmrl/two_reward_models.py \
  --data-dir /path/to/token_counted/alpaca --reward-key fsfairx_rm_reward \
  --length-key token_count --length-unit tokens \
  --replay-seed 20260923 --output server_runs/alpaca_fsfairx_tokens

python codex_results/distribution_free_dmrl/two_reward_models.py \
  --data-dir /path/to/token_counted/alpaca --reward-key mistral_rm_reward \
  --length-key token_count --length-unit tokens \
  --replay-seed 20260923 --output server_runs/alpaca_mistral_tokens
~~~

There is no character-to-token conversion or fallback for missing counts.
METHOD.json records the chosen length field, units, seeds, policy settings,
and input/source hashes. Missing/invalid data fails explicitly. A failure after
starting can leave partial artifacts; use another output directory for a retry.

## Outputs, audits, and figures

Each replay produces METHOD.json, per-generator NPZ arrays, profit.csv, and
frontier_diagnostic.csv. Profit compares with train-selected fixed N and also
reports test-oracle fixed N. Matched-quality costs minimize over fixed-count
mixtures chosen on test data. Requested-target frontiers choose both methods'
mixtures retrospectively and flag infeasible comparisons.

Current manuscript artifacts are:

- FSFairX: codex_results/distribution_free_dmrl/frozen_alpaca_fsfairx/
- Mistral: codex_results/distribution_free_dmrl/online_cost_optimism_results/
- Figure: codex_results/distribution_free_dmrl/fsfairx_generator_lines/paper_bands/

Regenerate FSFairX intervals and the side-by-side band figure for a NEW run:

~~~sh
python codex_results/distribution_free_dmrl/plot_fsfairx_generator_lines.py \
  --study server_runs/alpaca_fsfairx_tokens \
  --output server_runs/fsfairx_token_intervals --bootstrap 1000
python codex_results/distribution_free_dmrl/plot_fsfairx_paper_bands.py \
  --source server_runs/fsfairx_token_intervals \
  --output server_runs/fsfairx_token_figure
~~~

Bootstrap intervals condition on chosen policy settings and baseline counts;
they are pointwise, not simultaneous or adjusted for development-time selection.
The plot scripts preserve the actual measurement units in their metadata.

The original development code and immutable result artifacts are retained.
New runner/plot source hashes will differ from historical manifests because
this handoff adds explicit length-field and output-path options; the default
policy mathematics is unchanged and tested for equivalence.
