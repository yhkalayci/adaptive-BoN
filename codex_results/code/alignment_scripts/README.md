# Alignment experiments

The manuscript's selected alignment policy is the training-free
`local_exp_open5_conf06` rule in `alignment_simplification_study.py`. It opens
five responses, fits the inclusive upper half of exponentiated scores on the
current prompt, applies confidence multiplier 0.6 and EI bonus 0.002, and uses
the running mean response length. No fitted reward, benchmark, or cost prior
can affect its stopping decision.

Reproduce its ten-split utility, matched-cost win-rate, and target-quality
confirmation with:

```bash
bash codex_results/code/alignment_scripts/run_alignment_simple_final.sh
```

Results are written to:

```text
codex_results/results/alignment/simplification/final_confirmation/
```

The run includes the former training-shrunk policy on the exact same splits
and response permutations solely as a paired simplification reference.
`alignment_pandora_ucb.py` contains the stopping implementation, while
`alignment_pandora_all_generators.py` supplies shared evaluation machinery.

Run the focused tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 /home1/kalayci/env/bin/python \
  -m unittest discover -s codex_results/code/alignment_scripts -p 'test_*.py'
```
