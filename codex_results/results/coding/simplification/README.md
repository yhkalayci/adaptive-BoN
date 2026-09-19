# Coding simplification results

The manuscript uses exactly these folders:

- `final_confirmation_reservation/`: selected utility policy
  `global_exp_q75_conf08_085x` on final split IDs 75--84;
- `target_development/`: frozen one-policy target inverse profile from
  development IDs 55--59; and
- `target_final_profile_only/`: target confirmation on IDs 95--104 with zero
  current-split blend.

The `development_*` folders document the policy and single-multiplier
selection. Other `final_confirmation*` folders are paired diagnostics created
before the final profile-only selector was frozen; they are not manuscript
results.

Reproduction scripts are under `codex_results/code/coding_scripts/`.
