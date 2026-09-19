import numpy as np

from alignment_pandora_ucb import PandoraConfig, PriorFit, pandora_stop, pandora_stop_many


def _prior():
    return PriorFit(
        raw_sigma=1.0,
        gaussian_quantile_z=2.2,
        raw_tail_scales={0.5: 0.7},
        exp_tail_ratios={0.5: 0.8},
        raw_tail_quantile_k={0.5: 4.2},
        exp_tail_quantile_k={0.5: 4.5},
        cost_beta=np.zeros(6),
        feature_mean=np.zeros(5),
        feature_scale=np.ones(5),
    )


def test_reported_cost_is_exact_prefix_character_sum():
    rewards = np.array([0.0, 0.1, 0.2, 0.15, 0.3])
    chars = np.array([11.0, 13.0, 17.0, 19.0, 23.0])
    cfg = PandoraConfig("gaussian_raw", 0.2, 2.0, 2.0)
    opened, _, total_chars = pandora_stop(rewards, chars, 1e3, 20.0, _prior(), cfg, 3)
    assert total_chars == chars[:opened].sum()


def test_unseen_suffix_cannot_change_an_earlier_stop():
    rewards = np.array([4.0, 4.1, 9.0, 0.0, 0.0, 0.0])
    chars = np.full(6, 100.0)
    cfg = PandoraConfig("gaussian_raw", 0.1, 2.0, 2.0)
    first = pandora_stop(rewards, chars, 100.0, 100.0, _prior(), cfg, 3)
    changed = rewards.copy()
    changed[3:] = 1000.0
    second = pandora_stop(changed, chars, 100.0, 100.0, _prior(), cfg, 3)
    assert first[0] == 3
    assert second[0] == first[0]
    assert second[1] == first[1]


def test_all_supported_models_are_pandora_reservation_policies():
    rewards = np.array([0.0, 0.5, -0.2, 0.7, 0.1, 1.0])
    chars = np.full(6, 10.0)
    for model in (
        "gaussian_raw",
        "exp_tail_raw",
        "halfnormal_tail_raw",
        "rayleigh_tail_raw",
        "exp_tail_exponentiated",
    ):
        cfg = PandoraConfig(model, 0.2, 2.0, 2.0)
        opened, best, total_chars = pandora_stop(rewards, chars, 1e4, 10.0, _prior(), cfg, 3)
        assert 3 <= opened <= len(rewards)
        assert best == rewards[:opened].max()
        assert total_chars == 10.0 * opened


def test_shared_cost_sweep_matches_independent_pandora_runs():
    rng = np.random.default_rng(7)
    rewards = rng.normal(size=80)
    chars = rng.integers(20, 500, size=80).astype(float)
    divisors = (1e4, 5e4, 1e6)
    for model in (
        "exp_tail_raw",
        "halfnormal_tail_raw",
        "rayleigh_tail_raw",
        "exp_tail_exponentiated",
    ):
        cfg = PandoraConfig(model, 0.4, 5.0, 0.0, ei_bonus_scale=0.002)
        shared = pandora_stop_many(rewards, chars, divisors, 200.0, _prior(), cfg, 3)
        independent = {
            d: pandora_stop(rewards, chars, d, 200.0, _prior(), cfg, 3) for d in divisors
        }
        assert shared == independent
