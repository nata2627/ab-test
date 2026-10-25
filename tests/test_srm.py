import numpy as np

from src.srm import srm_test


def test_balanced_split_passes():
    rng = np.random.default_rng(1)
    n_t = rng.binomial(1_000_000, 0.85)
    assert srm_test(n_t, 1_000_000 - n_t, 0.85)["p_value"] > 0.001


def test_skewed_split_fails():
    assert srm_test(851_000, 149_000, 0.85)["p_value"] < 0.01


def test_matches_manual_chi2():
    res = srm_test(860, 140, 0.85)
    expected = (860 - 850) ** 2 / 850 + (140 - 150) ** 2 / 150
    assert abs(res["chi2"] - expected) < 1e-9


def test_too_close_flags_exact_hit():
    assert srm_test(8_500_000, 1_500_000, 0.85)["p_too_close"] < 0.01
    assert srm_test(8_501_200, 1_498_800, 0.85)["p_too_close"] > 0.5
