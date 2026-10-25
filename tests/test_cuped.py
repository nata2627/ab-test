import numpy as np

from src.cuped import center_within_strata, compare
from src.effect import from_sums


def synthetic(rng, n, effect):
    s = rng.integers(0, 3, n)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    t = (rng.random(n) < np.array([0.8, 0.85, 0.9])[s]).astype(np.int8)
    p = 1 / (1 + np.exp(-(-3 + 0.3 * s + 1.2 * x[:, 0])))
    y = (rng.random(n) < p + effect * t).astype(np.int8)
    return y, t, s, center_within_strata(x, s)


def test_cuped_reduces_variance_without_shift():
    res = compare(*synthetic(np.random.default_rng(31), 400_000, 0.005))
    assert res["variance_reduction"] > 0.05
    assert abs(res["shift_in_se"]) < 1
    assert res["cuped"]["ci95"][0] < 0.005 < res["cuped"]["ci95"][1]


def test_cuped_covers_planted_effect():
    rng = np.random.default_rng(32)
    covered = 0
    for _ in range(200):
        lo, hi = compare(*synthetic(rng, 40_000, 0.01))["cuped"]["ci95"]
        covered += lo < 0.01 < hi
    # 95% при 200 повторах, sd около 1,5 п. п.
    assert 180 <= covered <= 197


def test_continuous_variance_matches_manual():
    rng = np.random.default_rng(33)
    a, b = rng.normal(1, 2, 5000), rng.normal(0, 3, 3000)
    res = from_sums([5000], [a.sum()], [3000], [b.sum()], [(a**2).sum()], [(b**2).sum()])
    se = np.sqrt(a.var() / 5000 + b.var() / 3000)
    assert abs(res["se"] - se) < 1e-12
    assert abs(res["diff"] - (a.mean() - b.mean())) < 1e-12


def test_shift_in_own_sd_is_standard_normal_under_balance():
    rng = np.random.default_rng(34)
    shifts = [compare(*synthetic(rng, 40_000, 0.0))["shift_in_own_sd"] for _ in range(200)]
    assert abs(np.std(shifts) - 1) < 0.15
