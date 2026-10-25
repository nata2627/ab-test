import numpy as np
from statsmodels.stats import proportion

from src.effect import from_sums, ipw_diff, iv_wald, naive_diff, stratified_diff

N_REPS = 2000


def test_naive_matches_statsmodels_wald():
    res = from_sums([400_000], [20_800], [100_000], [5_000])
    _, p = proportion.test_proportions_2indep(20_800, 400_000, 5_000, 100_000, method="wald")
    lo, hi = proportion.confint_proportions_2indep(20_800, 400_000, 5_000, 100_000, method="wald")
    assert abs(res["p_value"] - p) < 1e-9
    assert np.allclose(res["ci95"], [lo, hi], atol=1e-9)


def test_planted_effect_is_covered():
    rng = np.random.default_rng(11)
    n_t, n_c, p0, d = 85_000, 15_000, 0.05, 0.004
    x_t = rng.binomial(n_t, p0 + d, N_REPS)
    x_c = rng.binomial(n_c, p0, N_REPS)
    res = [from_sums([n_t], [a], [n_c], [b]) for a, b in zip(x_t, x_c, strict=True)]
    covered = np.mean([r["ci95"][0] < d < r["ci95"][1] for r in res])
    rel_covered = np.mean([r["rel_ci95"][0] < d / p0 < r["rel_ci95"][1] for r in res])
    assert abs(np.mean([r["diff"] for r in res]) - d) < 3 * res[0]["se"] / np.sqrt(N_REPS)
    assert 0.935 < covered < 0.965
    assert 0.935 < rel_covered < 0.965


def test_aa_false_positive_rate_near_5pct():
    rng = np.random.default_rng(12)
    n = 20_000
    s = rng.integers(0, 5, n)
    p = 0.02 + 0.02 * s
    rejected = 0
    for _ in range(N_REPS):
        t = (rng.random(n) < 0.85).astype(np.int8)
        y = (rng.random(n) < p).astype(np.int8)
        rejected += stratified_diff(y, t, s)["p_value"] < 0.05
    assert 0.04 < rejected / N_REPS < 0.06


def test_stratification_removes_bias_from_unequal_assignment():
    # Доля treatment и базовая доля исхода растут вместе, как в склеенном наборе.
    # Простая разница ловит ложный эффект, стратифицированная нет.
    rng = np.random.default_rng(13)
    n = 2_000_000
    s = rng.integers(0, 4, n)
    t = (rng.random(n) < np.array([0.80, 0.84, 0.88, 0.92])[s]).astype(np.int8)
    y = (rng.random(n) < np.array([0.02, 0.04, 0.06, 0.08])[s]).astype(np.int8)
    naive, strat = naive_diff(y, t), stratified_diff(y, t, s)
    assert naive["p_value"] < 1e-6
    assert strat["ci95"][0] < 0 < strat["ci95"][1]


def test_poisson_weights_one_equal_plain_sums():
    rng = np.random.default_rng(14)
    y = rng.integers(0, 2, 1000)
    t = rng.integers(0, 2, 1000)
    s = rng.integers(0, 3, 1000)
    assert stratified_diff(y, t, s, np.ones(1000)) == stratified_diff(y, t, s)


def test_ipw_recovers_planted_effect_under_confounding():
    rng = np.random.default_rng(15)
    n = 2_000_000
    s = rng.integers(0, 4, n)
    e = np.array([0.80, 0.84, 0.88, 0.92])[s]
    t = (rng.random(n) < e).astype(np.int8)
    y = (rng.random(n) < np.array([0.02, 0.04, 0.06, 0.08])[s] + 0.003 * t).astype(np.int8)
    ipw, strat = ipw_diff(y, t, e), stratified_diff(y, t, s)
    assert ipw["ci95"][0] < 0.003 < ipw["ci95"][1]
    assert abs(ipw["diff"] - strat["diff"]) < 0.3 * strat["se"]
    assert abs(ipw["se"] / strat["se"] - 1) < 0.05


def test_empty_strata_are_skipped():
    with_gap = from_sums([100, 0, 300], [10, 0, 20], [50, 0, 60], [4, 0, 5])
    without = from_sums([100, 300], [10, 20], [50, 60], [4, 5])
    assert with_gap["diff"] == without["diff"] and with_gap["se"] == without["se"]
    assert with_gap["n_strata_nonempty"] == 2


def test_iv_recovers_effect_on_compliers():
    # Показ получают 30% treatment, эффект только у показанных, 0,02
    rng = np.random.default_rng(16)
    reps, covered = 300, 0
    for _ in range(reps):
        n = 60_000
        s = rng.integers(0, 2, n)
        t = (rng.random(n) < 0.85).astype(np.int8)
        d = (t == 1) & (rng.random(n) < 0.2 + 0.2 * s)
        y = (rng.random(n) < 0.05 + 0.03 * s + 0.02 * d).astype(np.int8)
        lo, hi = iv_wald(y, d, t, s)["ci95"]
        covered += lo < 0.02 < hi
    assert 0.92 < covered / reps < 0.98
