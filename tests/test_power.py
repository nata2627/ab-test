import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from src.power import mde_two_props, n_total_for, power_two_props


def test_power_close_to_statsmodels():
    # statsmodels через арксинус Коэна, расхождение с z-тестом на долях в тысячных
    p0, d, n_t, n_c = 0.05, 0.002, 400_000, 100_000
    es = proportion_effectsize(p0 + d, p0)
    sm = NormalIndPower().power(es, nobs1=n_t, ratio=n_c / n_t, alpha=0.05)
    assert abs(power_two_props(p0, d, n_t, n_c) - sm) < 0.01


def test_mde_inverts_power():
    d = mde_two_props(0.047, 1_000_000, 200_000)
    assert abs(power_two_props(0.047, d, 1_000_000, 200_000) - 0.8) < 1e-6


def test_balanced_split_needs_fewer_users():
    assert n_total_for(0.05, 0.002, 0.5) < n_total_for(0.05, 0.002, 0.85)


def test_power_by_simulation():
    rng = np.random.default_rng(7)
    p0, n_t, n_c = 0.05, 20_000, 20_000
    d = mde_two_props(p0, n_t, n_c)
    x_t = rng.binomial(n_t, p0 + d, 4000) / n_t
    x_c = rng.binomial(n_c, p0, 4000) / n_c
    se = np.sqrt(x_t * (1 - x_t) / n_t + x_c * (1 - x_c) / n_c)
    rejected = np.abs((x_t - x_c) / se) > 1.959964
    assert abs(rejected.mean() - 0.8) < 0.025
