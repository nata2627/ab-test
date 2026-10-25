import numpy as np

from src.bootstrap import cell_sums, estimates, run
from src.effect import stratified_diff


def test_plain_weights_reproduce_stratified_estimate():
    rng = np.random.default_rng(21)
    n = 50_000
    s, t = rng.integers(0, 3, n), (rng.random(n) < 0.85).astype(np.int64)
    y = (rng.random(n) < 0.05 + 0.01 * t).astype(np.float64)
    got = estimates(cell_sums(2 * s + t, [y], np.ones(n), 6))
    ref = stratified_diff(y, t, s)
    assert np.allclose(got, [ref["diff"], ref["rel"]])


def test_bootstrap_se_matches_analytic():
    rng = np.random.default_rng(22)
    n = 100_000
    s, t = rng.integers(0, 3, n), (rng.random(n) < 0.85).astype(np.int64)
    y = (rng.random(n) < 0.05 + 0.01 * t).astype(np.float64)
    reps = run(2 * s + t, [y], 400, None, log_every=10**9)
    se = stratified_diff(y, t, s)["se"]
    # sd оценки sd по 400 повторам около 3,5%
    assert abs(reps[:, 0].std(ddof=1) / se - 1) < 0.12
