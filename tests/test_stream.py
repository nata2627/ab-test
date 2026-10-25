import numpy as np

from src.effect import stratified_diff
from src.stream import cumulative_sums, look_index, look_z


def test_looks_are_equal_shares_and_reproducible():
    n = 100_003
    look = look_index(n, 10, seed=5)
    counts = np.bincount(look, minlength=10)
    assert counts.sum() == n
    assert counts.max() - counts.min() <= 1
    assert np.array_equal(look, look_index(n, 10, seed=5))
    assert not np.array_equal(look, look_index(n, 10, seed=6))


def test_cumulative_sums_match_prefix_of_stream():
    rng = np.random.default_rng(41)
    n, k_strata, k_looks = 50_000, 4, 5
    s = rng.integers(0, k_strata, n)
    t = (rng.random(n) < 0.85).astype(np.int8)
    y = (rng.random(n) < 0.03 + 0.01 * s).astype(np.int8)
    look = look_index(n, k_looks, seed=7)
    z = look_z(*cumulative_sums(look * k_strata + s, t, y, k_looks * k_strata, k_looks))
    for k in range(k_looks):
        seen = look <= k
        assert abs(z[k] - stratified_diff(y[seen], t[seen], s[seen])["z"]) < 1e-9
