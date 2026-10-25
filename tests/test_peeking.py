import numpy as np

from src.peeking import Stream, first_crossing, simulate, summarize
from src.sequential import gsd_bounds
from src.stream import look_index


def synthetic(n=30_000, seed=51):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 4, n)
    t = (rng.random(n) < np.array([0.8, 0.85, 0.85, 0.9])[s]).astype(np.int8)
    y = (rng.random(n) < 0.03 + 0.01 * s).astype(np.int8)
    return Stream(y, t, s, look_index(n, 10, seed=seed))


def test_permutation_keeps_treatment_count_in_each_stratum():
    st = synthetic()
    t = st.permuted(np.random.default_rng(1))
    for a, b in st.segments:
        assert t[a:b].sum() == st.t[a:b].sum()
    assert not np.array_equal(t, st.t)


def test_planted_effect_has_planned_size():
    st = synthetic(n=400_000)
    rng = np.random.default_rng(2)
    t = st.permuted(rng)
    y = st.planted(t, 0.01, rng)
    added = (y - st.y)[t == 1].mean()
    assert abs(added - 0.01) < 0.001
    assert np.all(y[t == 0] == st.y[t == 0])


def test_first_crossing():
    z = np.array([[0.5, 2.5, 3.0], [0.1, 0.2, 0.3], [4.0, 0.0, 0.0]])
    assert first_crossing(z, np.array([2.0, 2.0, 2.0])).tolist() == [2, 0, 1]


def test_peeking_inflates_and_sequential_holds_alpha():
    st = synthetic()
    bounds = gsd_bounds(np.arange(1, 11) / 10)
    z = simulate(st, 0.0, 400, checkpoint=None, log_every=10**6)
    peeking, sequential = summarize(z, bounds, 0.0)
    assert peeking["peek_every_look"]["share"] > 0.12
    assert peeking["fixed_one_look"]["share"] < 0.09
    assert sequential["false_positive"]["share"] < 0.09
