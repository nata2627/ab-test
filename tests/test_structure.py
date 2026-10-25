import numpy as np

from src.balance import smd
from src.structure import modal_patterns, strata_of, stratified_smd


def test_stratification_removes_planted_confounding():
    rng = np.random.default_rng(5)
    n = 400_000
    x = rng.normal(size=(n, 2))
    share = np.where(x[:, 0] > 0, 0.9, 0.8)
    t = (rng.random(n) < share).astype(np.int8)
    assert abs(smd(x, t)[0]) > 0.1
    s = strata_of(share + rng.normal(0, 1e-6, n), 2)
    assert abs(stratified_smd(x, t, s)[0]) < 0.01


def test_modal_patterns_flags_mass_point():
    x = np.array([[1.0, 5.0], [1.0, 6.0], [2.0, 5.0], [1.0, 5.0]])
    assert modal_patterns(x).tolist() == [3, 1, 2, 3]


def test_stratified_smd_same_for_float32_and_float64():
    rng = np.random.default_rng(6)
    n = 3_000_000
    x = 4 + 0.05 * rng.normal(size=(n, 4))
    t = (rng.random(n) < 0.85).astype(np.int8)
    s = (rng.random(n) < 0.5).astype(np.int8)
    got = stratified_smd(x.astype(np.float32), t, s)
    assert np.allclose(got, stratified_smd(x, t, s), atol=1e-4)
