import numpy as np

from src.sequential import gsd_bounds


def test_obf_lan_demets_k5_matches_published():
    # Jennison, Turnbull (2000), Lan-DeMets OBF, K=5, alpha=0.05 двусторонняя
    expected = [4.877, 3.357, 2.680, 2.290, 2.031]
    got = gsd_bounds(np.linspace(0.2, 1, 5))
    assert np.allclose(got, expected, atol=0.01)


def test_single_look_is_fixed_test():
    assert abs(gsd_bounds(np.array([1.0]))[0] - 1.95996) < 1e-3
