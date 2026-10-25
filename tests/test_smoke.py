import numpy as np


def test_numpy_seed_reproducible():
    a = np.random.default_rng(0).random(3)
    b = np.random.default_rng(0).random(3)
    assert np.array_equal(a, b)
