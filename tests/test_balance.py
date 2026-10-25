import numpy as np

from src import balance


def _data(shift: float, n: int = 200_000):
    rng = np.random.default_rng(3)
    t = (rng.random(n) < 0.85).astype(np.int8)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    x[:, 0] += shift * t
    return x, t


def _auc(x, t):
    balance.N_TRAIN, balance.N_TEST, balance.N_BOOT = 100_000, 100_000, 20
    return balance.classifiers_auc(x, t, np.random.default_rng(0))


def test_balanced_auc_near_half():
    res = _auc(*_data(0.0))
    assert abs(res["logit"]["auc"] - 0.5) < 0.02
    assert res["logit"]["ci95"][0] < 0.5 + 0.01


def test_group_dependent_covariate_detected():
    x, t = _data(0.3)
    assert abs(balance.smd(x, t)[0] - 0.3) < 0.03
    res = _auc(x, t)
    assert res["logit"]["auc"] > 0.55
    assert res["gbm"]["ci95"][0] > 0.5
