import numpy as np

from src.uplift import (
    ipw_weights,
    order,
    outcome_model,
    perfect_score,
    split,
    summarize,
    t_learner,
    transformed_outcome,
    uplift_curve,
)

FAST = dict(n_estimators=60, num_leaves=15, learning_rate=0.1, verbose=-1)


def synthetic(rng, n):
    """Эффект растет с x0, базовая доля визитов растет с x1, доля treatment зависит от x2.

    Модель исхода ранжирует по x1 и эффекта не видит.
    """
    x = rng.normal(size=(n, 4)).astype(np.float32)
    e = np.where(x[:, 2] > 0, 0.9, 0.8)
    t = (rng.random(n) < e).astype(np.int8)
    base = 1 / (1 + np.exp(-(-3 + 1.5 * x[:, 1] + 0.5 * x[:, 2])))
    tau = 0.10 / (1 + np.exp(-2 * x[:, 0]))
    y = (rng.random(n) < base + tau * t).astype(np.int8)
    return x, y, t, e, tau


def qini(score, y, t, w):
    perfect = uplift_curve(order(perfect_score(y, t)), y, t, w)
    return summarize(uplift_curve(order(score), y, t, w), perfect)["qini"]


def test_curve_endpoint_is_ipw_effect():
    x, y, t, e, tau = synthetic(np.random.default_rng(41), 400_000)
    curve = uplift_curve(order(tau), y, t, ipw_weights(t, e))
    assert abs(curve[-1] - tau.mean()) < 0.003


def test_split_keeps_shares():
    rng = np.random.default_rng(42)
    t = (rng.random(100_000) < 0.85).astype(np.int8)
    y = (rng.random(100_000) < 0.05).astype(np.int8)
    tr, te = split(t, y)
    assert len(te) == 30_000 and len(np.intersect1d(tr, te)) == 0
    assert abs(t[te].mean() - t.mean()) < 1e-4 and abs(y[te].mean() - y.mean()) < 1e-4


def test_models_rank_planted_effect():
    x, y, t, e, tau = synthetic(np.random.default_rng(43), 300_000)
    tr, te = split(t, y)
    w = ipw_weights(t[te], e[te])
    args = y[te], t[te], w

    oracle = qini(tau[te], *args)
    random = qini(np.random.default_rng(44).random(len(te)), *args)
    outcome = qini(outcome_model(x[tr], y[tr], FAST)(x[te]), *args)
    tl = qini(t_learner(x[tr], y[tr], t[tr], FAST)(x[te]), *args)
    to = qini(transformed_outcome(x[tr], y[tr], t[tr], e[tr], FAST)(x[te]), *args)

    # sd случайного порядка на этой выборке около 0,0075
    assert oracle > 0.05 and abs(random) < 0.025
    for got in (tl, to):
        assert got > 0.5 * oracle
        assert got > outcome + 0.02
