"""Uplift-модели и метрики ранжирования (план анализа, раздел 10)."""
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from src.balance import SEED

TEST_SHARE = 0.3
GBM_PARAMS = dict(n_estimators=300, num_leaves=31, learning_rate=0.05, verbose=-1)
GRID = np.linspace(0.01, 1.0, 100)


def split(t: np.ndarray, y: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(t))
    return train_test_split(idx, test_size=TEST_SHARE, stratify=2 * t + y, random_state=seed)


def ipw_weights(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.where(t == 1, 1 / e, 1 / (1 - e))


def order(score: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Индексы по убыванию оценки; совпадающие оценки идут в случайном порядке."""
    perm = np.random.default_rng(seed).permutation(len(score))
    return perm[np.argsort(-score[perm], kind="stable")]


def uplift_curve(idx: np.ndarray, y: np.ndarray, t: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Прирост исхода на одного пользователя выборки при охвате доли GRID лучших.

    Веса нормируются по всей выборке, а не по охваченным: в верхушке идеального порядка контроля
    нет вовсе, и доля по охваченным там не определена. В конце кривой получается IPW-оценка
    среднего эффекта.
    """
    y, t, w = y[idx], t[idx], w[idx]
    cuts = np.ceil(GRID * len(idx)).astype(np.int64) - 1
    wt, wc = w * t, w * (1 - t)
    gain = np.cumsum(wt * y) / wt.sum() - np.cumsum(wc * y) / wc.sum()
    return gain[cuts]


def summarize(curve: np.ndarray, perfect: np.ndarray) -> dict:
    """AUUC это площадь под кривой, Qini это площадь между кривой и случайным охватом,
    нормированный Qini делит ее на ту же площадь для идеального порядка."""
    random_area = curve[-1] / 2
    auuc = np.trapezoid(np.r_[0, curve], np.r_[0, GRID])
    best = np.trapezoid(np.r_[0, perfect], np.r_[0, GRID]) - perfect[-1] / 2
    return {"auuc": float(auuc), "qini": float((auuc - random_area) / best)}


def perfect_score(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Идеальный порядок: откликнувшиеся из treatment первыми, из контроля последними."""
    return y * (2 * t.astype(np.int8) - 1)


def t_learner(x, y, t, params=GBM_PARAMS):
    models = [
        lgb.LGBMClassifier(**params, random_state=SEED + g).fit(x[t == g], y[t == g])
        for g in (0, 1)
    ]
    return lambda z: models[1].predict_proba(z)[:, 1] - models[0].predict_proba(z)[:, 1]


def transformed_outcome(x, y, t, e, params=GBM_PARAMS):
    """Регрессия на z = y (t / e - (1 - t) / (1 - e)), у которой E[z | x] равно эффекту."""
    z = y * (t / e - (1 - t) / (1 - e))
    model = lgb.LGBMRegressor(**params, random_state=SEED).fit(x, z)
    return model.predict


def outcome_model(x, y, params=GBM_PARAMS):
    model = lgb.LGBMClassifier(**params, random_state=SEED).fit(x, y)
    return lambda z: model.predict_proba(z)[:, 1]
