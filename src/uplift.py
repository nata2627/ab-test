"""Uplift-модели и метрики ранжирования (план анализа, раздел 10)."""
import argparse
import json
import resource
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from src.balance import SEED
from src.data import FEATURES, ROOT, load
from src.structure import PROPENSITY

TEST_SHARE = 0.3
GBM_PARAMS = dict(n_estimators=300, num_leaves=31, learning_rate=0.05, verbose=-1)
GRID = np.linspace(0.01, 1.0, 100)
COVERAGE = (0.1, 0.3, 0.5)
N_BOOT = 500
SCORERS = ("random", "outcome", "t_learner", "transformed_outcome")


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


def peak_gb() -> float:
    # на macOS ru_maxrss в байтах
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def fit_and_score(fraction: float, cache: Path | None) -> dict:
    """Обучение на train и оценки на test; оценки кешируются, чтобы обрыв не терял модели."""
    if cache is not None and cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    df = load(FEATURES + ["treatment", "visit"])
    e = pq.read_table(PROPENSITY, columns=["propensity"]).to_pandas()["propensity"].to_numpy()
    if fraction < 1:
        keep = np.random.default_rng(SEED).random(len(df)) < fraction
        df, e = df[keep], e[keep]
    x = df[FEATURES].to_numpy()
    y, t = df["visit"].to_numpy(), df["treatment"].to_numpy()
    del df
    tr, te = split(t, y)
    xtr, xte = x[tr], x[te]
    del x
    out = {"y": y[te], "t": t[te], "e": e[te], "n_train": np.array(len(tr))}
    out["random"] = np.random.default_rng(SEED).random(len(te))
    fits = {
        "outcome": lambda: outcome_model(xtr, y[tr]),
        "t_learner": lambda: t_learner(xtr, y[tr], t[tr]),
        "transformed_outcome": lambda: transformed_outcome(xtr, y[tr], t[tr], e[tr]),
    }
    for name, fit in fits.items():
        start = time.time()
        out[name] = fit()(xte)
        print(f"{name}: {time.time() - start:.0f} с, пик памяти {peak_gb():.2f} ГБ", flush=True)
    if cache is not None:
        np.savez(cache, **out)
    return out


def evaluate(sc: dict, n_boot: int) -> dict:
    y, t = sc["y"], sc["t"]
    w = ipw_weights(t, sc["e"].astype(np.float64))
    orders = {name: order(sc[name]) for name in ("perfect", *SCORERS) if name != "perfect"}
    orders["perfect"] = order(perfect_score(y, t))
    at = np.searchsorted(GRID, COVERAGE)

    def stats(wb):
        curves = {k: uplift_curve(idx, y, t, wb) for k, idx in orders.items()}
        row = {}
        for k in SCORERS:
            c = curves[k]
            row[k] = [*summarize(c, curves["perfect"]).values(), *(c[at] - GRID[at] * c[-1])]
        return row, curves

    point, curves = stats(w)
    seeds = np.random.SeedSequence(SEED).spawn(n_boot)
    reps = {k: np.empty((n_boot, 2 + len(COVERAGE))) for k in SCORERS}
    start = time.time()
    for b in range(n_boot):
        wb = w * np.random.default_rng(seeds[b]).poisson(1.0, len(y))
        row, _ = stats(wb)
        for k in SCORERS:
            reps[k][b] = row[k]
        if (b + 1) % 50 == 0:
            print(f"бутстреп {b + 1}/{n_boot}, {(time.time() - start) / (b + 1):.2f} с на повтор",
                  flush=True)

    def ci(a):
        return np.quantile(a, [0.025, 0.975]).tolist()

    res = {"models": {}, "curves": {k: curves[k].tolist() for k in SCORERS}}
    total = curves["random"][-1]
    for k in SCORERS:
        r, p = reps[k], point[k]
        res["models"][k] = {
            "auuc": p[0], "auuc_ci95": ci(r[:, 0]),
            "qini": p[1], "qini_ci95": ci(r[:, 1]),
            "qini_minus_outcome_ci95": ci(r[:, 1] - reps["outcome"][:, 1]),
            "coverage": {
                f"{phi:.0%}": {
                    "gain_per_user": float(curves[k][i]),
                    "share_of_total_effect": float(curves[k][i] / total),
                    "gain_over_random": p[2 + j], "gain_over_random_ci95": ci(r[:, 2 + j]),
                    # вне плана анализа: парное сравнение с моделью исхода на той же доле
                    "gain_over_outcome_ci95": ci(r[:, 2 + j] - reps["outcome"][:, 2 + j]),
                }
                for j, (phi, i) in enumerate(zip(COVERAGE, at, strict=True))
            },
        }
    for k in ("t_learner", "transformed_outcome"):
        m = res["models"][k]
        m["accepted"] = bool(m["qini_ci95"][0] > 0 and m["qini_ci95"][0] > point["outcome"][1])
    res["total_effect_per_user"] = float(total)
    res["bootstrap_seconds"] = round(time.time() - start, 1)
    return res


def main(fraction: float, n_boot: int, out: Path | None) -> dict:
    start = time.time()
    full = out is not None
    sc = fit_and_score(fraction, ROOT / "logs/uplift_scores.npz" if full else None)
    res = {"seed": SEED, "fraction": fraction, "n_train": int(sc["n_train"]),
           "n_test": int(len(sc["y"])), "test_share": TEST_SHARE, "gbm_params": GBM_PARAMS,
           "n_boot": n_boot, "grid": GRID.tolist(), **evaluate(sc, n_boot)}
    res["seconds"] = round(time.time() - start, 1)
    res["peak_memory_gb"] = round(peak_gb(), 2)
    if full:
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--boot", type=int, default=N_BOOT)
    a = p.parse_args()
    full = a.fraction == 1 and a.boot == N_BOOT
    res = main(a.fraction, a.boot, ROOT / "results/uplift.json" if full else None)
    res.pop("curves"), res.pop("grid")
    print(json.dumps(res, indent=2, ensure_ascii=False))
