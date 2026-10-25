import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data import FEATURES, ROOT, load

SEED = 20261025
N_TRAIN = 2_000_000
N_TEST = 2_000_000
N_BOOT = 200


def smd(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    x1, x0 = x[t == 1].astype(np.float64), x[t == 0].astype(np.float64)
    pooled = np.sqrt((x1.var(axis=0) + x0.var(axis=0)) / 2)
    return (x1.mean(axis=0) - x0.mean(axis=0)) / pooled


def auc_with_ci(y: np.ndarray, score: np.ndarray, rng: np.random.Generator) -> dict:
    auc = roc_auc_score(y, score)
    n = len(y)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boot[b] = roc_auc_score(y[idx], score[idx])
    return {"auc": float(auc), "ci95": np.quantile(boot, [0.025, 0.975]).tolist()}


def classifiers_auc(x: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> dict:
    idx = rng.permutation(len(t))[: N_TRAIN + N_TEST]
    tr, te = idx[:N_TRAIN], idx[N_TRAIN:]

    scaler = StandardScaler().fit(x[tr])
    logit = LogisticRegression(max_iter=200).fit(scaler.transform(x[tr]), t[tr])
    p_logit = logit.predict_proba(scaler.transform(x[te]))[:, 1]

    gbm = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=15, learning_rate=0.1, random_state=SEED, verbose=-1
    ).fit(x[tr], t[tr])
    p_gbm = gbm.predict_proba(x[te])[:, 1]

    return {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "logit": auc_with_ci(t[te], p_logit, rng),
        "gbm": auc_with_ci(t[te], p_gbm, rng),
    }


def main(out: Path = ROOT / "results/randomization.json") -> dict:
    start = time.time()
    df = load(FEATURES + ["treatment"])
    x = df[FEATURES].to_numpy()
    t = df["treatment"].to_numpy()
    del df
    rng = np.random.default_rng(SEED)
    res = {
        "seed": SEED,
        "n": int(len(t)),
        "smd": dict(zip(FEATURES, smd(x, t).round(6).tolist(), strict=True)),
        "max_abs_smd": float(np.abs(smd(x, t)).max()),
        **classifiers_auc(x, t, rng),
        "n_boot_auc": N_BOOT,
        "seconds": round(time.time() - start, 1),
    }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
