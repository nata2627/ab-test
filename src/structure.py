"""Структура склейки тестов и стратификация по склонности, без исходов."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from src.balance import SEED, smd
from src.data import FEATURES, ROOT, load

N_FOLDS = 2
# 50 выбрано по балансу признаков до исходов: 10, 20, 50, 100 страт дают max |SMD|
# 0,016, 0,007, 0,0026, 0,0027
N_STRATA = 50
PROPENSITY = ROOT / "data/propensity.parquet"
GBM_PARAMS = dict(n_estimators=200, num_leaves=15, learning_rate=0.1, verbose=-1)


def modal_patterns(x: np.ndarray) -> np.ndarray:
    """Номер комбинации признаков, стоящих в своем модальном значении (точке сгущения)."""
    bits = np.zeros(len(x), dtype=np.int32)
    for i in range(x.shape[1]):
        vals, counts = np.unique(x[:, i], return_counts=True)
        bits |= (x[:, i] == vals[counts.argmax()]).astype(np.int32) << i
    return bits


def out_of_fold_propensity(x: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    fold = rng.integers(0, N_FOLDS, len(t))
    e = np.empty(len(t), dtype=np.float32)
    for k in range(N_FOLDS):
        fit = fold != k
        model = lgb.LGBMClassifier(**GBM_PARAMS, random_state=SEED + k).fit(x[fit], t[fit])
        e[~fit] = model.predict_proba(x[~fit])[:, 1]
    return e


def strata_of(e: np.ndarray, n_strata: int = N_STRATA) -> np.ndarray:
    edges = np.quantile(e, np.linspace(0, 1, n_strata + 1)[1:-1])
    return np.searchsorted(edges, e, side="right").astype(np.int8)


def stratified_smd(x: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Разность средних внутри страт, взвешенная долей страты, в единицах общего sd."""
    # float32 по оси 0 numpy суммирует последовательно, на миллионах строк это дает
    # ошибку округления больше самого перекоса
    x = x.astype(np.float64)
    pooled = np.sqrt((x[t == 1].var(axis=0) + x[t == 0].var(axis=0)) / 2)
    diff = np.zeros(x.shape[1])
    for k in np.unique(s):
        m = s == k
        tk = t[m]
        diff += m.mean() * (x[m][tk == 1].mean(axis=0) - x[m][tk == 0].mean(axis=0))
    return diff / pooled


def ipw_smd(x: np.ndarray, t: np.ndarray, e: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    pooled = np.sqrt((x[t == 1].var(axis=0) + x[t == 0].var(axis=0)) / 2)
    m1 = np.average(x[t == 1], weights=w[t == 1], axis=0)
    m0 = np.average(x[t == 0], weights=w[t == 0], axis=0)
    return (m1 - m0) / pooled


def main(out: Path = ROOT / "results/randomization_structure.json") -> dict:
    df = load(FEATURES + ["treatment"])
    x = df[FEATURES].to_numpy()
    t = df["treatment"].to_numpy()
    del df
    rng = np.random.default_rng(SEED)

    pat = modal_patterns(x)
    tab = pd.crosstab(pat, t)
    big = tab[tab.sum(axis=1) >= 1000]
    chi2, p, dof, _ = stats.chi2_contingency(big.to_numpy())
    share = big[1] / big.sum(axis=1)

    e = out_of_fold_propensity(x, t, rng)
    s = strata_of(e)
    pq.write_table(pa.table({"propensity": e, "stratum": s}), PROPENSITY)
    by_stratum = pd.DataFrame({"s": s, "t": t}).groupby("s")["t"].agg(["size", "mean"])

    raw = smd(x, t)
    adj = stratified_smd(x, t, s)
    ipw = ipw_smd(x, t, e.astype(np.float64))
    res = {
        "seed": SEED,
        "patterns": {
            "n_patterns_ge_1000": int(len(big)),
            "rows_covered": int(big.to_numpy().sum()),
            "n": int(len(t)),
            "share_treatment_min": float(share.min()),
            "share_treatment_max": float(share.max()),
            "chi2": float(chi2),
            "dof": int(dof),
            "p_value": float(p),
        },
        "propensity": {
            "n_folds": N_FOLDS,
            "n_strata": N_STRATA,
            "min": float(e.min()),
            "max": float(e.max()),
            "stratum_share_treatment": by_stratum["mean"].round(5).tolist(),
            "stratum_size": by_stratum["size"].tolist(),
        },
        "max_abs_smd_raw": float(np.abs(raw).max()),
        "max_abs_smd_stratified": float(np.abs(adj).max()),
        "max_abs_smd_ipw": float(np.abs(ipw).max()),
        "smd_stratified": dict(zip(FEATURES, adj.round(6).tolist(), strict=True)),
        "smd_ipw": dict(zip(FEATURES, ipw.round(6).tolist(), strict=True)),
        "top_stratum_propensity_range": [float(e[s == s.max()].min()), float(e.max())],
    }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
