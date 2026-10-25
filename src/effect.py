"""Разница долей между группами, простая и стратифицированная (план анализа, раздел 4)."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

from src.data import ROOT, load
from src.structure import N_STRATA, PROPENSITY

Z95 = stats.norm.ppf(0.975)


def stratum_sums(y: np.ndarray, t: np.ndarray, s: np.ndarray, w: np.ndarray | None = None):
    """Число пользователей и сумма исхода по группам в каждой страте.

    Веса w нужны пуассоновскому бутстрепу: каждая строка входит в суммы w раз."""
    k = int(s.max()) + 1
    w = np.ones(len(y)) if w is None else w
    t1 = t == 1
    n_t = np.bincount(s[t1], weights=w[t1], minlength=k)
    n_c = np.bincount(s[~t1], weights=w[~t1], minlength=k)
    x_t = np.bincount(s[t1], weights=(w * y)[t1], minlength=k)
    x_c = np.bincount(s[~t1], weights=(w * y)[~t1], minlength=k)
    return n_t, x_t, n_c, x_c


def from_sums(n_t, x_t, n_c, x_c, q_t=None, q_c=None) -> dict:
    """Стратифицированная разница средних с весами, равными доле страты в выборке.

    Дисперсия это сумма квадратов весов на дисперсии Вальда внутри страт. Для
    непрерывного исхода (CUPED) нужны суммы квадратов q, для бинарного q = x. Относительный
    эффект делится на стратифицированное среднее контроля, интервал дельта-методом."""
    q_t = x_t if q_t is None else q_t
    q_c = x_c if q_c is None else q_c
    arrays = [np.asarray(a, dtype=np.float64) for a in (n_t, x_t, n_c, x_c, q_t, q_c)]
    # Квантили склонности с совпадающими значениями дают пустые страты, у них вес ноль
    nonempty = arrays[0] + arrays[2] > 0
    n_t, x_t, n_c, x_c, q_t, q_c = (a[nonempty] for a in arrays)
    w = (n_t + n_c) / (n_t + n_c).sum()
    p_t, p_c = x_t / n_t, x_c / n_c
    v_t, v_c = (q_t / n_t - p_t**2) / n_t, (q_c / n_c - p_c**2) / n_c

    diff = float(w @ (p_t - p_c))
    var = float(w**2 @ (v_t + v_c))
    se = np.sqrt(var)
    z = diff / se

    base = float(w @ p_c)
    var_base = float(w**2 @ v_c)
    # Cov(разница, доля контроля) = -Var(доля контроля), группы независимы
    if base > 0:
        var_rel = var / base**2 + diff**2 * var_base / base**4 + 2 * diff * var_base / base**3
        rel, se_rel = diff / base, np.sqrt(var_rel)
    else:
        # показ в контроле, относительного эффекта нет
        rel = se_rel = float("nan")
    return {
        "n_treatment": float(n_t.sum()),
        "n_control": float(n_c.sum()),
        "n_strata_nonempty": int(nonempty.sum()),
        "rate_treatment": float(w @ p_t),
        "rate_control": base,
        "diff": diff,
        "se": float(se),
        "ci95": [diff - Z95 * se, diff + Z95 * se],
        "z": float(z),
        "p_value": float(2 * stats.norm.sf(abs(z))),
        "rel": rel,
        "se_rel": float(se_rel),
        "rel_ci95": [rel - Z95 * se_rel, rel + Z95 * se_rel],
    }


def stratified_diff(y, t, s, w=None) -> dict:
    return from_sums(*stratum_sums(y, t, s, w))


def naive_diff(y, t) -> dict:
    return stratified_diff(y, t, np.zeros(len(y), dtype=np.int8))


def iv_wald(y, d, t, s) -> dict:
    """Эффект показа d на тех, кто получил его из-за назначения t (CACE, план, раздел 7).

    Отношение стратифицированного ITT по исходу к ITT по показу, интервал дельта-методом
    с ковариацией двух ITT внутри страт."""
    k = int(s.max()) + 1
    t1 = t == 1
    y, d = y.astype(np.float64), d.astype(np.float64)

    def by_group(v):
        return np.bincount(s[t1], weights=v[t1], minlength=k), np.bincount(
            s[~t1], weights=v[~t1], minlength=k)

    (n_t, n_c), (y_t, y_c), (d_t, d_c), (yd_t, yd_c) = (by_group(v) for v in (
        np.ones(len(y)), y, d, y * d))
    nonempty = n_t + n_c > 0
    n_t, n_c, y_t, y_c, d_t, d_c, yd_t, yd_c = (a[nonempty] for a in (
        n_t, n_c, y_t, y_c, d_t, d_c, yd_t, yd_c))
    w = (n_t + n_c) / (n_t + n_c).sum()
    itt_y, itt_d = from_sums(n_t, y_t, n_c, y_c), from_sums(n_t, d_t, n_c, d_c)

    def cov(n, a, b, ab):
        return (ab / n - (a / n) * (b / n)) / n

    c = float(w**2 @ (cov(n_t, y_t, d_t, yd_t) + cov(n_c, y_c, d_c, yd_c)))
    ny, nd = itt_y["diff"], itt_d["diff"]
    var = (itt_y["se"] ** 2 / nd**2 + ny**2 * itt_d["se"] ** 2 / nd**4 - 2 * ny * c / nd**3)
    cace, se = ny / nd, float(np.sqrt(var))
    return {
        "cace": cace,
        "se": se,
        "ci95": [cace - Z95 * se, cace + Z95 * se],
        "itt_outcome": ny,
        "itt_exposure": nd,
        "exposure_rate_treatment": float(d_t.sum() / n_t.sum()),
        "exposure_rate_control": float(d_c.sum() / n_c.sum()),
        "n_exposed_control": int(d_c.sum()),
        "n_control": int(n_c.sum()),
    }


def ipw_diff(y, t, e) -> dict:
    """Проверка устойчивости: разница нормированных средних с весами 1/e и 1/(1-e).

    Ошибка линеаризацией с известной склонностью, ее оценку не учитывает."""
    y = y.astype(np.float64)
    out = {}
    for name, m, w in (("treatment", t == 1, 1 / e), ("control", t == 0, 1 / (1 - e))):
        wm, ym = w[m], y[m]
        mu = float(wm @ ym / wm.sum())
        out[name] = (mu, float(np.sqrt((wm**2) @ (ym - mu) ** 2) / wm.sum()))
    diff = out["treatment"][0] - out["control"][0]
    se = float(np.hypot(out["treatment"][1], out["control"][1]))
    return {"rate_treatment": out["treatment"][0], "rate_control": out["control"][0],
            "diff": diff, "se": se, "ci95": [diff - Z95 * se, diff + Z95 * se]}


def main(out: Path = ROOT / "results/main.json") -> dict:
    df = load(["treatment", "visit", "conversion", "exposure"])
    prop = pq.read_table(PROPENSITY).to_pandas()
    t, s = df["treatment"].to_numpy(), prop["stratum"].to_numpy()
    e = prop["propensity"].to_numpy().astype(np.float64)
    res = {"n_strata": N_STRATA, "metrics": {}}
    for metric in ("visit", "conversion"):
        y = df[metric].to_numpy()
        strat, ipw = stratified_diff(y, t, s), ipw_diff(y, t, e)
        res["metrics"][metric] = {
            "role": "primary" if metric == "visit" else "secondary",
            "stratified": strat,
            "naive": naive_diff(y, t),
            "ipw": ipw,
            "ipw_minus_stratified_in_se": (ipw["diff"] - strat["diff"]) / strat["se"],
            "n_events": int(y.sum()),
        }
    d = df["exposure"].to_numpy()
    res["exposure"] = {
        "note": "показ не рандомизирован, treatment как инструмент, оценка CACE",
        **{m: iv_wald(df[m].to_numpy(), d, t, s) for m in ("visit", "conversion")},
    }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
