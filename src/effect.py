"""Разница долей между группами, простая и стратифицированная (план анализа, раздел 4)."""
import numpy as np
from scipy import stats

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


def from_sums(n_t, x_t, n_c, x_c) -> dict:
    """Стратифицированная разница долей с весами, равными доле страты в выборке.

    Дисперсия это сумма квадратов весов на дисперсии Вальда внутри страт. Относительный
    эффект делится на стратифицированную долю контроля, интервал дельта-методом."""
    n_t, x_t, n_c, x_c = (np.asarray(a, dtype=np.float64) for a in (n_t, x_t, n_c, x_c))
    w = (n_t + n_c) / (n_t + n_c).sum()
    p_t, p_c = x_t / n_t, x_c / n_c
    v_t, v_c = p_t * (1 - p_t) / n_t, p_c * (1 - p_c) / n_c

    diff = float(w @ (p_t - p_c))
    var = float(w**2 @ (v_t + v_c))
    se = np.sqrt(var)
    z = diff / se

    base = float(w @ p_c)
    var_base = float(w**2 @ v_c)
    # Cov(разница, доля контроля) = -Var(доля контроля), группы независимы
    var_rel = var / base**2 + diff**2 * var_base / base**4 + 2 * diff * var_base / base**3
    rel, se_rel = diff / base, np.sqrt(var_rel)
    return {
        "n_treatment": float(n_t.sum()),
        "n_control": float(n_c.sum()),
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
