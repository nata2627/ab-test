import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats

from src.data import ROOT

# Базовые доли из карточки набора (исходы данных до плана анализа не смотрим).
BASE_RATES = {"visit": 0.046992, "conversion": 0.00292}
ALPHA = 0.05
POWER = 0.80


def power_two_props(p0: float, delta: float, n_t: int, n_c: int, alpha: float = ALPHA) -> float:
    """Мощность двустороннего z-теста разницы долей, дисперсия при альтернативе."""
    p1 = p0 + delta
    se = np.sqrt(p1 * (1 - p1) / n_t + p0 * (1 - p0) / n_c)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(z - delta / se) + stats.norm.cdf(-z - delta / se))


def mde_two_props(p0: float, n_t: int, n_c: int, alpha: float = ALPHA, power: float = POWER):
    return float(
        optimize.brentq(lambda d: power_two_props(p0, d, n_t, n_c, alpha) - power, 1e-9, 1 - p0)
    )


def n_total_for(p0: float, delta: float, share_t: float, alpha=ALPHA, power=POWER) -> int:
    """Сколько всего пользователей нужно при доле treatment share_t."""
    def gap(n):
        return power_two_props(p0, delta, share_t * n, (1 - share_t) * n, alpha) - power

    return int(np.ceil(optimize.brentq(gap, 10, 1e12)))


def main(out: Path = ROOT / "results/power.json") -> dict:
    srm = json.loads((ROOT / "results/srm.json").read_text())
    n_t, n_c = srm["n_treatment"], srm["n_control"]
    res = {"alpha": ALPHA, "power": POWER, "n_treatment": n_t, "n_control": n_c, "metrics": {}}
    for name, p0 in BASE_RATES.items():
        mde = mde_two_props(p0, n_t, n_c)
        res["metrics"][name] = {
            "base_rate": p0,
            "mde_abs": mde,
            "mde_rel": mde / p0,
            "n_for_mde_at_50_50": n_total_for(p0, mde, 0.5),
            "mde_abs_at_10pct_sample": mde_two_props(p0, n_t // 10, n_c // 10),
        }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
