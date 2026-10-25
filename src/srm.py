import json
from pathlib import Path

import numpy as np
from scipy import stats

from src.data import ROOT, load

# Доля treatment из карточки набора. Это округленная описательная цифра, а не
# записанная доля назначения в дизайне, поэтому тест против нее проверяет
# согласие с карточкой, а не исправность рандомизатора.
EXPECTED_SHARE = 0.85


def srm_test(n_treatment: int, n_control: int, expected_share: float) -> dict:
    n = n_treatment + n_control
    observed = np.array([n_treatment, n_control])
    expected = n * np.array([expected_share, 1 - expected_share])
    chi2, p = stats.chisquare(observed, expected)
    share = n_treatment / n
    se = np.sqrt(share * (1 - share) / n)
    return {
        "n": int(n),
        "n_treatment": int(n_treatment),
        "n_control": int(n_control),
        "share_treatment": share,
        "share_treatment_ci95": [share - 1.96 * se, share + 1.96 * se],
        "expected_share": expected_share,
        "chi2": float(chi2),
        "p_value": float(p),
        "p_too_close": p_too_close(n_treatment, n, expected_share),
    }


def p_too_close(n_treatment: int, n: int, share: float) -> float:
    """Вероятность, что при честном биномиальном назначении n_treatment окажется
    не дальше от n*share, чем наблюдено. Малое значение значит, что доля
    подогнана, например прореживанием, и обычный SRM-тест теряет смысл."""
    d = abs(n_treatment - n * share)
    lo, hi = np.ceil(n * share - d), np.floor(n * share + d)
    return float(stats.binom.cdf(hi, n, share) - stats.binom.cdf(lo - 1, n, share))


def main(out: Path = ROOT / "results/srm.json") -> dict:
    t = load(["treatment"])["treatment"].to_numpy()
    n_t = int(t.sum())
    res = srm_test(n_t, len(t) - n_t, EXPECTED_SHARE)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
