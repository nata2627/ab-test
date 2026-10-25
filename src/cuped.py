"""CUPED на признаках внутри страт склонности (план анализа, раздел 6)."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.data import FEATURES, ROOT, load
from src.effect import from_sums, stratified_diff
from src.structure import PROPENSITY

CHUNK = 1_000_000


def center_within_strata(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Признаки минус их среднее в страте по обеим группам, float32 ради памяти."""
    k = int(s.max()) + 1
    counts = np.maximum(np.bincount(s, minlength=k), 1)
    out = np.empty(x.shape, dtype=np.float32)
    for j in range(x.shape[1]):
        col = x[:, j].astype(np.float64)
        out[:, j] = col - (np.bincount(s, weights=col, minlength=k) / counts)[s]
    return out


def theta(xc: np.ndarray, y: np.ndarray) -> np.ndarray:
    """МНК исхода на отцентрированные признаки через нормальные уравнения по кускам."""
    xtx = np.zeros((xc.shape[1], xc.shape[1]))
    xty = np.zeros(xc.shape[1])
    for i in range(0, len(y), CHUNK):
        a = xc[i : i + CHUNK].astype(np.float64)
        xtx += a.T @ a
        xty += a.T @ y[i : i + CHUNK]
    return np.linalg.solve(xtx, xty)


def cuped_diff(y: np.ndarray, t: np.ndarray, s: np.ndarray, xc: np.ndarray) -> dict:
    th = theta(xc, y.astype(np.float64))
    adj = np.empty(len(y))
    for i in range(0, len(y), CHUNK):
        adj[i : i + CHUNK] = y[i : i + CHUNK] - xc[i : i + CHUNK].astype(np.float64) @ th
    k = int(s.max()) + 1
    t1 = t == 1

    def sums(m, v):
        return np.bincount(s[m], weights=v[m], minlength=k)

    ones = np.ones(len(y))
    res = from_sums(sums(t1, ones), sums(t1, adj), sums(~t1, ones), sums(~t1, adj),
                    sums(t1, adj**2), sums(~t1, adj**2))
    res["theta"] = th.tolist()
    return res


def compare(y, t, s, xc) -> dict:
    plain, cuped = stratified_diff(y, t, s), cuped_diff(y, t, s, xc)
    return {
        "plain": plain,
        "cuped": cuped,
        "variance_reduction": 1 - (cuped["se"] / plain["se"]) ** 2,
        "shift_in_se": (cuped["diff"] - plain["diff"]) / plain["se"],
        # Если обе оценки несмещены, Var(сдвига) = Var(без CUPED) - Var(с CUPED)
        "shift_in_own_sd": (cuped["diff"] - plain["diff"])
        / np.sqrt(plain["se"] ** 2 - cuped["se"] ** 2),
    }


def main(out: Path = ROOT / "results/cuped.json") -> dict:
    df = load(FEATURES + ["treatment", "visit", "conversion"])
    s = pq.read_table(PROPENSITY, columns=["stratum"]).to_pandas()["stratum"].to_numpy()
    xc = center_within_strata(df[FEATURES].to_numpy(), s)
    t = df["treatment"].to_numpy()
    res = {"features": FEATURES, "metrics": {}}
    for m in ("visit", "conversion"):
        res["metrics"][m] = compare(df[m].to_numpy(), t, s, xc)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
