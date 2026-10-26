"""Пуассоновский бутстреп стратифицированной оценки (план анализа, раздел 5)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.balance import SEED
from src.data import ROOT, load
from src.effect import from_sums
from src.structure import PROPENSITY

N_REPS = 2000
METRICS = ("visit", "conversion")
CHECKPOINT = ROOT / "logs/bootstrap_reps.npy"


def cell_sums(g: np.ndarray, ys: list[np.ndarray], w: np.ndarray, k: int) -> list[np.ndarray]:
    """Суммы по ячейкам g = 2*страта + группа: число пользователей, затем исходы."""
    return [np.bincount(g, weights=w, minlength=k)] + [
        np.bincount(g, weights=w * y, minlength=k) for y in ys
    ]


def estimates(sums: list[np.ndarray]) -> np.ndarray:
    """Разница и относительный эффект по каждой метрике из сумм по ячейкам."""
    n = sums[0]
    out = []
    for x in sums[1:]:
        r = from_sums(n[1::2], x[1::2], n[0::2], x[0::2])
        out += [r["diff"], r["rel"]]
    return np.array(out)


def run(g, ys, n_reps, checkpoint: Path | None, log_every: int = 50) -> np.ndarray:
    k = int(g.max()) + 1
    reps = np.full((n_reps, 2 * len(ys)), np.nan)
    if checkpoint is not None and checkpoint.exists():
        done = np.load(checkpoint)
        reps[: len(done)] = done[:n_reps]
    seeds = np.random.SeedSequence(SEED).spawn(n_reps)
    start = time.time()
    todo = np.flatnonzero(np.isnan(reps[:, 0]))
    for i, b in enumerate(todo, 1):
        w = np.random.default_rng(seeds[b]).poisson(1.0, len(g)).astype(np.float64)
        reps[b] = estimates(cell_sums(g, ys, w, k))
        if i % log_every == 0 or i == len(todo):
            rate = (time.time() - start) / i
            left = rate * (len(todo) - i)
            print(f"{b + 1}/{n_reps}, {rate:.3f} с на повтор, осталось {left:.0f} с", flush=True)
            if checkpoint is not None:
                checkpoint.parent.mkdir(exist_ok=True)
                np.save(checkpoint, reps[: b + 1])
    return reps


def main(fraction: float, n_reps: int, out: Path | None) -> dict:
    df = load(["treatment", *METRICS])
    s = pq.read_table(PROPENSITY, columns=["stratum"]).to_pandas()["stratum"].to_numpy()
    if fraction < 1:
        keep = np.random.default_rng(SEED).random(len(df)) < fraction
        df, s = df[keep], s[keep]
    g = 2 * s.astype(np.int64) + df["treatment"].to_numpy()
    ys = [df[m].to_numpy().astype(np.float64) for m in METRICS]

    start = time.time()
    reps = run(g, ys, n_reps, CHECKPOINT if out is not None else None)
    elapsed = time.time() - start

    k = int(g.max()) + 1
    plain = cell_sums(g, ys, np.ones(len(g)), k)
    point = estimates(plain)
    res = {"seed": SEED, "n_reps": n_reps, "fraction": fraction, "n_rows": int(len(g)),
           "seconds": round(elapsed, 1), "metrics": {}}
    for j, m in enumerate(METRICS):
        n, x = plain[0], plain[j + 1]
        analytic = from_sums(n[1::2], x[1::2], n[0::2], x[0::2])
        diff_ci = np.quantile(reps[:, 2 * j], [0.025, 0.975]).tolist()
        rel_ci = np.quantile(reps[:, 2 * j + 1], [0.025, 0.975]).tolist()
        width = diff_ci[1] - diff_ci[0]
        res["metrics"][m] = {
            "diff": float(point[2 * j]),
            "diff_ci95": diff_ci,
            "rel": float(point[2 * j + 1]),
            "rel_ci95": rel_ci,
            "boot_se": float(reps[:, 2 * j].std(ddof=1)),
            "analytic_se": analytic["se"],
            "analytic_ci95": analytic["ci95"],
            "width_ratio_boot_to_analytic": width / (analytic["ci95"][1] - analytic["ci95"][0]),
        }
    if out is not None:
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--reps", type=int, default=N_REPS)
    a = p.parse_args()
    full = a.fraction == 1 and a.reps == N_REPS
    print(json.dumps(main(a.fraction, a.reps, ROOT / "results/bootstrap.json" if full else None),
                     indent=2, ensure_ascii=False))
