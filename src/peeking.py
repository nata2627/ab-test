"""A/A-симуляция подглядывания и последовательный тест (план анализа, раздел 8).

Метки группы переставляются внутри страт с сохранением числа treatment в каждой страте,
исход остается реальным. В том же повторе на той же перестановке вносится эффект размером
MDE визитов: у treatment нули исхода превращаются в единицы с вероятностью mde / (1 - p),
так средний прирост доли равен mde. По каждому повтору хранится z на всех просмотрах."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from src.balance import SEED
from src.data import ROOT, load
from src.sequential import gsd_bounds
from src.stream import N_LOOKS, cumulative_sums, look_index, look_z
from src.structure import PROPENSITY

N_REPS = 2000
ALPHA = 0.05
CHECKPOINT = ROOT / "logs/peeking_z.npy"
# Повторы берут seed из своей ветки, порядок потока из SEED (src.stream)
REP_SEED = SEED + 1


class Stream:
    """Данные, отсортированные по страте: перестановка внутри страты это перемешивание отрезка."""

    def __init__(self, y, t, s, look, n_looks=N_LOOKS):
        order = np.argsort(s, kind="stable")
        s = s[order]
        self.y, self.t = y[order].astype(np.float64), t[order]
        self.k = int(s.max()) + 1
        self.n_looks = n_looks
        self.cells = n_looks * self.k
        self.cell = look[order].astype(np.int64) * self.k + s
        edges = np.flatnonzero(np.diff(s)) + 1
        self.segments = list(zip(np.r_[0, edges], np.r_[edges, len(s)], strict=True))
        self.base_rate = float(self.y.mean())

    def z(self, t, y=None) -> np.ndarray:
        y = self.y if y is None else y
        return look_z(*cumulative_sums(self.cell, t, y, self.cells, self.n_looks))

    def permuted(self, rng) -> np.ndarray:
        t = self.t.copy()
        for a, b in self.segments:
            rng.shuffle(t[a:b])
        return t

    def planted(self, t, effect, rng) -> np.ndarray:
        flip = (rng.random(len(t)) < effect / (1 - self.base_rate)) & (t == 1) & (self.y == 0)
        return np.where(flip, 1.0, self.y)


def simulate(stream: Stream, effect: float, n_reps: int, checkpoint: Path | None,
             log_every: int = 25) -> np.ndarray:
    """z формы (повтор, [без эффекта, с эффектом], просмотр)."""
    z = np.full((n_reps, 2, stream.n_looks), np.nan)
    if checkpoint is not None and checkpoint.exists():
        done = np.load(checkpoint)
        z[: len(done)] = done[:n_reps]
    seeds = np.random.SeedSequence(REP_SEED).spawn(n_reps)
    todo = np.flatnonzero(np.isnan(z[:, 0, 0]))
    start = time.time()
    for i, r in enumerate(todo, 1):
        rng = np.random.default_rng(seeds[r])
        t = stream.permuted(rng)
        z[r, 0] = stream.z(t)
        z[r, 1] = stream.z(t, stream.planted(t, effect, rng))
        if i % log_every == 0 or i == len(todo):
            rate = (time.time() - start) / i
            left = rate * (len(todo) - i)
            print(f"{r + 1}/{n_reps}, {rate:.2f} с на повтор, осталось {left:.0f} с", flush=True)
            if checkpoint is not None:
                np.save(checkpoint, z[: r + 1])
    return z


def share(hits: np.ndarray) -> dict:
    k, n = int(hits.sum()), len(hits)
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return {"count": k, "n": n, "share": k / n, "wilson_ci95": [float(lo), float(hi)]}


def first_crossing(z: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Номер первого просмотра (с 1), где |z| за границей, 0 если не пересек."""
    crossed = np.abs(z) >= bounds
    return np.where(crossed.any(axis=-1), crossed.argmax(axis=-1) + 1, 0)


def summarize(z: np.ndarray, bounds: np.ndarray, effect: float) -> tuple[dict, dict]:
    fixed = np.full(z.shape[-1], np.inf)
    fixed[-1] = stats.norm.ppf(1 - ALPHA / 2)
    naive = np.full(z.shape[-1], stats.norm.ppf(1 - ALPHA / 2))
    null, eff = z[:, 0], z[:, 1]
    peeking = {
        "n_looks": z.shape[-1],
        "z_threshold": naive[0],
        "fixed_one_look": share(first_crossing(null, fixed) > 0),
        "peek_every_look": share(first_crossing(null, naive) > 0),
        "false_positive_by_look": [share(first_crossing(null[:, : k + 1], naive[: k + 1]) > 0)
                                   for k in range(z.shape[-1])],
    }
    stop = first_crossing(eff, bounds)
    sequential = {
        "spending": "Lan-DeMets, O'Brien-Fleming, двусторонняя",
        "alpha": ALPHA,
        "bounds_z": bounds.tolist(),
        "effect_abs": effect,
        "false_positive": share(first_crossing(null, bounds) > 0),
        "power": share(stop > 0),
        "power_fixed": share(first_crossing(eff, fixed) > 0),
        "power_peek_every_look": share(first_crossing(eff, naive) > 0),
        # Доля выборки, на которой тест остановился; без остановки это вся выборка
        "mean_sample_share_under_effect": float(np.where(stop > 0, stop, z.shape[-1]).mean()
                                                / z.shape[-1]),
        "stop_look_counts_under_effect": np.bincount(stop, minlength=z.shape[-1] + 1).tolist(),
    }
    return peeking, sequential


def real_stream(stream: Stream, bounds: np.ndarray) -> dict:
    z = stream.z(stream.t)
    seq = int(first_crossing(z, bounds))
    peek = int(first_crossing(z, np.full(len(z), stats.norm.ppf(1 - ALPHA / 2))))
    return {
        "z_by_look": z.tolist(),
        "sequential_stop_look": seq,
        "sequential_stop_sample_share": seq / len(z),
        "sequential_decision": ("не остановился, эффект не обнаружен" if not seq else
                                "рост визитов" if z[seq - 1] > 0 else "падение визитов"),
        "naive_first_significant_look": peek,
    }


def main(fraction: float, n_reps: int, full: bool) -> dict:
    df = load(["treatment", "visit"])
    s = pq.read_table(PROPENSITY, columns=["stratum"]).to_pandas()["stratum"].to_numpy()
    look = look_index(len(df))
    y, t = df["visit"].to_numpy(), df["treatment"].to_numpy()
    if fraction < 1:
        keep = np.random.default_rng(SEED).random(len(df)) < fraction
        y, t, s, look = y[keep], t[keep], s[keep], look[keep]
    stream = Stream(y, t, s.astype(np.int64), look)
    effect = json.loads((ROOT / "results/power.json").read_text())["metrics"]["visit"]["mde_abs"]
    bounds = gsd_bounds(np.arange(1, N_LOOKS + 1) / N_LOOKS, ALPHA)

    start = time.time()
    z = simulate(stream, effect, n_reps, CHECKPOINT if full else None)
    meta = {"seed_stream": SEED, "seed_reps": REP_SEED, "n_reps": n_reps, "fraction": fraction,
            "n_rows": len(y), "seconds": round(time.time() - start, 1)}
    peeking, sequential = summarize(z, bounds, effect)
    peeking, sequential = {**meta, **peeking}, {**meta, **sequential}
    if full:
        sequential["real_stream"] = real_stream(stream, bounds)
        for name, res in (("peeking", peeking), ("sequential", sequential)):
            (ROOT / f"results/{name}.json").write_text(
                json.dumps(res, indent=2, ensure_ascii=False))
    return {"peeking": peeking, "sequential": sequential}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--reps", type=int, default=N_REPS)
    a = p.parse_args()
    full = a.fraction == 1 and a.reps == N_REPS
    print(json.dumps(main(a.fraction, a.reps, full), indent=2, ensure_ascii=False))
