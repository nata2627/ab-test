"""Графики README. Строятся из results/*.json и формул мощности, данные не нужны."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data import ROOT  # noqa: E402
from src.power import BASE_RATES, mde_two_props  # noqa: E402

OUT = ROOT / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK = "#8a8985", "#52514e"


def load(name: str) -> dict:
    return json.loads((ROOT / f"results/{name}.json").read_text())


def new_axes(*args, **kw):
    fig, axes = plt.subplots(*args, dpi=150, **kw)
    for ax in np.atleast_1d(axes):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK, labelsize=8)
    return fig, axes


def save(fig, name: str) -> Path:
    fig.tight_layout()
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    return path


def fmt_int(n: float) -> str:
    return f"{int(n):,}".replace(",", " ")


def effect() -> Path:
    main, cuped, boot = load("main"), load("cuped"), load("bootstrap")
    fig, axes = new_axes(1, 2, figsize=(7.2, 3.0))
    for ax, metric, title in zip(axes, ("visit", "conversion"), ("визиты", "конверсии"),
                                 strict=True):
        m = main["metrics"][metric]
        rows = [
            ("простая разница", m["naive"]["diff"], m["naive"]["ci95"], GRAY),
            ("стратифицированная", m["stratified"]["diff"], m["stratified"]["ci95"], BLUE),
            ("IPW", m["ipw"]["diff"], m["ipw"]["ci95"], BLUE),
            ("бутстреп", boot["metrics"][metric]["diff"], boot["metrics"][metric]["diff_ci95"],
             BLUE),
            ("CUPED", cuped["metrics"][metric]["cuped"]["diff"],
             cuped["metrics"][metric]["cuped"]["ci95"], AQUA),
        ]
        for i, (_, d, ci, color) in enumerate(rows):
            ax.plot(100 * np.array(ci), [i, i], color=color, lw=2, solid_capstyle="round")
            ax.plot(100 * d, i, "o", color=color, ms=5)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel("разница treatment и контроля, п. п.", fontsize=8, color=INK)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", color="#e6e5e1", lw=0.6)
        ax.set_axisbelow(True)
    n = m["stratified"]
    fig.suptitle(f"Эффект назначения, 95% интервалы, treatment {fmt_int(n['n_treatment'])}, "
                 f"контроль {fmt_int(n['n_control'])}", fontsize=9, color=INK)
    return save(fig, "effect.png")


def balance() -> Path:
    raw, strat = load("randomization"), load("randomization_structure")
    feats = list(raw["smd"])
    y = np.arange(len(feats))
    fig, ax = new_axes(figsize=(5.6, 3.6))
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.plot([raw["smd"][f] for f in feats], y, "o", color=GRAY, ms=5, label="до стратификации")
    ax.plot([strat["smd_stratified"][f] for f in feats], y, "o", color=BLUE, ms=5,
            label=f"внутри {strat['propensity']['n_strata']} страт склонности")
    ax.set_yticks(y, feats)
    ax.invert_yaxis()
    ax.set_xlabel("стандартизованная разность средних, treatment минус контроль", fontsize=8,
                  color=INK)
    ax.set_title(f"Баланс признаков, {fmt_int(raw['n'])} пользователей", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return save(fig, "balance.png")


def peeking() -> Path:
    pk, seq = load("peeking"), load("sequential")
    by_look = pk["false_positive_by_look"]
    looks = np.arange(1, pk["n_looks"] + 1)
    fp = np.array([b["share"] for b in by_look])
    ci = np.array([b["wilson_ci95"] for b in by_look]).T
    fig, ax = new_axes(figsize=(6.4, 3.6))
    ax.errorbar(looks, 100 * fp, yerr=100 * np.abs(ci - fp), color=BLUE, marker="o", ms=4,
                lw=2, capsize=2, label="z-тест, остановка на первом p < 0,05")
    s = seq["false_positive"]
    ax.errorbar([looks[-1] + 0.35], [100 * s["share"]],
                yerr=100 * np.abs(np.array(s["wilson_ci95"])[:, None] - s["share"]),
                color=ORANGE, marker="s", ms=5, lw=2, capsize=2,
                label="последовательный тест, все просмотры")
    ax.axhline(100 * seq["alpha"], color=GRAY, ls="--", lw=1)
    ax.text(1, 100 * seq["alpha"] - 0.4, "заявленные 5%", va="top", color=INK, fontsize=8)
    ax.set_xticks(looks)
    ax.set_xlabel("число просмотров (каждый через 10% выборки)", fontsize=8, color=INK)
    ax.set_ylabel("ложные срабатывания, % повторов", fontsize=8, color=INK)
    ax.set_title(f"A/A, {by_look[0]['n']} перестановок, 95% интервалы Уилсона", fontsize=10)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return save(fig, "peeking.png")


def uplift() -> Path:
    res = load("uplift")
    labels = {"t_learner": ("T-learner", BLUE),
              "transformed_outcome": ("трансформация исхода", ORANGE),
              "outcome": ("модель исхода без группы", AQUA)}
    grid = 100 * np.r_[0, res["grid"]]
    fig, ax = new_axes(figsize=(6.4, 3.6))
    for k, (label, color) in labels.items():
        ax.plot(grid, 100 * np.r_[0, res["curves"][k]], color=color, lw=2, label=label)
    total = 100 * res["total_effect_per_user"]
    ax.plot([0, 100], [0, total], color=GRAY, ls="--", lw=1, label="случайный охват")
    for share in res["models"]["t_learner"]["coverage"]:
        ax.axvline(float(share.rstrip("%")), color="#e6e5e1", lw=0.8, zorder=0)
    ax.set_xlabel("охват, % пользователей test по убыванию оценки модели", fontsize=8, color=INK)
    ax.set_ylabel("дополнительные визиты, п. п. от всех", fontsize=8, color=INK)
    ax.set_title(f"Кривая прироста на test, {fmt_int(res['n_test'])} пользователей",
                 fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return save(fig, "uplift.png")


def power() -> Path:
    pw = load("power")
    share = pw["n_treatment"] / (pw["n_treatment"] + pw["n_control"])
    n_all = pw["n_treatment"] + pw["n_control"]
    ns = np.geomspace(1e5, 3e7, 60)
    fig, ax = new_axes(figsize=(6.4, 3.4))
    for (name, p0), color, label in zip(BASE_RATES.items(), (BLUE, ORANGE),
                                        ("визит", "конверсия"), strict=True):
        mde = [mde_two_props(p0, int(share * n), int((1 - share) * n)) / p0 for n in ns]
        ax.plot(ns / 1e6, 100 * np.array(mde), color=color, lw=2, label=label)
        actual = 100 * pw["metrics"][name]["mde_rel"]
        ax.plot(n_all / 1e6, actual, "o", color=color, ms=6)
        ax.annotate(f"{actual:.2f}%".replace(".", ","), (n_all / 1e6, actual),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
    ax.axvline(n_all / 1e6, color=GRAY, ls="--", lw=1)
    ax.text(n_all / 1e6, ax.get_ylim()[1] * 0.95, " размер набора", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.set_xlabel("пользователей всего, млн (treatment 85%)", fontsize=8, color=INK)
    ax.set_ylabel("MDE относительный, %", fontsize=8, color=INK)
    ax.set_title("Минимальный обнаружимый эффект до анализа, альфа 0,05, мощность 0,8",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return save(fig, "power.png")


def strata() -> Path:
    st = load("randomization_structure")["propensity"]
    share = 100 * np.array(st["stratum_share_treatment"])
    size = np.array(st["stratum_size"])
    x = np.arange(1, len(share) + 1)
    fig, ax = new_axes(figsize=(6.4, 3.4))
    ax.axhline(85, color=GRAY, ls="--", lw=1)
    ax.text(1, 85.2, "0,85 из карточки набора", fontsize=8, color=INK, va="bottom")
    ax.scatter(x, share, s=12 + 60 * size / size.max(), color=BLUE, zorder=3)
    ax.set_xlabel(f"страта склонности, по возрастанию ({len(share)} непустых)", fontsize=8,
                  color=INK)
    ax.set_ylabel("доля treatment в страте, %", fontsize=8, color=INK)
    ax.set_title("Доля treatment зависит от признаков, размер точки это размер страты",
                 fontsize=10)
    return save(fig, "strata.png")


def sequential() -> Path:
    seq = load("sequential")
    looks = np.arange(1, len(seq["bounds_z"]) + 1)
    fig, (a1, a2) = new_axes(1, 2, figsize=(7.6, 3.2))
    a1.plot(10 * looks, seq["real_stream"]["z_by_look"], color=BLUE, marker="o", ms=4, lw=2,
            label="z на реальном потоке")
    a1.plot(10 * looks, seq["bounds_z"], color=ORANGE, marker="s", ms=4, lw=2,
            label="граница O'Brien-Fleming")
    a1.axhline(1.96, color=GRAY, ls="--", lw=1, label="1,96, фиксированный тест")
    a1.set_xlabel("просмотрено выборки, %", fontsize=8, color=INK)
    a1.set_ylabel("z визитов", fontsize=8, color=INK)
    a1.set_title("Реальный поток", fontsize=10)
    a1.legend(frameon=False, fontsize=7, loc="center right")
    counts = np.array(seq["stop_look_counts_under_effect"])
    n = seq["power"]["n"]
    a2.bar(10 * looks, 100 * counts[1:] / n, width=7, color=BLUE)
    a2.set_xlabel("остановка на просмотре, % выборки", fontsize=8, color=INK)
    a2.set_ylabel(f"% из {n} повторов", fontsize=8, color=INK)
    a2.set_title("Остановка при эффекте в MDE", fontsize=10)
    a2.set_ylim(0, 100 * counts[1:].max() / n * 1.25)
    a2.text(0.02, 0.97, f"не остановился до конца {counts[0]} из {n}", transform=a2.transAxes,
            fontsize=8, color=INK, va="top")
    a2.set_xticks(10 * looks)
    return save(fig, "sequential.png")


def coverage() -> Path:
    models = load("uplift")["models"]
    labels = {"t_learner": ("T-learner", BLUE),
              "transformed_outcome": ("трансформация исхода", ORANGE),
              "outcome": ("модель исхода без группы", AQUA)}
    shares = list(models["t_learner"]["coverage"])
    x = np.arange(len(shares))
    fig, ax = new_axes(figsize=(6.4, 3.4))
    for i, (k, (label, color)) in enumerate(labels.items()):
        cov = models[k]["coverage"]
        g = np.array([100 * cov[c]["gain_over_random"] for c in shares])
        ci = np.array([[100 * v for v in cov[c]["gain_over_random_ci95"]] for c in shares]).T
        ax.bar(x + (i - 1) * 0.26, g, width=0.24, color=color, label=label,
               yerr=np.abs(ci - g), capsize=2, error_kw={"lw": 1, "ecolor": INK})
    ax.set_xticks(x, [f"охват {c}" for c in shares])
    ax.set_ylabel("визитов сверх случайного охвата, п. п.", fontsize=8, color=INK)
    ax.set_title("Таргетинг лучших по модели против случайных, 95% интервалы", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return save(fig, "coverage.png")


def main() -> list[Path]:
    OUT.mkdir(exist_ok=True)
    return [power(), strata(), balance(), effect(), peeking(), sequential(), uplift(), coverage()]


if __name__ == "__main__":
    for p in main():
        print(p.relative_to(ROOT))
