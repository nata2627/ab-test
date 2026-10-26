"""Производные числа README: отношения и разности готовых результатов.

По данным ничего не пересчитывает, только читает results/*.json, чтобы у каждого
числа в тексте был файл-источник.
"""

import json
from pathlib import Path

from src.data import ROOT


def load(name: str) -> dict:
    return json.loads((ROOT / f"results/{name}.json").read_text())


def width(ci: list[float]) -> float:
    return ci[1] - ci[0]


def summarize() -> dict:
    srm, power, main, cuped = load("srm"), load("power"), load("main"), load("cuped")
    peek, seq, struct = load("peeking"), load("sequential"), load("randomization_structure")
    models = load("uplift")["models"]

    visit = main["metrics"]["visit"]
    strat, naive = visit["stratified"], visit["naive"]
    cuped_widths = {
        m: width(cuped["metrics"][m]["cuped"]["ci95"]) / width(cuped["metrics"][m]["plain"]["ci95"])
        for m in ("visit", "conversion")
    }
    return {
        "srm_excess_over_expected": srm["n_treatment"] - srm["expected_share"] * srm["n"],
        "n_strata_empty": struct["propensity"]["n_strata"] - strat["n_strata_nonempty"],
        "effect_to_mde_visit": strat["diff"] / power["metrics"]["visit"]["mde_abs"],
        "naive_minus_stratified_visit": naive["diff"] - strat["diff"],
        "naive_minus_stratified_visit_in_se": (naive["diff"] - strat["diff"]) / strat["se"],
        "naive_overstatement_visit": naive["diff"] / strat["diff"] - 1,
        "cuped_ci_narrowing": {m: 1 - r for m, r in cuped_widths.items()},
        "peeking_inflation": peek["peek_every_look"]["count"] / peek["fixed_one_look"]["count"],
        "sequential_power_loss": seq["power_fixed"]["share"] - seq["power"]["share"],
        "sequential_sample_saving": 1 - seq["mean_sample_share_under_effect"],
        # Точечная оценка к интервалам gain_over_outcome_ci95 из uplift.json.
        "uplift_minus_outcome": {
            m: {
                c: v["gain_per_user"] - models["outcome"]["coverage"][c]["gain_per_user"]
                for c, v in models[m]["coverage"].items()
            }
            for m in ("t_learner", "transformed_outcome")
        },
    }


def main(out: Path = ROOT / "results/summary.json") -> dict:
    res = summarize()
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
