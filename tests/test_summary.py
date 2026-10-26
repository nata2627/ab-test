from src.summary import summarize


def test_summary_consistent_with_results():
    s = summarize()
    assert s["n_strata_empty"] == 3
    assert abs(s["peeking_inflation"] - 423 / 111) < 1e-12
    assert 0 < s["cuped_ci_narrowing"]["visit"] < 1
