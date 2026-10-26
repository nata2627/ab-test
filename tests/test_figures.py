from src import figures


def test_figures_build_from_results(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "OUT", tmp_path)
    paths = figures.main()
    assert [p.name for p in paths] == [
        "power.png", "strata.png", "balance.png", "effect.png",
        "peeking.png", "sequential.png", "uplift.png", "coverage.png",
    ]
    assert all(p.stat().st_size > 10_000 for p in paths)
