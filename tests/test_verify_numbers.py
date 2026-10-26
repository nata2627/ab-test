import json

from scripts.verify_numbers import tokenize, verify


def values(text):
    return [(t.value, t.decimals, t.scale) for t in tokenize(text)]


def test_russian_format():
    assert values("n 11 882 655, доля 0,8500") == [(11882655, 0, 1), (0.85, 4, 1)]


def test_units_set_scale():
    assert values("21,2% и 0,74 п. п., 14 млн") == [(21.2, 1, 100), (0.74, 2, 100), (14, 0, 1e-6)]


def test_range_hyphen_is_not_minus():
    assert values("(-0,037-0,062)") == [(-0.037, 3, 1), (0.062, 3, 1)]


def test_names_code_and_allowed_skipped():
    assert values("признак f3, `seed=7`, [x](a/1.md), Criteo v2.1, CC BY-NC-SA 4.0") == []


def check(tmp_path, readme_text, results):
    (tmp_path / "results").mkdir()
    (tmp_path / "results/r.json").write_text(json.dumps(results))
    (tmp_path / "README.md").write_text(readme_text)
    return verify(tmp_path / "README.md", tmp_path / "results")


def test_rounded_number_matches(tmp_path):
    results = {"s": 0.2115, "c": 423, "n": 2000}
    matched, missing = check(tmp_path, "доля 21,2%, 423 из 2000", results)
    assert not missing and [src for _, src in matched] == ["r.json.s", "r.json.c", "r.json.n"]


def test_range_inherits_unit(tmp_path):
    results = {"lo": 0.00711, "hi": 0.00769}
    _, missing = check(tmp_path, "интервал 0,711-0,769 п. п.", results)
    assert not missing


def test_unsourced_number_fails(tmp_path):
    _, missing = check(tmp_path, "эффект 0,75 п. п.", {"d": 0.00739})
    assert [t.text for t in missing] == ["0,75 п. п."]
