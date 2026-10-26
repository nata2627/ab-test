"""Сверка чисел README с results/*.json.

Каждое число в README должно совпасть, с точностью до напечатанного округления,
хотя бы с одним числом из results/. Число без источника роняет скрипт.
Масштаб берется из единицы после числа: «%» и «п. п.» умножают значение из
results на 100, «тыс.» и «млн» делят. Код в обратных кавычках, адреса ссылок и
строки из ALLOWED не проверяются.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Числа, которые не являются результатом расчета: версия набора, лицензия, год статьи.
ALLOWED = ["v2.1", "CC BY-NC-SA 4.0", "AdKDD 2018", "Python 3.11"]

SPACE = "[   ]"
NUMBER = re.compile(
    rf"(?P<sign>[-−+]?)(?P<int>\d{{1,3}}(?:{SPACE}\d{{3}})+|\d+)(?:,(?P<frac>\d+))?"
    rf"(?P<unit>{SPACE}?%|{SPACE}п\.{SPACE}?п\.|{SPACE}тыс\.?|{SPACE}млн)?"
)
SCALE = {"%": 100.0, "п.п.": 100.0, "тыс": 1e-3, "тыс.": 1e-3, "млн": 1e-6}


@dataclass
class Token:
    text: str
    value: float
    decimals: int
    scale: float
    line: int


def collect_values(obj) -> list[tuple[str, float]]:
    out = []

    def walk(o, path):
        if isinstance(o, bool):
            return
        if isinstance(o, int | float):
            out.append((path, float(o)))
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    walk(obj, "")
    return out


def load_results(results_dir: Path) -> list[tuple[str, float]]:
    values = []
    for f in sorted(results_dir.glob("*.json")):
        values += [(f.name + p, v) for p, v in collect_values(json.loads(f.read_text()))]
    return values


def strip_unchecked(text: str) -> str:
    text = re.sub(r"```.*?```", lambda m: "\n" * m.group().count("\n"), text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"<!--.*?-->", lambda m: "\n" * m.group().count("\n"), text, flags=re.S)
    for s in ALLOWED:
        text = text.replace(s, " ")
    return text


def tokenize(text: str) -> list[Token]:
    tokens = []
    for lineno, line in enumerate(strip_unchecked(text).splitlines(), 1):
        prev = None
        for m in NUMBER.finditer(line):
            start = m.start()
            sign = m["sign"]
            # Дефис сразу после числа это диапазон «0,057-0,117», а не минус.
            is_range = sign == "-" and start > 0 and line[start - 1].isdigit()
            if is_range:
                sign = ""
            # Буква или точка вплотную слева значит имя вроде f3 или хвост «2.1».
            elif start > 0 and (line[start - 1].isalnum() or line[start - 1] in "._"):
                prev = None
                continue
            digits = re.sub(SPACE, "", m["int"])
            frac = m["frac"] or ""
            value = float(f"{digits}.{frac}" if frac else digits)
            if sign in ("-", "\u2212"):
                value = -value
            unit = re.sub(SPACE, "", m["unit"] or "")
            text = m.group().strip()
            tok = Token(
                text[1:] if is_range else text, value, len(frac), SCALE.get(unit, 1.0), lineno
            )
            # В «0,711-0,769 п. п.» и «от -0,048 до -0,001 п. п.» единица стоит один раз
            # и относится к обеим границам.
            joined = prev is not None and (is_range or line[prev[2] : start].strip() == "до")
            if joined and prev[1] == "" and unit:
                prev[0].scale = tok.scale
            tokens.append(tok)
            prev = (tok, unit, m.end())
    return tokens


def find_source(tok: Token, values: list[tuple[str, float]]) -> str | None:
    half = 0.5 * 10.0 ** (-tok.decimals) + 1e-12
    for path, v in values:
        if abs(v * tok.scale - tok.value) <= half:
            return path
    return None


def verify(readme: Path, results_dir: Path) -> tuple[list[tuple[Token, str]], list[Token]]:
    values = load_results(results_dir)
    matched, missing = [], []
    for tok in tokenize(readme.read_text()):
        src = find_source(tok, values)
        if src is None:
            missing.append(tok)
        else:
            matched.append((tok, src))
    return matched, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", type=Path, default=ROOT / "README.md")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--explain", action="store_true", help="напечатать источник каждого числа")
    args = ap.parse_args()

    matched, missing = verify(args.readme, args.results)
    if args.explain:
        for tok, src in matched:
            print(f"{tok.line}: {tok.text} <- {src}")
    for tok in missing:
        print(f"verify_numbers: строка {tok.line}, «{tok.text}» нет в results/", file=sys.stderr)
    if missing:
        return 1
    print(f"verify_numbers: {len(matched)} чисел, у всех есть источник в results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
