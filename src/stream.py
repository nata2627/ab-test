"""Поток пользователей для подглядывания (план анализа, раздел 8).

Времени в данных нет, поэтому порядок прихода задается перестановкой с фиксированным seed.
Просмотр k видит первых ceil(k*n/K) пользователей потока."""
import numpy as np

from src.balance import SEED
from src.effect import from_sums

N_LOOKS = 10


def look_index(n: int, n_looks: int = N_LOOKS, seed: int = SEED) -> np.ndarray:
    """Номер первого просмотра, на котором пользователь уже в выборке (0..n_looks-1)."""
    position = np.empty(n, dtype=np.int64)
    position[np.random.default_rng(seed).permutation(n)] = np.arange(n)
    return (position * n_looks // n).astype(np.int8)


def cumulative_sums(cell: np.ndarray, t: np.ndarray, y: np.ndarray, n_cells: int, n_looks: int):
    """Накопленные к каждому просмотру суммы по стратам, массивы формы (n_looks, страты).

    cell = look * страты + страта. Итог совпадает с stratum_sums на первых пользователях потока."""
    y = y.astype(np.float64)
    t = t.astype(np.float64)
    n_all = np.bincount(cell, minlength=n_cells)
    x_all = np.bincount(cell, weights=y, minlength=n_cells)
    n_t = np.bincount(cell, weights=t, minlength=n_cells)
    x_t = np.bincount(cell, weights=t * y, minlength=n_cells)
    out = [a.reshape(n_looks, -1).cumsum(axis=0) for a in (n_t, x_t, n_all - n_t, x_all - x_t)]
    return tuple(out)


def look_z(n_t, x_t, n_c, x_c) -> np.ndarray:
    """z стратифицированной разницы на каждом просмотре."""
    return np.array([from_sums(*(a[k] for a in (n_t, x_t, n_c, x_c)))["z"]
                     for k in range(len(n_t))])
