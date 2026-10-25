import numpy as np
from scipy import stats

GRID = 4001


def obf_spending(t: np.ndarray, alpha: float) -> np.ndarray:
    """Функция расходования Lan-DeMets типа O'Brien-Fleming, двусторонняя."""
    z = stats.norm.ppf(1 - alpha / 4)
    return 2 * (2 - 2 * stats.norm.cdf(z / np.sqrt(t)))


def gsd_bounds(info: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Двусторонние границы по z для просмотров с долями информации info.

    Численная рекурсия Армитиджа: плотность z-статистики, не остановившейся
    к k-му просмотру, переносится на сетке, граница подбирается так, чтобы
    вероятность первого пересечения равнялась приросту расходуемой альфы.
    """
    info = np.asarray(info, dtype=float)
    spent = np.diff(np.concatenate([[0.0], obf_spending(info, alpha)]))
    x = np.linspace(-12, 12, GRID)
    dx = x[1] - x[0]
    bounds = np.empty(len(info))
    density = None
    for k, t in enumerate(info):
        if k == 0:
            dens_k = stats.norm.pdf(x)
        else:
            dt = t - info[k - 1]
            # z_k*sqrt(t_k) = z_{k-1}*sqrt(t_{k-1}) + приращение с дисперсией dt
            mu = x[:, None] * np.sqrt(info[k - 1] / t)
            kern = stats.norm.pdf(x[None, :], loc=mu, scale=np.sqrt(dt / t))
            dens_k = (density[:, None] * kern).sum(axis=0) * dx

        cdf = np.concatenate([[0.0], np.cumsum((dens_k[1:] + dens_k[:-1]) / 2) * dx])

        def cross(b, cdf=cdf):
            return np.interp(-b, x, cdf) + cdf[-1] - np.interp(b, x, cdf)

        lo, hi = 0.0, 12.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if cross(mid) > spent[k] else (lo, mid)
        bounds[k] = hi
        density = np.where(np.abs(x) < hi, dens_k, 0.0)
    return bounds
