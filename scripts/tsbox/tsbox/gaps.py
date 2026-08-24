"""Detección de datos que faltan. Dos cosas distintas que la gente confunde:

1. NaN/null dentro de la columna  -> la muestra existe pero el valor no.
2. Salto en el eje X              -> la muestra no existe siquiera.

La 2 solo es detectable si el muestreo es aproximadamente regular.
"""
from __future__ import annotations

import numpy as np

Interval = tuple[float, float]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Índices [ini, fin] (inclusive) de cada racha de True."""
    if mask.size == 0 or not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1))
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size - 1)
    return list(zip(starts, ends))


def nan_intervals(x: np.ndarray, y: np.ndarray,
                  min_len: int = 1, step: float | None = None) -> list[Interval]:
    """Tramos de valores ausentes, expandidos a las muestras válidas vecinas.

    Vectorizado. La versión anterior llamaba a median_step() DENTRO del bucle:
    con 28.667 tramos eso son 28.667 medianas sobre el array completo, o sea
    O(n·m). Medido sobre un fichero real: 72 segundos, el 99% del tiempo de
    abrir un panel. Ahora el paso se calcula una vez.
    """
    mask = ~np.isfinite(y)
    if not mask.any():
        return []
    runs = _runs_vec(mask)
    if runs.size == 0:
        return []
    if min_len > 1:
        runs = runs[(runs[:, 1] - runs[:, 0] + 1) >= min_len]
        if runs.size == 0:
            return []

    n = len(x)
    i0, i1 = runs[:, 0], runs[:, 1]
    a = x[np.maximum(i0 - 1, 0)]
    b = x[np.minimum(i1 + 1, n - 1)]

    same = a == b            # NaN pegado a un extremo: dale ancho visible
    if same.any():
        if step is None:
            step = median_step(x)
        a = np.where(same, a - step / 2, a)
        b = np.where(same, b + step / 2, b)
    return list(zip(a.astype(float).tolist(), b.astype(float).tolist()))


def _runs_vec(mask: np.ndarray) -> np.ndarray:
    """Igual que _runs pero devuelve un array (m, 2), sin bucle de Python."""
    if mask.size == 0 or not mask.any():
        return np.empty((0, 2), dtype=np.int64)
    m = mask.astype(np.int8)
    d = np.diff(m)
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1)
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [mask.size - 1]])
    return np.column_stack([starts, ends])


def median_step(x: np.ndarray) -> float:
    if len(x) < 2:
        return 1.0
    d = np.diff(x)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else 1.0


def time_gaps(x: np.ndarray, factor: float = 1.8,
              step: float | None = None) -> list[Interval]:
    """Saltos donde dt > factor * dt_mediano. factor bajo = falsos positivos
    en señales con jitter; súbelo si tu adquisición no es determinista."""
    if len(x) < 3:
        return []
    if step is None:
        step = median_step(x)
    if step <= 0:
        return []
    d = np.diff(x)
    idx = np.flatnonzero(d > factor * step)
    return [(float(x[i]), float(x[i + 1])) for i in idx]


def x_is_duplicated(x: np.ndarray) -> bool:
    """El eje X repite valores -> hay varias entidades apiladas. En ese caso
    dt_mediano no es el periodo de muestreo de nada y la anchura de los huecos
    que se marquen no significa lo que parece."""
    if x.size < 3:
        return False
    return bool((np.diff(x) == 0).mean() > 0.1)


def missing_report(x: np.ndarray, y: np.ndarray, factor: float = 1.8) -> dict:
    step = median_step(x)                 # una sola vez, no una por tramo
    dup = x_is_duplicated(x)
    gaps = [] if dup else time_gaps(x, factor, step)
    nans = nan_intervals(x, y, step=step)
    lost = sum((b - a) / step - 1 for a, b in gaps)
    return {
        "n_nan": int((~np.isfinite(y)).sum()),
        "nan_intervals": nans,
        "time_gaps": gaps,
        "x_duplicado": dup,
        "muestras_perdidas_estimadas": int(round(max(0.0, lost))),
        "dt_mediano": step,
    }
