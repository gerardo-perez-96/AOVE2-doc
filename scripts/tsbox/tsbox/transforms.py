"""Recetas y estadísticas. Se recalculan siempre desde el origen."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .model import (KIND_DERIVATIVE, KIND_LAG, KIND_ROLLING_MEAN,
                    KIND_ROLLING_STD)


def fmt_x(v: float, is_dt: bool) -> str:
    """Formato común del eje X, en UTC siempre (ver docs/zona_horaria...).

    Vive aquí (sin Qt) para que cualquier módulo pueda usarlo sin arriesgar
    un import circular con mainwindow.py -- lo necesita también
    analysis_ui.py para el selector de secciones.
    """
    if not np.isfinite(v):
        return "—"
    if is_dt:
        try:
            return (dt.datetime.fromtimestamp(v, dt.timezone.utc)
                      .strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        except (OSError, ValueError, OverflowError):
            return f"{v:.6g}"
    return f"{v:.6g}"


def rolling_mean(y: np.ndarray, window: int, center: bool = True) -> np.ndarray:
    window = max(1, int(window))
    return (pd.Series(y).rolling(window, center=center, min_periods=1)
            .mean().to_numpy(dtype=np.float64))


def rolling_std(y: np.ndarray, window: int, center: bool = True) -> np.ndarray:
    window = max(2, int(window))
    return (pd.Series(y).rolling(window, center=center, min_periods=2)
            .std().to_numpy(dtype=np.float64))


def derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """d/dx respetando el espaciado real. Los NaN se propagan a los vecinos:
    es correcto, no un bug. La derivada de un hueco no existe."""
    if len(y) < 2:
        return np.full_like(y, np.nan, dtype=np.float64)
    xs = np.asarray(x, dtype=np.float64)
    if not np.all(np.diff(xs) > 0):  # np.gradient exige X estrictamente creciente
        xs = np.arange(len(y), dtype=np.float64)
    return np.gradient(np.asarray(y, dtype=np.float64), xs)


def lag_shift(y: np.ndarray, lag: int) -> np.ndarray:
    """Desplaza en muestras. lag>0 retrasa la señal (la mueve al futuro).
    Los extremos quedan NaN: no se inventa dato donde no lo hay."""
    y = np.asarray(y, dtype=np.float64)
    if lag == 0:
        return y.copy()
    out = np.full_like(y, np.nan)
    if lag > 0:
        out[lag:] = y[:-lag]
    else:
        out[:lag] = y[-lag:]
    return out


def apply_recipe(kind: str, params: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if kind == KIND_ROLLING_MEAN:
        return rolling_mean(y, params.get("window", 10), params.get("center", True))
    if kind == KIND_ROLLING_STD:
        return rolling_std(y, params.get("window", 10), params.get("center", True))
    if kind == KIND_DERIVATIVE:
        return derivative(x, y)
    if kind == KIND_LAG:
        return lag_shift(y, int(params.get("lag", 0)))
    raise ValueError(f"Receta desconocida: {kind}")


def window_stats(x: np.ndarray, y: np.ndarray,
                 x0: float | None = None, x1: float | None = None) -> dict:
    """Estadísticas de la ventana visible. Si no se pasa rango, de todo."""
    if x0 is not None and x1 is not None:
        i0, i1 = np.searchsorted(x, [x0, x1])
        i0 = max(0, int(i0) - 1)
        i1 = min(len(x), int(i1) + 1)
        y = y[i0:i1]
    if y.size == 0:
        return {k: float("nan") for k in
                ("mean", "std", "var", "min", "max", "ptp")} | {"n": 0, "n_nan": 0}
    finite = np.isfinite(y)
    n_nan = int((~finite).sum())
    if not finite.any():
        return {k: float("nan") for k in
                ("mean", "std", "var", "min", "max", "ptp")} | {"n": int(y.size),
                                                                "n_nan": n_nan}
    v = y[finite]
    return {
        "n": int(y.size), "n_nan": n_nan,
        "mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "var": float(v.var(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()), "max": float(v.max()),
        "ptp": float(v.max() - v.min()),
    }


def fmt_stats(s: dict) -> str:
    if s["n"] == 0:
        return "sin datos en la ventana"
    def f(v):
        return "—" if not np.isfinite(v) else f"{v:.4g}"
    txt = (f"μ {f(s['mean'])}   σ {f(s['std'])}   var {f(s['var'])}   "
           f"min {f(s['min'])}   max {f(s['max'])}   n {s['n']}")
    if s["n_nan"]:
        txt += f"   faltan {s['n_nan']}"
    return txt
