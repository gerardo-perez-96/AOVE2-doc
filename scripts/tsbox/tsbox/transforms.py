"""Recetas y estadísticas. Se recalculan siempre desde el origen."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy import signal as _sig

from .gaps import median_step
from .model import (KIND_BUTTER, KIND_DERIVATIVE, KIND_LAG, KIND_ROLLING_MEAN,
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


def _interp_nan(y: np.ndarray) -> np.ndarray:
    """Interpola NaN interiores y extiende los extremos con el valor finito
    más cercano. A diferencia de fill_gaps (en analysis.py), NO recorta el
    array: un filtro derivado tiene que conservar la longitud y la
    correspondencia posicional con x, igual que el resto de recetas."""
    ok = np.isfinite(y)
    if ok.all():
        return y.copy()
    if ok.sum() < 2:
        return np.zeros_like(y)
    idx = np.arange(y.size)
    out = y.copy()
    # np.interp extiende con el valor del extremo más cercano fuera de rango,
    # así que huecos al principio o al final también quedan cubiertos.
    out[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    return out


def butterworth(x: np.ndarray, y: np.ndarray, btype: str = "low", order: int = 4,
                cutoff: float = 1.0, cutoff2: float | None = None,
                zero_phase: bool = True) -> np.ndarray:
    """Filtro IIR de Butterworth: paso bajo/alto con `cutoff`, paso banda o
    rechazo banda con `cutoff`+`cutoff2` como bordes de la banda.

    La frecuencia de muestreo se estima como 1 / mediana(diff(x)) -- si el
    eje X es tiempo en segundos, `cutoff` va en Hz; si es el índice de
    muestra, va en ciclos/muestra. Con muestreo muy irregular el corte real
    varía a lo largo de la señal; esto es una aproximación, igual que la
    ventana en muestras de rolling_mean lo es para muestreo irregular.

    zero_phase=True usa sosfiltfilt (adelante y atrás): no desplaza la señal
    en el tiempo, a cambio de "ver" un poco hacia delante -- vale para
    análisis a posteriori, no para tiempo real. zero_phase=False usa sosfilt
    (causal), con el retardo de fase típico de un IIR.

    Los huecos (NaN) contaminarían TODA la salida si se dejaran pasar tal
    cual: un IIR tiene respuesta al impulso infinita, así que un solo NaN se
    propaga -- con sosfiltfilt, incluso hacia atrás -- al resto del filtrado.
    Se interpolan antes de filtrar (ver _interp_nan) y se restauran a NaN
    justo en las mismas posiciones después, mismo criterio que derivative():
    el valor filtrado de un instante sin dato no existe.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 3:
        return np.full(n, np.nan)
    nan_mask = ~np.isfinite(y)
    yy = _interp_nan(y)

    step = median_step(np.asarray(x, dtype=np.float64))
    fs = 1.0 / step if step > 0 else 1.0
    nyq = fs / 2.0
    eps = 1e-9
    if btype in ("bandpass", "bandstop"):
        lo = float(min(cutoff, cutoff2 if cutoff2 is not None else cutoff))
        hi = float(max(cutoff, cutoff2 if cutoff2 is not None else cutoff))
        lo = np.clip(lo / nyq, eps, 1 - eps)
        hi = np.clip(hi / nyq, eps, 1 - eps)
        if hi <= lo:
            hi = min(1 - eps, lo + eps)
        wn = [lo, hi]
    else:
        wn = float(np.clip(cutoff / nyq, eps, 1 - eps))

    try:
        sos = _sig.butter(max(1, int(order)), wn, btype=btype, output="sos")
        out = (_sig.sosfiltfilt if zero_phase else _sig.sosfilt)(sos, yy)
    except (ValueError, np.linalg.LinAlgError):
        # Corte inválido para esta fs, u orden demasiado alto para la
        # longitud de la señal (padlen de sosfiltfilt). Un parámetro fuera
        # de rango no debe reventar la app -- se devuelve NaN, igual que un
        # histograma sin datos válidos devuelve "vacío" en vez de excepción.
        return np.full(n, np.nan)

    out = np.asarray(out, dtype=np.float64)
    out[nan_mask] = np.nan
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
    if kind == KIND_BUTTER:
        return butterworth(x, y, params.get("btype", "low"),
                           int(params.get("order", 4)),
                           float(params.get("cutoff", 1.0)),
                           params.get("cutoff2"),
                           bool(params.get("zero_phase", True)))
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
