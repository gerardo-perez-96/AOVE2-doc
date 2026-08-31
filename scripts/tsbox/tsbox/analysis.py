"""Análisis estadístico. Sin Qt: todo testeable en headless.

AVISO QUE IMPORTA MÁS QUE EL CÓDIGO
-----------------------------------
Las series temporales violan la hipótesis de independencia sobre la que se
construyen los p-valores de correlación de toda la vida. Dos paseos aleatorios
sin ninguna relación causal correlacionan a r=0.9 con p<1e-30. Eso no es un
descubrimiento: es el test aplicándose donde no toca.

Por eso aquí:
  * la significancia se calcula con tamaño muestral EFECTIVO (Bartlett),
    no con n.
  * la correlación cruzada ofrece pre-blanqueado (prewhitening) y avisa
    cuando no lo usas.
  * se corrigen los p-valores por comparaciones múltiples (Benjamini-Hochberg)
    en la matriz, porque con 20 series haces 190 tests y ~10 saldrán
    "significativos" por puro azar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal, stats


# ----------------------------------------------------------------------
# utilidades
# ----------------------------------------------------------------------
def clean(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float64)


def pair_finite(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Casos completos. Ojo: al tirar filas rompes el espaciado temporal,
    así que esto vale para correlación puntual, NO para ACF/CCF."""
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def fill_gaps(y: np.ndarray) -> np.ndarray:
    """Interpola NaN interiores linealmente y recorta los extremos.
    Necesario para ACF/CCF, que exigen muestreo continuo."""
    y = clean(y)
    ok = np.isfinite(y)
    if ok.all():
        return y
    if ok.sum() < 2:
        return np.empty(0)
    idx = np.arange(y.size)
    first, last = idx[ok][0], idx[ok][-1]
    out = y.copy()
    out[first:last + 1] = np.interp(idx[first:last + 1], idx[ok], y[ok])
    return out[first:last + 1]


def slice_window(x: np.ndarray, y: np.ndarray, x0=None, x1=None) -> np.ndarray:
    if x0 is None or x1 is None or x.size == 0:
        return y
    i0, i1 = np.searchsorted(x, [x0, x1])
    return y[max(0, int(i0)):min(y.size, int(i1) + 1)]


def mask_excluded(x: np.ndarray, y: np.ndarray,
                  intervals: list[tuple[float, float]]) -> np.ndarray:
    """Pone NaN en los instantes de `y` cuyo `x` cae dentro de algún
    intervalo excluido (ambos extremos incluidos). No recorta filas -- eso
    rompería la correspondencia posicional con `x`, que ACF/heatmap/etc.
    asumen; un NaN se ignora igual que cualquier otro dato faltante en el
    resto de este módulo.
    """
    if not intervals or x.size == 0:
        return y
    excluded = np.zeros(x.shape, dtype=bool)
    for t0, t1 in intervals:
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        excluded |= (x >= lo) & (x <= hi)
    if not excluded.any():
        return y
    out = y.astype(np.float64, copy=True)
    out[excluded] = np.nan
    return out


def normalize_signal(y: np.ndarray, mode: str = "none") -> np.ndarray:
    """Normaliza para comparar dispersión/separación entre regímenes en
    unidades relativas, no en las unidades originales de la señal.

    "zscore" -- (y - media) / std. Estándar para preguntar "¿qué tan
      separados están los regímenes en desviaciones típicas?".
    "minmax" -- (y - min) / (max - min), a [0, 1]. Sensible a outliers: un
      solo pico extremo aplasta la separación real entre regímenes.
    "none"  -- sin cambios.
    """
    if mode == "none":
        return y
    v = y[np.isfinite(y)]
    if v.size < 2:
        return y
    if mode == "zscore":
        sd = v.std()
        return (y - v.mean()) / sd if sd > 0 else y - v.mean()
    if mode == "minmax":
        lo, hi = v.min(), v.max()
        return (y - lo) / (hi - lo) if hi > lo else y - lo
    raise ValueError(f"modo de normalización desconocido: {mode!r}")


# ----------------------------------------------------------------------
# histograma y modos
# ----------------------------------------------------------------------
@dataclass
class HistResult:
    counts: np.ndarray
    edges: np.ndarray
    centers: np.ndarray
    kde_x: np.ndarray
    kde_y: np.ndarray
    modes: np.ndarray          # posiciones de los picos de densidad
    mode_weights: np.ndarray   # altura relativa de cada pico
    n: int
    n_nan: int
    bin_rule: str


# Fracción mínima de los datos que debe acumular un pico para contar como
# modo. Un nivel que la señal visita el 1% del tiempo es un régimen real
# (en una serie de 10k muestras son 100 visitas); por debajo de eso lo más
# probable es que sea un rizo de la KDE en la cola. Este es el filtro que
# de verdad implementa "un valor con muchas referencias a lo largo de la
# serie": cuenta visitas, no altura.
MIN_MODE_MASS = 0.01

# Altura mínima de un pico respecto al máximo. Filtra los rizos de la cola
# de una lognormal/exponencial, que pueden acumular masa pero no son un
# nivel al que la señal vuelva.
FLOOR_FRAC = 0.08

# Prominencia relativa a partir de la cual un pico se considera "bien
# definido": un valle profundo lo separa de lo que tiene al lado. Los
# rizos de una cola larga se quedan en 0.05-0.15; un nivel de operación
# real, aunque se visite poco, pasa de 0.9.
SOLID_PROM = 0.5


def freedman_diaconis_bins(v: np.ndarray) -> int:
    """Regla robusta a colas pesadas. 'sqrt(n)' o '10 bins' te esconden modos."""
    if v.size < 4:
        return max(1, v.size)
    q75, q25 = np.percentile(v, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return int(np.clip(np.sqrt(v.size), 10, 200))
    h = 2 * iqr / np.cbrt(v.size)
    if h <= 0:
        return int(np.clip(np.sqrt(v.size), 10, 200))
    return int(np.clip(np.ceil((v.max() - v.min()) / h), 10, 500))


def _area(y: np.ndarray, x: np.ndarray) -> float:
    """np.trapz se renombró a np.trapezoid en NumPy 2; soportamos ambos."""
    if y.size < 2:
        return 0.0
    f = getattr(np, "trapezoid", None) or np.trapz
    return float(f(y, x))


def _kde_bw(v: np.ndarray) -> float:
    """Ancho de banda para la KDE, robusto a modos muy separados.

    La regla de Scott que usa gaussian_kde por defecto escala con la
    *desviación global*. Si la señal pasa la mayor parte del tiempo en un
    nivel y visita otro muy distinto, esa sigma global la infla la
    SEPARACIÓN entre niveles, no la anchura de ninguno de ellos: con
    niveles en 0.05 y 0.9 la sigma sale ~0.29 aunque cada nivel sea de
    ancho 0.01, y el suavizado se traga el nivel secundario.

    Usar sigma robusta = min(std, IQR/1.349) ata el ancho de banda a la
    dispersión del grueso de los datos, no al recorrido total. Para una
    normal ambas coinciden, así que el caso unimodal no cambia; en cuanto
    hay dos niveles separados, el IQR ignora el salto y el ancho de banda
    se queda pequeño -- que es lo que deja ver los dos picos.
    """
    n = v.size
    sd = float(np.std(v))
    q75, q25 = np.percentile(v, [75, 25])
    iqr = float(q75 - q25)
    sigma = min(sd, iqr / 1.349) if iqr > 0 else sd
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = sd
    if not np.isfinite(sigma) or sigma <= 0:
        return 1.0
    # Silverman sobre la sigma robusta, devuelto como FACTOR relativo a la
    # std global, que es lo que gaussian_kde espera en bw_method.
    h = 0.9 * sigma * n ** (-0.2)
    return float(np.clip(h / sd, 1e-3, 1.0)) if sd > 0 else 1.0


def _find_modes(kx: np.ndarray, ky: np.ndarray,
                prominence: float) -> tuple[np.ndarray, np.ndarray]:
    """Picos de la densidad, con el umbral medido contra el pico LOCAL.

    Escalar el umbral por el pico más alto (`prominence * ky.max()`) hace
    que un nivel raro pero perfectamente definido sea invisible: si el
    98% del tiempo estás en un nivel, ese pico es altísimo y cualquier
    otro nivel -- aunque la señal lo visite 300 veces y se vea clarísimo
    en el gráfico -- queda por debajo del 8% de ESE máximo y se descarta.
    Es justo el caso que se reportó: un valor con muchísimas referencias a
    lo largo de la serie que no salía como modo.

    El criterio correcto es relativo al propio pico: un pico cuenta si
    sobresale de su entorno al menos `prominence` de su PROPIA altura.
    Eso mide "¿está bien separado de lo que tiene al lado?", que es lo que
    hace que un ojo lo lea como modo, y no "¿compite en altura con el
    pico dominante?", que es otra pregunta.

    Pero "relativo a su propia altura" a secas es demasiado permisivo: una
    ondulación mínima en la cola de una exponencial también sobresale el
    100% de su propia (minúscula) altura. Por eso un pico entra si cumple
    el criterio local Y además tiene MASA suficiente: la fracción de datos
    bajo su entorno debe llegar a MIN_MODE_MASS. Eso es lo que distingue
    "un nivel que la señal visita de verdad muchas veces" de "un rizo de la
    KDE": la pregunta del usuario era justamente sobre un valor con muchas
    referencias a lo largo de la serie, y la masa es la que las cuenta.
    """
    if ky.size == 0 or ky.max() <= 0:
        return np.empty(0), np.empty(0)
    pk, props = signal.find_peaks(ky, prominence=0.0)
    if pk.size == 0:
        return np.empty(0), np.empty(0)
    prom = props["prominences"]
    # Criterio local: sobresale de su entorno respecto a su propia altura.
    local_ok = prom >= prominence * ky[pk]
    # Masa: fracción de datos que acumula el pico en SU cuenca, delimitada
    # por los valles inmediatos a cada lado. Las bases que devuelve
    # find_peaks no sirven aquí: para un rizo en la cola de una
    # exponencial la base se extiende hasta el arranque de la curva, y la
    # masa sale inflada por densidad que no es suya.
    total = _area(ky, kx)
    valleys = signal.find_peaks(-ky)[0]
    mass = np.empty(pk.size)
    for i, p_i in enumerate(pk):
        left = valleys[valleys < p_i]
        right = valleys[valleys > p_i]
        lo = int(left[-1]) if left.size else 0
        hi = int(right[0]) + 1 if right.size else ky.size
        mass[i] = _area(ky[lo:hi], kx[lo:hi])
    mass = mass / total if total > 0 else np.zeros_like(mass)
    mass_ok = mass >= MIN_MODE_MASS
    # Tercer filtro, contra el máximo global. Existe por las colas largas
    # (lognormal, exponencial): ahí los rizos de la KDE pueden acumular
    # masa suficiente sin ser un nivel al que la señal vuelva.
    #
    # Pero un nivel poco frecuente ES bajito por definición -- un nivel que
    # ocupa el 5% del tiempo tiene un pico ~20 veces menor que el
    # dominante -- así que aplicar el suelo a todos escondería justo lo que
    # se quiere ver. Por eso el suelo se levanta para los picos BIEN
    # definidos: si un pico está claramente separado de su entorno
    # (prominencia relativa alta) y además acumula masa de sobra, es un
    # nivel real por bajo que sea, y el suelo no le aplica. Un rizo de cola
    # no cumple lo primero: su prominencia relativa es de 0.05-0.15.
    solid = prom >= SOLID_PROM * ky[pk]
    floor_ok = (ky[pk] >= FLOOR_FRAC * ky.max()) | solid
    keep = local_ok & mass_ok & floor_ok
    # El pico dominante entra siempre: una distribución con datos tiene al
    # menos un modo, y devolver cero modos por un umbral sería absurdo.
    if not keep.any():
        keep[int(np.argmax(ky[pk]))] = True
    return kx[pk[keep]], ky[pk[keep]] / ky.max()


def histogram(y: np.ndarray, bins: int | str = "auto",
              kde: bool = True, prominence: float = 0.08) -> HistResult:
    """Histograma + densidad suavizada + detección de modos.

    Los modos se buscan sobre la KDE, no sobre el histograma: los picos del
    histograma dependen del binning y te inventan multimodalidad.
    `prominence` es la fracción de su PROPIA altura que un pico debe
    sobresalir de su entorno (ver _find_modes); súbelo si ves modos
    fantasma. El ancho de banda lo fija _kde_bw, robusto a niveles muy
    separados.
    """
    y = clean(y)
    n_nan = int((~np.isfinite(y)).sum())
    v = y[np.isfinite(y)]
    if v.size == 0:
        e = np.array([0.0, 1.0])
        return HistResult(np.zeros(1), e, np.array([0.5]), np.empty(0),
                          np.empty(0), np.empty(0), np.empty(0), 0, n_nan, "vacío")

    if isinstance(bins, str):
        nb, rule = freedman_diaconis_bins(v), "Freedman-Diaconis"
    else:
        nb, rule = max(1, int(bins)), "manual"

    counts, edges = np.histogram(v, bins=nb)
    centers = 0.5 * (edges[:-1] + edges[1:])

    kx = ky = modes = weights = np.empty(0)
    if kde and v.size >= 20 and np.ptp(v) > 0:
        try:
            kd = stats.gaussian_kde(v, bw_method=_kde_bw(v))
            kx = np.linspace(v.min(), v.max(), 512)
            ky = kd(kx)
            if ky.max() > 0:
                modes, weights = _find_modes(kx, ky, prominence)
        except Exception:
            kx = ky = np.empty(0)

    return HistResult(counts, edges, centers, kx, ky, modes, weights,
                      int(v.size), n_nan, rule)


# ----------------------------------------------------------------------
# heatmap tiempo x valor (cambios de régimen)
# ----------------------------------------------------------------------
@dataclass
class HeatmapResult:
    density: np.ndarray     # (n_bins_valor, n_bins_tiempo), lista para imshow
    x_edges: np.ndarray     # n_bins_tiempo + 1
    y_edges: np.ndarray     # n_bins_valor + 1
    n: int
    n_nan: int
    col_mass: np.ndarray    # muestras válidas por columna de tiempo


def time_value_heatmap(x: np.ndarray, y: np.ndarray,
                       n_bins_time: int = 100, n_bins_value: int = 50,
                       normalize: str = "column") -> HeatmapResult:
    """Histograma 2D tiempo x valor: para cada franja de tiempo, cómo se
    reparte la señal en valor. Un cambio de régimen se ve como un
    desplazamiento, ensanchamiento o desdoblamiento de la banda de color
    de una columna a la siguiente -- lo que un solo histograma global o una
    curva de media móvil no muestran, porque ambos aplanan la distribución
    dentro de su ventana.

    normalize:
      "column" (por defecto) -- cada columna de tiempo se normaliza a su
        propia masa (histograma de densidad independiente por franja). Es
        lo que hay que usar para comparar FORMA entre tramos: si un tramo
        tiene menos muestras válidas (huecos, NaN) no debe verse más tenue
        solo por eso.
      "global" -- densidad conjunta sin renormalizar por columna. Aquí un
        tramo con más datos válidos sí pesa más que uno con huecos; útil
        si lo que quieres ver es dónde hay más MUESTRAS, no la forma.
      "none" -- cuentas crudas, sin normalizar en ningún eje.
    """
    x, y = clean(x), clean(y)
    ok = np.isfinite(x) & np.isfinite(y)
    n_nan = int((~ok).sum())
    xv, yv = x[ok], y[ok]

    if xv.size == 0:
        xe = np.array([0.0, 1.0])
        ye = np.array([0.0, 1.0])
        return HeatmapResult(np.zeros((1, 1)), xe, ye, 0, n_nan, np.zeros(1))

    counts, x_edges, y_edges = np.histogram2d(
        xv, yv, bins=[max(1, n_bins_time), max(1, n_bins_value)])
    density = counts.T   # (valor, tiempo), lo que espera imshow con origin='lower'
    col_mass = counts.sum(axis=1)   # muestras válidas por columna de tiempo

    if normalize == "column":
        col_width = np.diff(y_edges)
        safe = np.where(col_mass > 0, col_mass, 1.0)
        density = density / safe[np.newaxis, :] / col_width[:, np.newaxis]
    elif normalize == "global":
        total = xv.size
        area = np.diff(x_edges).mean() * np.diff(y_edges).mean()
        density = density / max(1, total) / max(1e-12, area)

    return HeatmapResult(density, x_edges, y_edges, int(xv.size), n_nan, col_mass)


# ----------------------------------------------------------------------
# boxplot / outliers
# ----------------------------------------------------------------------
@dataclass
class BoxStats:
    q1: float
    median: float
    q3: float
    whisker_lo: float          # último dato dentro de q1 - k·IQR
    whisker_hi: float          # último dato dentro de q3 + k·IQR
    outliers: np.ndarray       # valores fuera de los bigotes
    outlier_idx: np.ndarray    # sus índices en el y de entrada (tras quitar NaN)
    n: int
    n_nan: int
    k: float


def boxplot_stats(y: np.ndarray, k: float = 1.5) -> BoxStats:
    """Cuartiles y outliers a la Tukey: bigotes en q1/q3 ± k·IQR, recortados
    al dato real más cercano (no al límite teórico, que puede caer fuera del
    rango de la serie). k=1.5 es el criterio clásico; k=3 marca solo los
    outliers "extremos".
    """
    y = clean(y)
    n_nan = int((~np.isfinite(y)).sum())
    finite_idx = np.flatnonzero(np.isfinite(y))
    v = y[finite_idx]
    if v.size == 0:
        return BoxStats(np.nan, np.nan, np.nan, np.nan, np.nan,
                        np.empty(0), np.empty(0, dtype=int), 0, n_nan, k)

    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - k * iqr, q3 + k * iqr

    inside = v[(v >= lo_fence) & (v <= hi_fence)]
    whisker_lo = float(inside.min()) if inside.size else float(v.min())
    whisker_hi = float(inside.max()) if inside.size else float(v.max())

    out_mask = (v < lo_fence) | (v > hi_fence)
    return BoxStats(float(q1), float(med), float(q3), whisker_lo, whisker_hi,
                    v[out_mask], finite_idx[out_mask], int(v.size), n_nan, k)


# ----------------------------------------------------------------------
# autocorrelación
# ----------------------------------------------------------------------
def acf(y: np.ndarray, nlags: int = 50, demean: bool = True) -> np.ndarray:
    """ACF sesgada (divide por n), que es la que garantiza definida positiva."""
    y = fill_gaps(y)
    n = y.size
    if n < 3:
        return np.zeros(min(nlags, 1) + 1)
    nlags = int(min(nlags, n - 1))
    v = y - y.mean() if demean else y
    denom = np.dot(v, v)
    if denom == 0:
        return np.zeros(nlags + 1)
    f = np.fft.rfft(v, n=2 * n)
    r = np.fft.irfft(f * np.conj(f), n=2 * n)[:nlags + 1]
    return r / denom


def pacf(y: np.ndarray, nlags: int = 50) -> np.ndarray:
    """PACF por Durbin-Levinson sobre la ACF."""
    r = acf(y, nlags)
    k = r.size - 1
    out = np.zeros(k + 1)
    out[0] = 1.0
    if k == 0:
        return out
    phi = np.zeros((k + 1, k + 1))
    phi[1, 1] = r[1]
    out[1] = r[1]
    v = 1.0 - r[1] ** 2
    for m in range(2, k + 1):
        if v <= 1e-12:
            break
        num = r[m] - np.dot(phi[m - 1, 1:m], r[m - 1:0:-1])
        phi[m, m] = num / v
        phi[m, 1:m] = phi[m - 1, 1:m] - phi[m, m] * phi[m - 1, m - 1:0:-1]
        v *= (1.0 - phi[m, m] ** 2)
        out[m] = phi[m, m]
    return out


def acf_conf(r: np.ndarray, n: int, alpha: float = 0.05,
             bartlett: bool = True) -> np.ndarray:
    """Banda de confianza. Bartlett ensancha con el lag: es lo honesto,
    la banda plana ±1.96/√n solo vale bajo hipótesis de ruido blanco."""
    z = stats.norm.ppf(1 - alpha / 2)
    if n <= 1:
        return np.zeros_like(r)
    if not bartlett:
        return np.full_like(r, z / np.sqrt(n))
    cum = np.concatenate([[0.0], np.cumsum(2 * r[1:] ** 2)])
    return z * np.sqrt((1 + cum) / n)


def pacf_conf(k: int, n: int, alpha: float = 0.05) -> np.ndarray:
    z = stats.norm.ppf(1 - alpha / 2)
    return np.full(k, z / np.sqrt(max(n, 1)))


def ljung_box(y: np.ndarray, lags: int = 20) -> tuple[float, float]:
    """H0: no hay autocorrelación. Si p es minúsculo, la serie NO es ruido
    blanco y cualquier p-valor de correlación clásico que calcules miente."""
    y = fill_gaps(y)
    n = y.size
    if n < 10:
        return float("nan"), float("nan")
    lags = int(min(lags, n // 4))
    r = acf(y, lags)[1:]
    k = np.arange(1, lags + 1)
    q = n * (n + 2) * np.sum(r ** 2 / (n - k))
    return float(q), float(1 - stats.chi2.cdf(q, lags))


# ----------------------------------------------------------------------
# tamaño muestral efectivo y correlación con significancia
# ----------------------------------------------------------------------
def lag1(y: np.ndarray) -> float:
    r = acf(y, 1)
    return float(r[1]) if r.size > 1 else 0.0


def effective_n(a: np.ndarray, b: np.ndarray) -> float:
    """Bartlett / Dawdy-Matalas: n_eff = n · (1−ρ1ρ2)/(1+ρ1ρ2).

    Analogía: preguntar a 1000 personas de la misma familia no son 1000
    opiniones independientes. Con series autocorreladas, 1000 muestras
    consecutivas pueden valer como 40 muestras independientes.
    """
    n = min(a.size, b.size)
    if n < 10:
        return float(n)
    ra, rb = lag1(a), lag1(b)
    prod = np.clip(ra * rb, -0.99, 0.99)
    return float(np.clip(n * (1 - prod) / (1 + prod), 3.0, n))


def corr_with_p(a: np.ndarray, b: np.ndarray, method: str = "pearson",
                adjust_autocorr: bool = True) -> dict:
    a, b = pair_finite(clean(a), clean(b))
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"r": float("nan"), "p": float("nan"), "n": int(a.size),
                "n_eff": float(a.size)}
    if method == "spearman":
        r = float(stats.spearmanr(a, b).statistic)
    elif method == "kendall":
        r = float(stats.kendalltau(a, b).statistic)
    else:
        r = float(np.corrcoef(a, b)[0, 1])

    ne = effective_n(a, b) if adjust_autocorr else float(a.size)
    if ne <= 3 or abs(r) >= 1:
        p = 0.0 if abs(r) >= 1 else float("nan")
    else:
        t = r * np.sqrt((ne - 2) / max(1e-12, 1 - r ** 2))
        p = float(2 * stats.t.sf(abs(t), ne - 2))
    return {"r": r, "p": p, "n": int(a.size), "n_eff": ne}


def benjamini_hochberg(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Devuelve máscara de rechazos controlando la tasa de falsos
    descubrimientos. Con 20 series haces 190 tests: sin esto, ~10 salen
    'significativos' aunque todo sea ruido."""
    p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(p)
    out = np.zeros(p.shape, dtype=bool)
    vals = p[ok]
    if vals.size == 0:
        return out
    order = np.argsort(vals)
    m = vals.size
    thr = alpha * (np.arange(1, m + 1) / m)
    passed = vals[order] <= thr
    kmax = np.flatnonzero(passed)
    if kmax.size:
        cut = vals[order][kmax[-1]]
        sel = vals <= cut
    else:
        sel = np.zeros(m, dtype=bool)
    out[ok] = sel
    return out


@dataclass
class CorrMatrix:
    names: list[str]
    r: np.ndarray
    p: np.ndarray
    n: np.ndarray
    n_eff: np.ndarray
    sig_raw: np.ndarray
    sig_fdr: np.ndarray
    method: str
    alpha: float
    adjusted: bool
    notes: list[str] = field(default_factory=list)


def corr_matrix(series: dict[str, np.ndarray], method: str = "pearson",
                alpha: float = 0.05, adjust_autocorr: bool = True) -> CorrMatrix:
    names = list(series.keys())
    k = len(names)
    r = np.full((k, k), np.nan)
    p = np.full((k, k), np.nan)
    n = np.zeros((k, k), dtype=int)
    ne = np.full((k, k), np.nan)
    np.fill_diagonal(r, 1.0)

    for i in range(k):
        for j in range(i + 1, k):
            d = corr_with_p(series[names[i]], series[names[j]], method,
                            adjust_autocorr)
            r[i, j] = r[j, i] = d["r"]
            p[i, j] = p[j, i] = d["p"]
            n[i, j] = n[j, i] = d["n"]
            ne[i, j] = ne[j, i] = d["n_eff"]

    iu = np.triu_indices(k, 1)
    sig_raw = np.zeros((k, k), dtype=bool)
    sig_fdr = np.zeros((k, k), dtype=bool)
    if k > 1:
        pu = p[iu]
        sig_raw[iu] = np.isfinite(pu) & (pu < alpha)
        sig_fdr[iu] = benjamini_hochberg(pu, alpha)
        sig_raw |= sig_raw.T
        sig_fdr |= sig_fdr.T

    notes = []
    ntests = k * (k - 1) // 2
    if ntests > 10:
        notes.append(f"{ntests} tests simultáneos: fíate de la columna FDR, "
                     f"no de p<{alpha} en crudo.")
    if adjust_autocorr and k > 1:
        ratio = np.nanmean(ne[iu]) / max(1, np.nanmax(n[iu]))
        if ratio < 0.3:
            notes.append(f"n efectivo medio es el {ratio:.0%} de n. Tus series "
                         "están muy autocorreladas; sin esta corrección todo "
                         "saldría significativo.")
    return CorrMatrix(names, r, p, n, ne, sig_raw, sig_fdr, method, alpha,
                      adjust_autocorr, notes)


# ----------------------------------------------------------------------
# correlación cruzada con desfase (lag)
# ----------------------------------------------------------------------
def ar_fit(y: np.ndarray, order: int) -> np.ndarray:
    """Coeficientes AR(p) por Yule-Walker."""
    r = acf(y, order)
    if r.size <= order:
        return np.zeros(order)
    R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
    try:
        return np.linalg.solve(R + np.eye(order) * 1e-10, r[1:order + 1])
    except np.linalg.LinAlgError:
        return np.zeros(order)


def prewhiten(x: np.ndarray, y: np.ndarray, order: int = 0
              ) -> tuple[np.ndarray, np.ndarray, int]:
    """Ajusta un AR(p) a x y aplica ESE MISMO filtro a x e y.

    Por qué: si x tiene inercia (hoy se parece a ayer) e y también, la CCF
    entre ambas sale llena de picos aunque no haya ninguna relación. Es como
    comparar dos ríos crecidos y concluir que uno causa el otro cuando lo que
    pasa es que ambos vienen de lluvias lentas. Prewhitening quita la inercia
    de x y deja solo la parte impredecible, que es la que puede informar de y.
    """
    x, y = clean(x), clean(y)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    if order <= 0:
        order = int(np.clip(round(n ** 0.25) * 2, 1, 20))
    phi = ar_fit(x - np.nanmean(x), order)
    if not np.any(phi):
        return x, y, 0
    def filt(v):
        v = v - np.nanmean(v)
        out = v[order:].copy()
        for k in range(1, order + 1):
            out -= phi[k - 1] * v[order - k:-k if k else None]
        return out
    return filt(x), filt(y), order


@dataclass
class CCFResult:
    lags: np.ndarray
    ccf: np.ndarray
    conf: float
    best_lag: int
    best_r: float
    best_p: float
    n: int
    prewhitened: bool
    ar_order: int
    notes: list[str] = field(default_factory=list)


def ccf(x: np.ndarray, y: np.ndarray, maxlag: int = 50,
        prewhitened: bool = True, ar_order: int = 0,
        alpha: float = 0.05) -> CCFResult:
    """Correlación cruzada. Convenio de signo: lag>0 significa que X ADELANTA
    a Y (X en t se compara con Y en t+lag), o sea X es candidato a predictor.
    """
    x, y = fill_gaps(clean(x)), fill_gaps(clean(y))
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    notes: list[str] = []
    used_order = 0

    if prewhitened:
        x, y, used_order = prewhiten(x, y, ar_order)
        n = x.size
        if used_order == 0:
            notes.append("X no tenía estructura AR detectable; sin pre-blanquear.")
    else:
        notes.append("SIN pre-blanquear: los picos anchos que veas pueden ser "
                     "inercia de las propias señales, no relación entre ellas.")

    maxlag = int(np.clip(maxlag, 1, max(1, n // 2 - 1)))
    if n < 10:
        return CCFResult(np.zeros(1), np.zeros(1), np.inf, 0, float("nan"),
                         float("nan"), n, prewhitened, used_order,
                         notes + ["Muy pocas muestras."])

    a = x - x.mean()
    b = y - y.mean()
    den = np.sqrt(np.dot(a, a) * np.dot(b, b))
    if den == 0:
        return CCFResult(np.zeros(1), np.zeros(1), np.inf, 0, float("nan"),
                         float("nan"), n, prewhitened, used_order,
                         notes + ["Señal constante."])

    full = signal.correlate(b, a, mode="full") / den   # índice = lag de X→Y
    mid = n - 1
    lags = np.arange(-maxlag, maxlag + 1)
    vals = full[mid - maxlag: mid + maxlag + 1]

    conf = float(stats.norm.ppf(1 - alpha / 2) / np.sqrt(n))
    k = int(np.nanargmax(np.abs(vals)))
    best_lag, best_r = int(lags[k]), float(vals[k])

    # p del pico, penalizado por haber mirado 2·maxlag+1 lags (Šidák).
    if abs(best_r) < 1:
        t = best_r * np.sqrt((n - 2) / (1 - best_r ** 2))
        p1 = float(2 * stats.t.sf(abs(t), n - 2))
        best_p = float(1 - (1 - p1) ** len(lags))
    else:
        best_p = 0.0

    if best_lag == lags[0] or best_lag == lags[-1]:
        notes.append("El pico está en el borde del rango de lags: amplía maxlag, "
                     "el máximo real puede estar fuera.")
    if abs(best_r) < conf:
        notes.append("Ningún lag supera la banda de confianza: no hay relación "
                     "lineal desfasada detectable.")
    return CCFResult(lags, vals, conf, best_lag, best_r, best_p, n,
                     prewhitened, used_order, notes)


def shift(y: np.ndarray, lag: int) -> np.ndarray:
    """Desplaza rellenando con NaN. lag>0 mueve la señal hacia el futuro."""
    y = clean(y)
    out = np.full_like(y, np.nan)
    if lag == 0:
        return y.copy()
    if lag > 0:
        out[lag:] = y[:-lag]
    else:
        out[:lag] = y[-lag:]
    return out


def lagged_corr(x: np.ndarray, y: np.ndarray, lag: int,
                method: str = "pearson", adjust_autocorr: bool = True) -> dict:
    """Correlación de X(t) con Y(t+lag). Devuelve r, p, n, n_eff."""
    if lag >= 0:
        a, b = clean(x)[:len(x) - lag or None], clean(y)[lag:]
    else:
        a, b = clean(x)[-lag:], clean(y)[:len(y) + lag or None]
    m = min(a.size, b.size)
    return corr_with_p(a[:m], b[:m], method, adjust_autocorr)


def lag_table(x: np.ndarray, y: np.ndarray, lags: list[int],
              method: str = "pearson", alpha: float = 0.05) -> list[dict]:
    rows = [dict(lag=L, **lagged_corr(x, y, L, method)) for L in lags]
    ps = np.array([r["p"] for r in rows])
    sig = benjamini_hochberg(ps, alpha)
    for r, s in zip(rows, sig):
        r["sig_fdr"] = bool(s)
        r["sig_raw"] = bool(np.isfinite(r["p"]) and r["p"] < alpha)
    return rows


# ----------------------------------------------------------------------
# estacionariedad (Dickey-Fuller aumentado) y causalidad de Granger
# ----------------------------------------------------------------------
# Valores críticos de MacKinnon (1994) para el estadístico ADF con
# constante, sin tendencia. Tabla asintótica (n grande); para n pequeño
# el test es algo conservador, pero clavar los polinomios de superficie
# de respuesta completos no aporta nada que un aviso cualitativo no dé
# aquí -- el punto es "¿decae la serie o se pasea?", no un p-valor exacto.
_ADF_CRIT = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mínimos cuadrados vía pseudo-inversa. Devuelve (coeficientes, residuos).

    lstsq (SVD, driver gelsd de LAPACK) puede lanzar LinAlgError si no
    converge -- pasa con matrices de diseño mal condicionadas, típico con
    lags altos donde y_{t-1}..y_{t-lag} son casi colineales entre sí. pinv
    también usa SVD pero recorta valores singulares pequeños en vez de
    exigir convergencia exacta: sale un resultado (con más varianza en
    columnas casi redundantes, que es justo lo que hay) en vez de reventar
    el escaneo de lags a mitad.
    """
    beta = np.linalg.pinv(X) @ y
    resid = y - X @ beta
    return beta, resid


def adf_test(y: np.ndarray, lags: int = 0) -> dict:
    """Dickey-Fuller aumentado, H0: la serie tiene raíz unitaria (NO es
    estacionaria -- tiene tendencia/paseo aleatorio, su varianza crece con
    el tiempo). Regresión: Δy_t = c + γ·y_{t-1} + Σφ_i·Δy_{t-i} + ε_t.
    Se rechaza H0 (serie estacionaria) si γ es significativamente negativo,
    es decir, si el estadístico t de γ cae por debajo del valor crítico.

    Por qué importa para Granger: con dos series NO estacionarias, casi
    cualquier regresión entre ellas sale "significativa" aunque no tengan
    relación real -- es la misma trampa de la correlación espuria descrita
    al principio de este módulo, aplicada a modelos autorregresivos.
    """
    y = fill_gaps(clean(y))
    n = y.size
    if n < 20:
        return {"stat": float("nan"), "p_approx": float("nan"),
                "n": n, "lags": 0, "stationary_5pct": None}

    if lags <= 0:
        lags = int(np.clip(round(12 * (n / 100) ** 0.25), 0, n // 3))

    dy = np.diff(y)
    y_lag1 = y[:-1]
    m = dy.size - lags
    if m < 10:
        lags = max(0, dy.size - 10)
        m = dy.size - lags

    rows = m
    X = np.ones((rows, 2 + lags))
    X[:, 1] = y_lag1[lags:lags + rows]
    for i in range(1, lags + 1):
        X[:, 1 + i] = dy[lags - i:lags - i + rows]
    target = dy[lags:lags + rows]

    beta, resid = _ols(X, target)
    dof = rows - X.shape[1]
    if dof <= 0:
        return {"stat": float("nan"), "p_approx": float("nan"),
                "n": n, "lags": lags, "stationary_5pct": None}
    sigma2 = float(np.dot(resid, resid)) / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    se_gamma = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    gamma = float(beta[1])
    t_stat = gamma / se_gamma if se_gamma > 0 else float("nan")

    return {"stat": t_stat, "n": n, "lags": lags,
            "stationary_5pct": bool(np.isfinite(t_stat) and t_stat < _ADF_CRIT[0.05]),
            "crit_1pct": _ADF_CRIT[0.01], "crit_5pct": _ADF_CRIT[0.05],
            "crit_10pct": _ADF_CRIT[0.10]}


def _lagmat(y: np.ndarray, lags: int) -> np.ndarray:
    """Matriz de columnas [y_{t-1}, ..., y_{t-lags}], alineada al final."""
    n = y.size - lags
    return np.column_stack([y[lags - i:lags - i + n] for i in range(1, lags + 1)])


@dataclass
class GrangerResult:
    lag: int
    f_stat: float
    p_value: float
    n: int
    rss_restricted: float     # solo pasado de Y
    rss_full: float           # pasado de Y + pasado de X
    r2_gain: float            # cuánta varianza extra explica X
    sig_fdr: bool = False     # solo lo rellena granger_scan()


def granger_causality(x: np.ndarray, y: np.ndarray, lag: int,
                      diff: bool = False) -> GrangerResult:
    """¿El pasado de X mejora la predicción de Y más allá de lo que ya
    predice el propio pasado de Y? Compara dos regresiones por mínimos
    cuadrados sobre Y_t:

      restringido:  Y_t = c + Σ a_i·Y_{t-i}
      completo:     Y_t = c + Σ a_i·Y_{t-i} + Σ b_i·X_{t-i}

    y hace un test F sobre si los b_i añaden algo. H0: b_i = 0 para todo i
    (X NO Granger-causa Y). Esto NO es causalidad en el sentido físico --
    solo dice "X pasado tiene poder predictivo incremental sobre Y", que
    puede deberse a una causa común no incluida en el modelo.

    diff=True aplica una diferencia (Δy_t = y_t - y_{t-1}) antes del test.
    Con series no estacionarias el test de Granger da falsos positivos
    sistemáticos -- exactamente el mismo problema que la correlación
    espuria de dos paseos aleatorios descrita al principio del módulo.
    Si adf_test() dice que la serie no es estacionaria, diferénciala.
    """
    x, y = clean(x), clean(y)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    if diff:
        x, y = np.diff(x), np.diff(y)

    rows = y.size - lag
    if rows < 3 * lag + 5:
        return GrangerResult(lag, float("nan"), float("nan"), y.size,
                             float("nan"), float("nan"), float("nan"))

    target = y[lag:]
    y_lags = _lagmat(y, lag)
    x_lags = _lagmat(x, lag)

    X_restricted = np.column_stack([np.ones(rows), y_lags])
    X_full = np.column_stack([np.ones(rows), y_lags, x_lags])

    try:
        _, resid_r = _ols(X_restricted, target)
        _, resid_f = _ols(X_full, target)
    except np.linalg.LinAlgError:
        # Colinealidad extrema (p. ej. una señal casi constante en la
        # ventana, o un lag mayor que la variación real de los datos) puede
        # dejar la matriz de diseño sin descomposición numérica estable.
        # Un lag roto no debe tumbar el resto del escaneo -- se marca como
        # sin resultado y se sigue con los demás.
        return GrangerResult(lag, float("nan"), float("nan"), rows,
                             float("nan"), float("nan"), float("nan"))
    rss_r = float(np.dot(resid_r, resid_r))
    rss_f = float(np.dot(resid_f, resid_f))

    df1 = lag                                   # parámetros añadidos (b_i)
    df2 = rows - X_full.shape[1]                # grados de libertad residuales
    if df2 <= 0 or rss_f <= 0:
        return GrangerResult(lag, float("nan"), float("nan"), rows,
                             rss_r, rss_f, float("nan"))

    f_stat = ((rss_r - rss_f) / df1) / (rss_f / df2)
    f_stat = max(0.0, f_stat)   # rss_f puede superar rss_r por ruido de muestra
    p_value = float(stats.f.sf(f_stat, df1, df2))
    r2_gain = float(max(0.0, (rss_r - rss_f) / rss_r)) if rss_r > 0 else 0.0

    return GrangerResult(lag, float(f_stat), p_value, rows, rss_r, rss_f, r2_gain)


def granger_scan(x: np.ndarray, y: np.ndarray, max_lag: int,
                 diff: bool = False, alpha: float = 0.05) -> list[GrangerResult]:
    """Granger para lags 1..max_lag, con BH sobre los p-valores: probar
    muchos lags es el mismo problema de comparaciones múltiples que probar
    muchos lags de CCF."""
    results = [granger_causality(x, y, L, diff) for L in range(1, max_lag + 1)]
    ps = np.array([r.p_value for r in results])
    sig = benjamini_hochberg(ps, alpha)
    for r, s in zip(results, sig):
        r.sig_fdr = bool(s)
    return results
