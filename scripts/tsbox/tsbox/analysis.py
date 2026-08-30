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
