import numpy as np
import pytest

from tsbox import analysis as A


rng = np.random.default_rng(0)


def ar1(n, phi=0.9, seed=1):
    r = np.random.default_rng(seed)
    e = r.standard_normal(n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y


# --- histograma / modos ------------------------------------------------
def test_histogram_detecta_bimodal():
    y = np.concatenate([rng.normal(-5, 0.5, 3000), rng.normal(5, 0.5, 3000)])
    h = A.histogram(y)
    assert h.n == 6000 and h.n_nan == 0
    assert len(h.modes) == 2
    assert sorted(round(m) for m in h.modes) == [-5, 5]


def test_histogram_unimodal():
    h = A.histogram(rng.normal(0, 1, 5000))
    assert len(h.modes) == 1


def test_histogram_cuenta_nan():
    y = np.array([1.0, 2.0, np.nan, 4.0])
    h = A.histogram(y, bins=4, kde=False)
    assert h.n == 3 and h.n_nan == 1


def test_histogram_vacio():
    h = A.histogram(np.full(10, np.nan))
    assert h.n == 0


# --- ACF / PACF --------------------------------------------------------
def test_acf_ruido_blanco():
    r = A.acf(rng.standard_normal(4000), 20)
    assert r[0] == pytest.approx(1.0)
    assert np.abs(r[1:]).max() < 0.1


def test_acf_ar1_decae_geometricamente():
    r = A.acf(ar1(8000, 0.8), 10)
    assert r[1] == pytest.approx(0.8, abs=0.06)
    assert r[2] == pytest.approx(0.64, abs=0.08)


def test_pacf_ar1_corta_en_lag1():
    p = A.pacf(ar1(8000, 0.8), 10)
    assert p[1] == pytest.approx(0.8, abs=0.06)
    assert np.abs(p[2:]).max() < 0.1


def test_conf_bartlett_crece():
    y = ar1(2000, 0.9)
    r = A.acf(y, 30)
    c = A.acf_conf(r, 2000)
    assert c[-1] > c[1]


def test_ljung_box_distingue():
    _, p_white = A.ljung_box(rng.standard_normal(2000))
    _, p_ar = A.ljung_box(ar1(2000, 0.8))
    assert p_white > 0.05
    assert p_ar < 1e-6


# --- correlación y significancia --------------------------------------
def test_corr_exacta():
    x = np.arange(200.0)
    d = A.corr_with_p(x, 2 * x + 1, adjust_autocorr=False)
    assert d["r"] == pytest.approx(1.0)


def test_n_efectivo_menor_con_autocorrelacion():
    a, b = ar1(3000, 0.95, 1), ar1(3000, 0.95, 2)
    assert A.effective_n(a, b) < 300
    w1, w2 = rng.standard_normal(3000), rng.standard_normal(3000)
    assert A.effective_n(w1, w2) > 2500


def test_correccion_evita_falso_positivo_de_paseo_aleatorio():
    """Dos paseos aleatorios independientes. Sin corrección salen
    'significativos'; con corrección, mucho menos."""
    hits_raw = hits_adj = 0
    for s in range(30):
        r = np.random.default_rng(100 + s)
        a = np.cumsum(r.standard_normal(1000))
        b = np.cumsum(r.standard_normal(1000))
        if A.corr_with_p(a, b, adjust_autocorr=False)["p"] < 0.05:
            hits_raw += 1
        if A.corr_with_p(a, b, adjust_autocorr=True)["p"] < 0.05:
            hits_adj += 1
    assert hits_raw > hits_adj
    assert hits_raw >= 20          # el test clásico se dispara casi siempre


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
    m = A.benjamini_hochberg(p, 0.05)
    assert m[0] and m[1] and not m[4]


def test_corr_matrix_shape_y_simetria():
    d = {f"s{i}": rng.standard_normal(500) for i in range(5)}
    d["copia"] = d["s0"] * 3 + 1
    M = A.corr_matrix(d)
    k = len(M.names)
    assert M.r.shape == (k, k)
    assert np.allclose(M.r, M.r.T, equal_nan=True)
    assert np.allclose(np.diag(M.r), 1.0)
    i, j = M.names.index("s0"), M.names.index("copia")
    assert M.r[i, j] == pytest.approx(1.0)
    assert M.sig_fdr[i, j]


def test_corr_matrix_maneja_nan():
    a = rng.standard_normal(300)
    b = a.copy()
    b[50:100] = np.nan
    M = A.corr_matrix({"a": a, "b": b})
    assert M.n[0, 1] == 250
    assert M.r[0, 1] == pytest.approx(1.0)


# --- CCF / lags --------------------------------------------------------
def test_ccf_encuentra_lag_conocido():
    n = 4000
    x = rng.standard_normal(n)
    y = np.roll(x, 7) + 0.1 * rng.standard_normal(n)
    res = A.ccf(x, y, maxlag=30, prewhitened=True)
    assert res.best_lag == 7
    assert res.best_r > 0.8
    assert res.best_p < 1e-6


def test_ccf_lag_negativo():
    n = 4000
    x = rng.standard_normal(n)
    y = np.roll(x, -5) + 0.1 * rng.standard_normal(n)
    assert A.ccf(x, y, maxlag=30).best_lag == -5


def test_ccf_sin_relacion_no_supera_banda():
    res = A.ccf(rng.standard_normal(3000), rng.standard_normal(3000), maxlag=40)
    assert abs(res.best_r) < 4 * res.conf


def test_prewhitening_limpia_picos_espurios():
    """Dos AR(1) independientes: sin prewhiten la CCF es ancha y alta."""
    a, b = ar1(3000, 0.95, 11), ar1(3000, 0.95, 22)
    crudo = A.ccf(a, b, maxlag=50, prewhitened=False)
    limpio = A.ccf(a, b, maxlag=50, prewhitened=True)
    assert np.abs(crudo.ccf).mean() > np.abs(limpio.ccf).mean()


def test_shift_y_lagged_corr():
    x = rng.standard_normal(1000)
    y = A.shift(x, 3)
    assert np.isnan(y[:3]).all()
    d = A.lagged_corr(x, y, 3, adjust_autocorr=False)
    assert d["r"] == pytest.approx(1.0)


def test_lag_table():
    n = 2000
    x = rng.standard_normal(n)
    y = np.roll(x, 4) + 0.2 * rng.standard_normal(n)
    rows = A.lag_table(x, y, list(range(-10, 11)))
    best = max(rows, key=lambda r: abs(r["r"]))
    assert best["lag"] == 4 and best["sig_fdr"]


def test_fill_gaps_interpola_interior():
    y = np.array([0.0, np.nan, 2.0, np.nan, np.nan, 5.0])
    out = A.fill_gaps(y)
    assert np.isfinite(out).all()
    assert out[1] == pytest.approx(1.0)
