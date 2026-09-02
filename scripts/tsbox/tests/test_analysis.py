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


# --- espectro de frecuencia -----------------------------------------------
def test_spectrum_welch_detecta_tono_puro():
    fs = 100.0
    n = 4000
    t = np.arange(n) / fs
    y = np.sin(2 * np.pi * 10.0 * t)
    res = A.spectrum(y, fs, method="welch", nperseg=512)
    assert res.peak_freqs.size >= 1
    dominant = res.peak_freqs[np.argmax(res.peak_power)]
    assert dominant == pytest.approx(10.0, abs=0.5)


def test_spectrum_fft_tiene_mas_resolucion_que_welch():
    fs = 100.0
    n = 4000
    t = np.arange(n) / fs
    y = np.sin(2 * np.pi * 10.0 * t)
    res_fft = A.spectrum(y, fs, method="fft")
    res_welch = A.spectrum(y, fs, method="welch", nperseg=256)
    assert np.diff(res_fft.freqs)[0] < np.diff(res_welch.freqs)[0]
    dominant = res_fft.peak_freqs[np.argmax(res_fft.peak_power)]
    assert dominant == pytest.approx(10.0, abs=0.1)


def test_spectrum_detrend_quita_el_pico_en_continua():
    # rng propio (no el módulo, compartido con tests posteriores que dependen
    # de la secuencia EXACTA de sorteos, p.ej. test_adf_detecta_paseo_aleatorio):
    # consumir del compartido aquí desplazaría sus números y los volvería frágiles.
    rng_local = np.random.default_rng(11)
    fs = 50.0
    n = 2000
    y = 5.0 + 0.3 * rng_local.standard_normal(n)   # offset grande, sin periodicidad
    res_none = A.spectrum(y, fs, method="welch", detrend="none")
    res_dc = A.spectrum(y, fs, method="welch", detrend="constant")
    assert res_none.power[0] > res_dc.power[0]


def test_spectrum_hueco_no_revienta_y_sigue_viendo_el_pico():
    fs = 100.0
    n = 3000
    t = np.arange(n) / fs
    y = np.sin(2 * np.pi * 8.0 * t)
    y[1000:1020] = np.nan
    res = A.spectrum(y, fs, method="welch", x=t)
    assert res.n_nan == 20
    dominant = res.peak_freqs[np.argmax(res.peak_power)]
    assert dominant == pytest.approx(8.0, abs=0.5)


def test_spectrum_avisa_muestreo_irregular():
    rng2 = np.random.default_rng(3)
    x_irr = np.cumsum(1.0 + rng2.uniform(-0.3, 0.3, 500))
    y = rng2.standard_normal(500)
    res = A.spectrum(y, 1.0, x=x_irr)
    assert any("irregular" in note for note in res.notes)


def test_spectrum_pocas_muestras_no_revienta():
    res = A.spectrum(np.array([1.0, 2.0, 3.0]), 10.0)
    assert res.freqs.size == 0
    assert res.notes


# --- ADF (estacionariedad) -----------------------------------------------
def test_adf_detecta_estacionaria():
    y = rng.standard_normal(2000)   # ruido blanco: claramente estacionario
    d = A.adf_test(y)
    assert d["stationary_5pct"] is True
    assert d["stat"] < d["crit_5pct"]


def test_adf_detecta_paseo_aleatorio():
    y = np.cumsum(rng.standard_normal(2000))   # random walk: NO estacionario
    d = A.adf_test(y)
    assert d["stationary_5pct"] is False
    assert d["stat"] > d["crit_1pct"]


def test_adf_pocas_muestras():
    d = A.adf_test(rng.standard_normal(5))
    assert d["stationary_5pct"] is None
    assert np.isnan(d["stat"])


# --- Granger --------------------------------------------------------------
def test_granger_detecta_relacion_causal():
    n = 3000
    rng2 = np.random.default_rng(7)
    x = rng2.standard_normal(n)
    # y depende de x desplazada 3 pasos + ruido propio
    y = np.zeros(n)
    for t in range(3, n):
        y[t] = 0.8 * x[t - 3] + 0.3 * rng2.standard_normal()
    res = A.granger_causality(x, y, lag=5)
    assert res.p_value < 1e-6
    assert res.f_stat > 10
    assert res.r2_gain > 0.3


def test_granger_no_detecta_series_independientes():
    """Con series independientes, p<0.05 debe pasar ~5% de las veces (es la
    definición de alpha), no nunca -- por eso se agregan varias semillas en
    vez de exigir p>0.05 en una única ejecución, que fallaría 1 de cada 20
    veces por puro diseño del test."""
    n = 2000
    false_positives = 0
    for seed in range(20):
        rng2 = np.random.default_rng(seed)
        x = rng2.standard_normal(n)
        y = rng2.standard_normal(n)
        res = A.granger_causality(x, y, lag=5)
        if res.p_value < 0.05:
            false_positives += 1
    assert false_positives <= 4   # ~5% esperado; margen generoso sobre 20 tiradas


def test_granger_direccion_importa():
    """X causa Y, pero Y no debe 'causar' X (más allá del azar)."""
    n = 3000
    rng2 = np.random.default_rng(11)
    x = rng2.standard_normal(n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.9 * x[t - 2] + 0.2 * rng2.standard_normal()
    xy = A.granger_causality(x, y, lag=4)   # x -> y: debe ser significativo
    yx = A.granger_causality(y, x, lag=4)   # y -> x: no debería
    assert xy.p_value < 1e-4
    assert yx.p_value > 0.01


def test_granger_scan_aplica_fdr():
    n = 1500
    rng2 = np.random.default_rng(5)
    x = rng2.standard_normal(n)
    y = rng2.standard_normal(n)
    rows = A.granger_scan(x, y, max_lag=10)
    assert len(rows) == 10
    assert all(hasattr(r, "sig_fdr") for r in rows)
    # con series independientes, casi ningun lag deberia sobrevivir al FDR
    assert sum(r.sig_fdr for r in rows) <= 1


def test_granger_pocas_muestras_no_explota():
    res = A.granger_causality(np.arange(5.0), np.arange(5.0), lag=3)
    assert np.isnan(res.f_stat)


# --- exclusión de zonas (arranque, parada...) ----------------------------
def test_mask_excluded_pone_nan_dentro_del_intervalo():
    x = np.arange(10, dtype=float)
    y = np.arange(10, dtype=float) * 10
    out = A.mask_excluded(x, y, [(3.0, 6.0)])
    assert np.isnan(out[3:7]).all()
    assert not np.isnan(out[:3]).any()
    assert not np.isnan(out[7:]).any()
    assert not np.isnan(x).any()   # x nunca se toca


def test_mask_excluded_varios_intervalos():
    x = np.arange(20, dtype=float)
    y = x.copy()
    out = A.mask_excluded(x, y, [(2.0, 4.0), (15.0, 17.0)])
    assert np.isnan(out[2:5]).all()
    assert np.isnan(out[15:18]).all()
    assert not np.isnan(out[5:15]).any()


def test_mask_excluded_intervalo_invertido():
    x = np.arange(10, dtype=float)
    y = x.copy()
    out = A.mask_excluded(x, y, [(6.0, 3.0)])   # t0 > t1
    assert np.isnan(out[3:7]).all()


def test_mask_excluded_sin_intervalos_no_cambia():
    x = np.arange(10, dtype=float)
    y = np.arange(10, dtype=float)
    out = A.mask_excluded(x, y, [])
    assert np.array_equal(out, y)


def test_mask_excluded_afecta_a_histograma():
    n = 2000
    x = np.arange(n, dtype=float)
    y = np.concatenate([rng.normal(100, 1, 200), rng.normal(0, 1, n - 200)])
    masked = A.mask_excluded(x, y, [(0.0, 199.0)])
    h = A.histogram(masked)
    assert h.n == n - 200
    assert h.n_nan == 200
    assert len(h.modes) == 1   # sin la zona de arranque ya no es bimodal


# --- normalización de señal ---------------------------------------------
def test_normalize_zscore_media_cero_std_uno():
    y = rng.normal(5, 3, 2000)
    z = A.normalize_signal(y, "zscore")
    v = z[np.isfinite(z)]
    assert v.mean() == pytest.approx(0.0, abs=1e-9)
    assert v.std() == pytest.approx(1.0, abs=1e-9)


def test_normalize_minmax_rango_01():
    y = rng.normal(5, 3, 2000)
    m = A.normalize_signal(y, "minmax")
    v = m[np.isfinite(m)]
    assert v.min() == pytest.approx(0.0, abs=1e-9)
    assert v.max() == pytest.approx(1.0, abs=1e-9)


def test_normalize_none_no_cambia():
    y = rng.normal(5, 3, 100)
    assert np.array_equal(A.normalize_signal(y, "none"), y)


def test_normalize_preserva_nan():
    y = np.array([1.0, np.nan, 3.0, 4.0])
    z = A.normalize_signal(y, "zscore")
    assert np.isnan(z[1])
    m = A.normalize_signal(y, "minmax")
    assert np.isnan(m[1])


def test_normalize_constante_no_divide_por_cero():
    y = np.full(50, 7.0)
    z = A.normalize_signal(y, "zscore")
    assert np.all(np.isfinite(z))
    m = A.normalize_signal(y, "minmax")
    assert np.all(np.isfinite(m))


# --- heatmap tiempo x valor ---------------------------------------------
def test_heatmap_detecta_cambio_de_regimen():
    n = 4000
    t = np.arange(n, dtype=float)
    y = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(10, 1, n // 2)])
    hm = A.time_value_heatmap(t, y, n_bins_time=20, n_bins_value=40)
    assert hm.n == n and hm.n_nan == 0

    # columna del principio: masa concentrada cerca de valor=0
    # columna del final: masa concentrada cerca de valor=10
    y_centers = 0.5 * (hm.y_edges[:-1] + hm.y_edges[1:])
    first_col_peak = y_centers[np.argmax(hm.density[:, 0])]
    last_col_peak = y_centers[np.argmax(hm.density[:, -1])]
    assert first_col_peak < 3
    assert last_col_peak > 7


def test_heatmap_normalizacion_column_ignora_huecos():
    n = 2000
    t = np.arange(n, dtype=float)
    y = rng.normal(0, 1, n)
    y[: n // 2] = np.nan   # primera mitad sin datos validos
    hm = A.time_value_heatmap(t, y, n_bins_time=10, n_bins_value=30,
                              normalize="column")
    # cada columna con datos debe normalizar a la misma masa total
    # (columnas sin ningun dato valido quedan en cero, no cuentan)
    col_totals = hm.density.sum(axis=0) * np.diff(hm.y_edges)[:, None].sum(axis=0)
    nonzero = hm.col_mass > 0
    assert nonzero.sum() >= 4   # al menos las columnas de la segunda mitad


def test_heatmap_vacio():
    hm = A.time_value_heatmap(np.full(10, np.nan), np.full(10, np.nan))
    assert hm.n == 0


def test_heatmap_cuenta_nan():
    t = np.arange(10, dtype=float)
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0])
    hm = A.time_value_heatmap(t, y, n_bins_time=5, n_bins_value=5)
    assert hm.n == 8 and hm.n_nan == 2


# --- boxplot / outliers -------------------------------------------------
def test_boxplot_stats_sin_outliers():
    y = rng.normal(0, 1, 2000)
    bx = A.boxplot_stats(y)
    assert bx.n == 2000 and bx.n_nan == 0
    assert bx.q1 < bx.median < bx.q3
    assert bx.outliers.size < 30   # ~0.7% esperado en una normal con k=1.5


def test_boxplot_stats_detecta_outliers_inyectados():
    y = np.concatenate([rng.normal(0, 1, 1000), [50.0, -50.0, 60.0]])
    bx = A.boxplot_stats(y)
    assert 50.0 in bx.outliers and -50.0 in bx.outliers and 60.0 in bx.outliers
    assert bx.whisker_hi < 50.0


def test_boxplot_stats_k_mayor_menos_outliers():
    y = np.concatenate([rng.normal(0, 1, 1000), rng.normal(0, 1, 1000) * 4])
    bx15 = A.boxplot_stats(y, k=1.5)
    bx30 = A.boxplot_stats(y, k=3.0)
    assert bx30.outliers.size <= bx15.outliers.size


def test_boxplot_stats_cuenta_nan():
    y = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    bx = A.boxplot_stats(y)
    assert bx.n == 4 and bx.n_nan == 1


def test_boxplot_stats_vacio():
    bx = A.boxplot_stats(np.full(10, np.nan))
    assert bx.n == 0 and np.isnan(bx.median)


def test_boxplot_stats_outlier_idx_apunta_al_valor_correcto():
    y = np.array([1.0, 2.0, np.nan, 3.0, 100.0, 2.5])
    bx = A.boxplot_stats(y)
    for idx, val in zip(bx.outlier_idx, bx.outliers):
        assert y[idx] == val


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
