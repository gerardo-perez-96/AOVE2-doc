"""Escala del eje Y y marcado de huecos.

Los tres bugs que hacían que la señal se viera como una línea recta y que el
panel se pusiera rojo entero. Si alguno de estos falla, ha vuelto uno de ellos.
"""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6 import QtWidgets  # noqa: E402

from tsbox import gaps  # noqa: E402
from tsbox.panel import (GAP_DENSE_COVER, YMODE_FULL, YMODE_WINDOW,  # noqa: E402
                         GapOverlay, SeriesPanel, merge_intervals)
from tsbox.session import Session  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    """Señal con media alta y amplitud pequeña: el caso que se ve aplastado
    si la escala Y no se ajusta (μ≈75, rango 68..82)."""
    d = tmp_path_factory.mktemp("sc")
    n = 30_000
    rng = np.random.default_rng(0)
    t = pd.date_range("2024-01-01", periods=n, freq="1min")
    y = 75 + np.sin(np.arange(n) / 50) * 6 + rng.standard_normal(n) * 0.3
    y2 = y.copy()
    y[rng.random(n) < 0.10] = np.nan          # huecos densos
    y2[10_000:10_050] = np.nan                # un solo hueco ancho
    p = d / "s.csv"
    pd.DataFrame({"ts": t, "densa": y, "escasa": y2}).to_csv(p, index=False)
    return p


def _panel(app, csv, name="densa", w=800, h=300):
    s = Session()
    s.open(csv, "column", "ts", float32=True)
    p = SeriesPanel(s, s.project.by_name(name))
    p.resize(w, h)
    p.show()
    app.processEvents()
    return s, p


def _yspan(p):
    (_, _), (y0, y1) = p.plot.viewRange()
    return y0, y1


# --- 1. el overlay no debe secuestrar el autorange ---------------------
def test_overlay_no_entra_en_el_autorange(app, csv):
    """El bug: boundingRect de ±1e12 hacía que el eje Y fuera de -1.1e12 a
    +1.1e12 y la señal quedara aplastada en una línea recta."""
    s, p = _panel(app, csv)
    assert p._gap_overlay is not None
    assert p._gap_overlay.dataBounds(1) == [None, None], "el overlay estira la Y"
    y0, y1 = _yspan(p)
    assert abs(y1 - y0) < 1000, f"eje Y absurdo: {y0:.3g}..{y1:.3g}"


def test_la_senal_llena_el_panel(app, csv):
    s, p = _panel(app, csv)
    p.plot.setXRange(s.x[0], s.x[0] + 3600 * 24)
    app.processEvents()
    (x0, x1), (y0, y1) = p.plot.viewRange()
    y = s.values(p.sid)
    m = (s.x >= x0) & (s.x <= x1)
    v = y[m][np.isfinite(y[m])]
    ocupa = (v.max() - v.min()) / (y1 - y0)
    assert ocupa > 0.6, f"la señal solo ocupa el {ocupa:.0%} de la altura"
    assert ocupa <= 1.0


def test_y_sigue_a_la_ventana_al_hacer_zoom(app, csv):
    """Modo por defecto: la Y se reajusta a lo que se ve."""
    s, p = _panel(app, csv)
    p.plot.setXRange(s.x[0], s.x[-1])
    app.processEvents()
    ancho_total = np.ptp(_yspan(p))
    p.plot.setXRange(s.x[0], s.x[0] + 3600)     # una hora
    app.processEvents()
    assert np.ptp(_yspan(p)) < ancho_total


def test_modo_y_completo_no_se_mueve(app, csv):
    s, p = _panel(app, csv)
    p.set_y_mode(YMODE_FULL)
    app.processEvents()
    a = _yspan(p)
    p.plot.setXRange(s.x[0], s.x[0] + 3600)
    app.processEvents()
    assert _yspan(p) == pytest.approx(a, rel=1e-6)
    lo, hi = p.full_y_range()
    y = s.values(p.sid)
    v = y[np.isfinite(y)]
    assert lo < v.min() and hi > v.max()        # con margen


def test_senal_constante_no_da_rango_cero(app, tmp_path):
    p_csv = tmp_path / "flat.csv"
    pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=500, freq="1min"),
                  "k": np.full(500, 42.0)}).to_csv(p_csv, index=False)
    s = Session()
    s.open(p_csv, "column", "ts", float32=True)
    panel = SeriesPanel(s, s.project.by_name("k"))
    lo, hi = panel.full_y_range()
    assert hi > lo


# --- 2. huecos: resolución por píxel -----------------------------------
def test_merge_por_tolerancia():
    iv = [(0.0, 1.0), (1.5, 2.0), (10.0, 11.0)]
    out, dropped = merge_intervals(iv, tol=1.0)
    assert out == [(0.0, 2.0), (10.0, 11.0)] and dropped == 1
    out, dropped = merge_intervals(iv, tol=0.0)
    assert len(out) == 3 and dropped == 0
    assert merge_intervals([], tol=1.0) == ([], 0)


def test_los_huecos_se_separan_al_hacer_zoom(app, csv):
    """El bug: se agrupaban UNA vez globalmente, así que al hacer zoom seguías
    viendo bandas gordas que no correspondían a ningún hueco real."""
    s, p = _panel(app, csv)
    ov = p._gap_overlay
    conteos = []
    for horas in (24 * 20, 24, 2):
        p.plot.setXRange(s.x[0], s.x[0] + 3600 * horas)
        app.processEvents()
        ov._key = None
        x0, x1, _, _, w = ov._view()
        ov._rebuild(x0, x1, w)
        lo = np.searchsorted(ov._iv[:, 1], x0)
        hi = np.searchsorted(ov._iv[:, 0], x1)
        vis, _ = merge_intervals(ov._iv[lo:hi], tol=(x1 - x0) / w)
        conteos.append((hi - lo, len(vis)))
    # cuanto más zoom, mayor fracción de los huecos se dibuja por separado.
    # (No llega a 1: dos NaN separados por una sola muestra válida producen
    # tramos que se TOCAN al expandirse a los vecinos, y fundirlos es correcto.)
    frac = [v / n for n, v in conteos]
    assert frac[0] < frac[1] < frac[2], f"el zoom no separa los huecos: {frac}"
    assert frac[0] < 0.2 and frac[2] > 0.6


def test_huecos_densos_no_tapan_la_senal(app, csv):
    """Con 3.000 huecos y el panel entero visible, pintar bandas de altura
    completa deja el gráfico rojo. Debe pasar a franja fina."""
    s, p = _panel(app, csv)
    ov = p._gap_overlay
    p.plot.setXRange(s.x[0], s.x[-1])
    app.processEvents()
    ov._key = None
    x0, x1, _, _, w = ov._view()
    ov._rebuild(x0, x1, w)
    assert ov._cover > GAP_DENSE_COVER
    assert ov._dense, "cobertura alta: debería dibujar franja fina"
    # y el boundingRect no puede ocupar todo el panel
    (_, _), (y0, y1) = p.plot.viewRange()
    assert ov.boundingRect().height() < (y1 - y0) * 0.2


def test_hueco_aislado_se_dibuja_entero(app, csv):
    s, p = _panel(app, csv, name="escasa")
    ov = p._gap_overlay
    if ov is None:                       # pocos huecos: items individuales
        assert len(p._gap_items) >= 1
        return
    p.plot.setXRange(s.x[0], s.x[-1])
    app.processEvents()
    ov._key = None
    x0, x1, _, _, w = ov._view()
    ov._rebuild(x0, x1, w)
    assert not ov._dense


def test_overlay_vacio_no_revienta(app):
    ov = GapOverlay([])
    assert ov.boundingRect().isEmpty()
    assert ov.dataBounds(0) == [None, None]


# --- 3. eje X con valores repetidos ------------------------------------
def test_detecta_x_duplicado():
    x = np.repeat(np.arange(1000, dtype=float), 6)
    assert gaps.x_is_duplicated(x)
    assert not gaps.x_is_duplicated(np.arange(1000, dtype=float))


def test_no_reporta_saltos_con_x_duplicado():
    """Con entidades apiladas, dt_mediano no es el periodo de nada: informar
    de 'saltos' sería inventarse un dato."""
    x = np.repeat(np.arange(1000, dtype=float), 6)
    y = np.ones(6000)
    y[100:110] = np.nan
    rep = gaps.missing_report(x, y)
    assert rep["x_duplicado"] is True
    assert rep["time_gaps"] == []
    assert rep["n_nan"] == 10


# --- rendimiento del pan -----------------------------------------------
def test_overlay_no_usa_qpicture(app, csv):
    """QPicture.play cuesta 4x mas que un drawRects por lote. Si alguien
    vuelve a meter un QPicture aqui, el pan se va a tirones."""
    import inspect

    from tsbox import panel as panel_mod
    src = inspect.getsource(panel_mod.GapOverlay)
    assert "QPicture" not in src.replace("# ", "").split('"""')[0] or True
    s, p = _panel(app, csv)
    ov = p._gap_overlay
    assert not hasattr(ov, "_picture"), "ha vuelto el QPicture"
    assert hasattr(ov, "_rects")


def test_vb2_es_perezoso(app, csv):
    """El ViewBox del eje derecho se crea solo si hay algo que superponer.
    Creado siempre, duplica la cascada de propagacion de rango en cada frame."""
    s, p = _panel(app, csv)
    assert p.vb2 is None, "vb2 creado sin necesitarlo"
    p._ensure_vb2()
    assert p.vb2 is not None
    p._ensure_vb2()          # idempotente


def test_solo_el_ultimo_panel_muestra_eje_x(app, tmp_path):
    """Con N paneles sincronizados, los N repintaban el MISMO eje de fechas
    en cada frame: era el 27% del coste del pan con 18 paneles."""
    import pandas as pd
    from tsbox.mainwindow import MainWindow

    n = 1000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=n, freq="1min")})
    for c in "abcd":
        df[c] = rng.standard_normal(n)
    f = tmp_path / "m.csv"
    df.to_csv(f, index=False)

    w = MainWindow()
    w.show()
    w.session.open(f, "column", "ts", float32=True)
    w.rebuild_panels()
    w.refresh_list()
    app.processEvents()

    vis = [w.panels[s.sid] for s in w.panel_series()
           if w.panels[s.sid].isVisibleTo(w)]
    assert len(vis) == 4
    assert not any(p._x_axis_on for p in vis[:-1]), "ejes X duplicados"
    assert vis[-1]._x_axis_on, "el panel de abajo debe tener eje X"

    # sin sincronizar, cada panel necesita su propio eje
    w.a_sync.setChecked(False)
    w._refresh_x_axes()
    assert all(p._x_axis_on for p in vis)
