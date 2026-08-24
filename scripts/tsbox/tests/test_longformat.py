"""Formato largo y el bug O(n*m) que congelaba la ventana.

Estos dos tests son los que habrían pillado el "No responde": uno mide, el otro
comprueba la detección. Si vuelven a fallar, la regresión es exactamente esa.
"""
import os
import time

import numpy as np
import pandas as pd
import pytest

from tsbox import gaps, longformat
from tsbox.session import Session

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def stacked(tmp_path_factory):
    """6 entidades apiladas sobre el mismo eje de tiempo, con NaN dispersos."""
    d = tmp_path_factory.mktemp("long")
    n_t, ents = 20_000, [f"R{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n_t, freq="1min")
    rows = []
    for k, e in enumerate(ents):
        temp = 180 + 5 * k + rng.standard_normal(n_t) * 0.5
        temp[rng.random(n_t) < 0.06] = np.nan          # NaN dispersos
        rows.append(pd.DataFrame({"timestamp": ts, "regime": e[0],
                                  "reactor_id": e, "temp": temp,
                                  "pres": 15 + rng.standard_normal(n_t) * 0.2}))
    df = pd.concat(rows).sort_values(["timestamp", "reactor_id"])
    p = d / "stacked.csv"
    df.to_csv(p, index=False)
    return p, n_t, len(ents)


# --- el bug que congelaba: median_step dentro del bucle -----------------
def test_nan_intervals_no_es_cuadratico():
    """Con 25.000 tramos de NaN, la versión antigua tardaba ~70 s porque
    recalculaba la mediana del array entero en cada iteración."""
    n = 500_000
    x = np.arange(n, dtype=np.float64)
    y = np.ones(n)
    y[::20] = np.nan                       # 25.000 tramos
    t = time.perf_counter()
    iv = gaps.nan_intervals(x, y)
    dur = time.perf_counter() - t
    assert len(iv) == 25_000
    assert dur < 1.0, f"nan_intervals volvió a ser O(n·m): {dur:.1f} s"


def test_nan_intervals_equivalente_a_la_version_lenta():
    rng = np.random.default_rng(3)
    x = np.arange(2000, dtype=np.float64)
    y = rng.standard_normal(2000)
    y[rng.random(2000) < 0.15] = np.nan

    step = gaps.median_step(x)
    lento = []
    for i0, i1 in gaps._runs(~np.isfinite(y)):
        a = x[i0 - 1] if i0 > 0 else x[i0]
        b = x[i1 + 1] if i1 + 1 < len(x) else x[i1]
        if a == b:
            a, b = a - step / 2, b + step / 2
        lento.append((float(a), float(b)))
    assert gaps.nan_intervals(x, y) == lento


def test_missing_report_calcula_el_paso_una_vez(monkeypatch):
    calls = []
    real = gaps.median_step
    monkeypatch.setattr(gaps, "median_step",
                        lambda x: (calls.append(1), real(x))[1])
    x = np.arange(50_000, dtype=np.float64)
    y = np.ones(50_000)
    y[::10] = np.nan
    gaps.missing_report(x, y)
    assert len(calls) == 1, f"median_step llamado {len(calls)} veces"


def test_runs_vec_coincide_con_runs():
    rng = np.random.default_rng(5)
    for _ in range(20):
        m = rng.random(200) < 0.3
        assert [tuple(r) for r in gaps._runs_vec(m)] == gaps._runs(m)
    assert gaps._runs_vec(np.zeros(10, bool)).size == 0
    assert [tuple(r) for r in gaps._runs_vec(np.ones(5, bool))] == [(0, 4)]


# --- deteccion de formato largo ----------------------------------------
def test_detecta_entidades_apiladas(stacked):
    from tsbox import loader
    p, n_t, k = stacked
    df = loader.read_table(p, nrows=500)
    lf = longformat.detect(df, "timestamp")
    assert lf.is_long
    assert lf.group_columns[0] == "reactor_id"
    assert lf.n_groups == k
    assert lf.repeats == pytest.approx(k, abs=0.5)
    assert "formato LARGO" in lf.message


def test_no_marca_formato_ancho_normal(tmp_path):
    n = 5000
    df = pd.DataFrame({"t": np.arange(n, dtype=float),
                       "a": np.random.default_rng(0).standard_normal(n),
                       "b": np.random.default_rng(1).standard_normal(n)})
    assert not longformat.detect(df, "t").is_long


def test_pivot_produce_eje_x_sin_duplicados(stacked):
    p, n_t, k = stacked
    s = Session()
    s.open(p, "column", "timestamp", long_mode="pivot",
           group_column="reactor_id", float32=True)
    assert len(s.x) == n_t
    assert len(np.unique(s.x)) == len(s.x)
    assert len(s.project.series) == 2 * k          # temp y pres x 6 reactores
    assert s.project.by_name("temp·R0") is not None


def test_filter_carga_una_sola_entidad(stacked):
    p, n_t, k = stacked
    s = Session()
    s.open(p, "column", "timestamp", long_mode="filter",
           group_column="reactor_id", group_value="R3", float32=True)
    assert len(s.x) == n_t
    assert len(np.unique(s.x)) == len(s.x)
    assert (s.df.reactor_id == "R3").all()


def test_crudo_avisa_en_vez_de_callarse(stacked):
    p, n_t, k = stacked
    s = Session()
    s.open(p, "column", "timestamp", float32=True)
    assert len(s.x) == n_t * k
    assert any("formato LARGO" in w for w in s.warnings)


def test_las_medias_apiladas_mienten(stacked):
    """La media en crudo es la de 6 máquinas: un valor que ninguna tiene."""
    p, n_t, k = stacked
    crudo = Session(); crudo.open(p, "column", "timestamp", float32=True)
    piv = Session(); piv.open(p, "column", "timestamp", long_mode="pivot",
                              group_column="reactor_id", float32=True)
    m_crudo = crudo.stats(crudo.project.by_name("temp").sid)["mean"]
    medias = [piv.stats(piv.project.by_name(f"temp·R{i}").sid)["mean"]
              for i in range(k)]
    assert min(medias) < m_crudo < max(medias)
    assert not any(abs(m - m_crudo) < 1.0 for m in medias)


# --- rendimiento del panel ---------------------------------------------
def test_panel_con_muchos_huecos_es_rapido(stacked):
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    from tsbox.panel import SeriesPanel

    p, n_t, k = stacked
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = Session()
    s.open(p, "column", "timestamp", float32=True)
    sd = s.project.by_name("temp")
    rep = s.missing(sd.sid)
    assert len(rep["nan_intervals"]) > 5_000, "el fixture debe tener muchos huecos"

    t = time.perf_counter()
    panel = SeriesPanel(s, sd)
    panel.resize(1000, 200)
    panel.show()
    app.processEvents()
    dur = time.perf_counter() - t
    assert dur < 5.0, f"pintar un panel tardó {dur:.1f} s"
    # y no ha creado un QGraphicsItem por hueco
    assert len(panel._gap_items) == 0
    assert panel._gap_overlay is not None


def test_merge_intervals_reduce_y_conserva_el_span():
    from tsbox.panel import merge_intervals

    iv = [(float(i), float(i) + 0.4) for i in range(0, 50_000)]
    out, dropped = merge_intervals(iv, max_out=1000)
    assert len(out) <= 1000 and dropped > 0
    assert out[0][0] == pytest.approx(iv[0][0])
    assert out[-1][1] == pytest.approx(iv[-1][1])
    # pocos intervalos y sin tolerancia: se devuelven tal cual
    same, d = merge_intervals(iv[:10], max_out=1000)
    assert d == 0 and len(same) == 10
