"""Smoke test de la ventana de análisis en modo offscreen."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtWidgets  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

from tsbox.mainwindow import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


@pytest.fixture
def csv(tmp_path):
    n = 3000
    rng = np.random.default_rng(7)
    t = pd.date_range("2024-01-01", periods=n, freq="1s")
    base = rng.standard_normal(n).cumsum()
    lead = rng.standard_normal(n)
    df = pd.DataFrame({
        "ts": t,
        "a": base,
        "b": base * 0.8 + rng.standard_normal(n) * 0.5,
        "c": np.roll(lead, 9) + rng.standard_normal(n) * 0.2,
        "driver": lead,
        "bimodal": np.concatenate([rng.normal(-4, .5, n // 2),
                                   rng.normal(4, .5, n - n // 2)]),
    })
    df.loc[500:540, "a"] = np.nan          # NaN interiores
    df = pd.concat([df.iloc[:1500], df.iloc[1700:]])  # salto en el eje X
    p = tmp_path / "demo.csv"
    df.to_csv(p, index=False)
    return p


def _open(app, csv):
    w = MainWindow()
    w.session.open(csv, "column", "ts")
    w.rebuild_panels()
    w.refresh_list()
    w.open_analysis()
    return w


def test_ventana_analisis_abre_y_todas_las_pestanas(app, csv):
    w = _open(app, csv)
    an = w.analysis
    assert an is not None
    for i in range(an.tabs.count()):
        an.tabs.setCurrentIndex(i)
        an.refresh()
    assert an.m_table.rowCount() >= 4


def test_matriz_marca_significancia(app, csv):
    w = _open(app, csv)
    an = w.analysis
    an.tabs.setCurrentIndex(1)
    an.chk_window.setChecked(False)
    an.refresh_matrix()
    M = an._M
    i, j = M.names.index("a"), M.names.index("b")
    assert M.r[i, j] > 0.9
    assert M.n_eff[i, j] < M.n[i, j]           # corrección activa


def test_ccf_encuentra_el_lag_del_csv(app, csv):
    w = _open(app, csv)
    an = w.analysis
    an.tabs.setCurrentIndex(3)
    an.chk_window.setChecked(False)
    an.c_x.setCurrentIndex(an.c_x.findText("driver"))
    an.c_y.setCurrentIndex(an.c_y.findText("c"))
    an.refresh_ccf()
    assert an._ccf.best_lag == 9
    assert an._ccf.best_p < 1e-6


def test_scan_todas_contra_target(app, csv):
    w = _open(app, csv)
    an = w.analysis
    an.tabs.setCurrentIndex(3)
    an.chk_window.setChecked(False)
    an.c_y.setCurrentIndex(an.c_y.findText("c"))
    an._scan_all()
    assert an.c_table.rowCount() >= 4
    assert an.c_table.item(0, 0).text() == "driver"   # ordenado por |r|


def test_histograma_detecta_bimodal_en_ui(app, csv):
    w = _open(app, csv)
    an = w.analysis
    an.tabs.setCurrentIndex(0)
    an.chk_window.setChecked(False)
    for i in range(an.h_pick.count()):
        it = an.h_pick.item(i)
        it.setCheckState(Qt.Checked if it.text() == "bimodal" else Qt.Unchecked)
    an.refresh_hist()
    assert "multimodal" in an.h_info.toPlainText()


def test_serie_desplazada_se_materializa(app, csv):
    w = _open(app, csv)
    an = w.analysis
    an.tabs.setCurrentIndex(3)
    an.c_x.setCurrentIndex(an.c_x.findText("driver"))
    an.c_y.setCurrentIndex(an.c_y.findText("c"))
    an.refresh_ccf()
    before = len(w.session.project.series)
    an._materialize_lag()
    assert len(w.session.project.series) == before + 1
    new = w.session.project.series[-1]
    assert np.isfinite(w.session.values(new.sid)).sum() > 0


def test_informe_huecos_detecta_nan_y_salto(app, csv):
    w = _open(app, csv)
    sid = w.session.project.by_name("a").sid
    rep = w.session.missing(sid)
    assert rep["n_nan"] == 41
    assert len(rep["time_gaps"]) == 1
    assert rep["muestras_perdidas_estimadas"] == 200   # se tiraron las filas 1500:1700
