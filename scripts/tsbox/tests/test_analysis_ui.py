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


@pytest.fixture
def csv_wave(tmp_path):
    """Serie con una frecuencia conocida, para probar el espectro y el
    filtro Butterworth sin depender del ruido de la fixture `csv`."""
    n = 4096
    fs = 50.0
    rng = np.random.default_rng(1)
    t = pd.date_range("2024-01-01", periods=n, freq=pd.Timedelta(seconds=1 / fs))
    sig = (np.sin(2 * np.pi * 5.0 * np.arange(n) / fs)
          + 0.05 * rng.standard_normal(n))
    df = pd.DataFrame({"ts": t, "wave": sig})
    p = tmp_path / "wave.csv"
    df.to_csv(p, index=False)
    return p


def _open(app, csv):
    w = MainWindow()
    w.session.open(csv, "column", "ts")
    w.rebuild_panels()
    w.refresh_list()
    w.open_analysis()
    return w


def _check_all(picker) -> None:
    """SeriesPicker ya no marca nada por defecto (a propósito, para que el
    histograma/matriz no salgan ilegibles nada más abrir). Los tests que
    necesitan "todo marcado" lo piden explícitamente con esto."""
    for i in range(picker.count()):
        picker.item(i).setCheckState(Qt.Checked)


def test_ventana_analisis_abre_y_todas_las_pestanas(app, csv):
    w = _open(app, csv)
    an = w.analysis
    assert an is not None
    _check_all(an.m_pick)
    for i in range(an.tabs.count()):
        an.tabs.setCurrentIndex(i)
        an.refresh()
    assert an.m_table.rowCount() >= 4


def test_nada_marcado_por_defecto_al_abrir(app, csv):
    """Lo que se pidió explícitamente: abrir el análisis sin ninguna
    variable pre-seleccionada, para decidir a mano qué mirar."""
    w = _open(app, csv)
    an = w.analysis
    assert an.h_pick.checked() == []
    assert an.m_pick.checked() == []


def test_matriz_marca_significancia(app, csv):
    w = _open(app, csv)
    an = w.analysis
    an.tabs.setCurrentIndex(1)
    an.section_combo.setCurrentIndex(0)  # "Todo el dataset"
    _check_all(an.m_pick)
    an.refresh_matrix()
    M = an._M
    i, j = M.names.index("a"), M.names.index("b")
    assert M.r[i, j] > 0.9
    assert M.n_eff[i, j] < M.n[i, j]           # corrección activa


def test_ccf_encuentra_el_lag_del_csv(app, csv):
    w = _open(app, csv)
    an = w.analysis
    an.tabs.setCurrentIndex(3)
    an.section_combo.setCurrentIndex(0)  # "Todo el dataset"
    an.c_x.setCurrentIndex(an.c_x.findText("driver"))
    an.c_y.setCurrentIndex(an.c_y.findText("c"))
    an.refresh_ccf()
    assert an._ccf.best_lag == 9
    assert an._ccf.best_p < 1e-6


def test_scan_todas_contra_target(app, csv):
    w = _open(app, csv)
    an = w.analysis
    an.tabs.setCurrentIndex(3)
    an.section_combo.setCurrentIndex(0)  # "Todo el dataset"
    an.c_y.setCurrentIndex(an.c_y.findText("c"))
    an._scan_all()
    assert an.c_table.rowCount() >= 4
    assert an.c_table.item(0, 0).text() == "driver"   # ordenado por |r|


def test_histograma_detecta_bimodal_en_ui(app, csv):
    w = _open(app, csv)
    an = w.analysis
    an.tabs.setCurrentIndex(0)
    an.section_combo.setCurrentIndex(0)  # "Todo el dataset"
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


def test_espectro_detecta_frecuencia_conocida(app, csv_wave):
    w = _open(app, csv_wave)
    an = w.analysis
    an.tabs.setCurrentIndex(5)   # Espectro de frecuencia
    an.section_combo.setCurrentIndex(0)  # "Todo el dataset"
    an.sp_sid.setCurrentIndex(an.sp_sid.findText("wave"))
    an.refresh_spectrum()
    res = an._spectrum
    assert res.peak_freqs.size >= 1
    dominant = res.peak_freqs[np.argmax(res.peak_power)]
    assert dominant == pytest.approx(5.0, abs=0.5)


def test_espectro_metodo_fft_tambien_encuentra_el_pico(app, csv_wave):
    w = _open(app, csv_wave)
    an = w.analysis
    an.tabs.setCurrentIndex(5)
    an.sp_sid.setCurrentIndex(an.sp_sid.findText("wave"))
    an.sp_method.setCurrentIndex(an.sp_method.findData("fft"))
    an.refresh_spectrum()
    res = an._spectrum
    assert res.method == "fft"
    dominant = res.peak_freqs[np.argmax(res.peak_power)]
    assert dominant == pytest.approx(5.0, abs=0.3)


def test_espectro_opciones_de_eje_no_revientan(app, csv_wave):
    """Periodo, log-X y log-Y (dB) son solo formas de dibujar el mismo
    resultado -- combinarlas no debería fallar ni cambiar qué pico se
    detecta."""
    w = _open(app, csv_wave)
    an = w.analysis
    an.tabs.setCurrentIndex(5)
    an.sp_sid.setCurrentIndex(an.sp_sid.findText("wave"))
    for period in (False, True):
        for logx in (False, True):
            for logy in (False, True):
                an.sp_period.setChecked(period)
                an.sp_logx.setChecked(logx)
                an.sp_logy.setChecked(logy)
                an.refresh_spectrum()
    dominant = an._spectrum.peak_freqs[np.argmax(an._spectrum.peak_power)]
    assert dominant == pytest.approx(5.0, abs=0.5)


def test_espectro_crea_filtro_butterworth_desde_el_pico(app, csv_wave, monkeypatch):
    w = _open(app, csv_wave)
    an = w.analysis
    an.tabs.setCurrentIndex(5)
    an.sp_sid.setCurrentIndex(an.sp_sid.findText("wave"))
    an.refresh_spectrum()
    before = len(w.session.project.series)

    from tsbox.dialogs import DerivedDialog
    # El diálogo pide confirmación con Ok/Cancel -- exec() bloquearía el
    # test esperando un click real, así que se simula "Ok" directamente.
    monkeypatch.setattr(DerivedDialog, "exec",
                        lambda self: QtWidgets.QDialog.Accepted)
    an._spectrum_to_filter()

    assert len(w.session.project.series) == before + 1
    new = w.session.project.series[-1]
    assert new.kind == "butterworth"
    assert new.params["cutoff"] == pytest.approx(5.0, abs=0.5)
    assert np.isfinite(w.session.values(new.sid)).sum() > 0


def test_espectro_sin_calcular_no_crea_filtro(app, csv_wave, monkeypatch):
    """Sin haber calculado el espectro todavía, el botón avisa en vez de
    abrir el diálogo con datos inventados. OJO: no se cambia de pestaña
    aquí -- tabs.setCurrentIndex dispara un refresh() completo que
    calcularía el espectro solo, y la ventana abre en la pestaña de
    histograma (índice 0), así que _spectrum sigue sin existir."""
    w = _open(app, csv_wave)
    an = w.analysis
    assert an.tabs.currentIndex() == 0
    assert getattr(an, "_spectrum", None) is None
    before = len(w.session.project.series)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    an._spectrum_to_filter()
    assert len(w.session.project.series) == before
