"""Árbol de series: cada derivada tiene panel propio, plegado bajo la
original por defecto. Antes la única forma de sacar una derivada de su
panel era superponerla en el mismo gráfico con eje Y secundario -- que es
justo lo que costaba leer con varias señales.
"""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6 import QtWidgets  # noqa: E402

from tsbox.mainwindow import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(app, tmp_path):
    n = 2000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=n, freq="1min"),
                       "a": rng.standard_normal(n).cumsum(),
                       "b": rng.standard_normal(n)})
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    w = MainWindow()
    w.show()                      # imprescindible: isVisible() depende de esto
    w.session.open(p, "column", "ts", float32=True)
    w.rebuild_panels()
    w.refresh_list()
    app.processEvents()
    yield w
    # Nota: NO se llama a w.close() aquí. En este entorno offscreen, cerrar
    # explícitamente una QMainWindow con varios PlotWidget/pyqtgraph anidados
    # cuelga o segfaulta en el teardown de Qt (ya visto en otras sesiones de
    # test de este proyecto). pytest + QApplication compartida limpian solos
    # al terminar el proceso; forzar el cierre aquí no aporta nada y sí rompe.


def test_derivada_tiene_panel_propio(win):
    sid = win.session.project.by_name("a").sid
    d = win.session.add_derived(sid, "rolling_mean", {"window": 10})
    win.rebuild_panels()
    win.refresh_list()
    assert d.sid in win.panels
    assert win.panels[d.sid] is not win.panels[sid]


def test_derivada_plegada_por_defecto(win):
    sid = win.session.project.by_name("a").sid
    d = win.session.add_derived(sid, "derivative", {})
    win.rebuild_panels()
    win.refresh_list()
    root = win._tree_items[sid]
    assert not root.isExpanded()
    assert not win.panels[d.sid].isVisible()
    assert win.panels[sid].isVisible()


def test_expandir_muestra_la_derivada_y_plegar_la_oculta(win):
    sid = win.session.project.by_name("a").sid
    d = win.session.add_derived(sid, "rolling_mean", {"window": 5})
    win.rebuild_panels()
    win.refresh_list()
    root = win._tree_items[sid]

    root.setExpanded(True)
    win.refresh_visibility()
    assert win.panels[d.sid].isVisible()

    root.setExpanded(False)
    win.refresh_visibility()
    assert not win.panels[d.sid].isVisible()
    # el checkbox no se ha tocado: al re-expandir, vuelve a aparecer
    assert win.session.project.by_id(d.sid).visible is True
    root.setExpanded(True)
    win.refresh_visibility()
    assert win.panels[d.sid].isVisible()


def test_checkbox_manda_incluso_expandido(win):
    """Desmarcar una derivada la oculta aunque su padre esté expandido."""
    sid = win.session.project.by_name("a").sid
    d = win.session.add_derived(sid, "rolling_std", {"window": 5})
    win.rebuild_panels()
    win.refresh_list()
    root = win._tree_items[sid]
    root.setExpanded(True)
    d.visible = False
    win.refresh_visibility()
    assert not win.panels[d.sid].isVisible()


def test_varias_derivadas_cuelgan_del_mismo_padre(win):
    sid = win.session.project.by_name("a").sid
    d1 = win.session.add_derived(sid, "rolling_mean", {"window": 5})
    d2 = win.session.add_derived(sid, "derivative", {})
    win.rebuild_panels()
    win.refresh_list()
    root = win._tree_items[sid]
    assert root.childCount() == 2
    hijos = {root.child(i).data(0, Qt_UserRole_workaround(win)) for i in range(2)}
    assert hijos == {d1.sid, d2.sid}


def Qt_UserRole_workaround(win):
    from PySide6.QtCore import Qt
    return Qt.UserRole


def test_eliminar_original_elimina_sus_derivadas_y_paneles(win):
    sid = win.session.project.by_name("a").sid
    d = win.session.add_derived(sid, "rolling_mean", {"window": 5})
    win.rebuild_panels()
    win.refresh_list()
    win.session.project.remove_series(sid)
    win.rebuild_panels()
    win.refresh_list()
    assert sid not in win.panels
    assert d.sid not in win.panels
    assert win.session.project.by_id(d.sid) is None


def test_series_sin_hijos_es_hoja_sin_flecha(win):
    b = win.session.project.by_name("b")
    root = win._tree_items[b.sid]
    assert root.childCount() == 0


def test_reordenar_arriba_conserva_los_hijos(win):
    sid = win.session.project.by_name("a").sid
    d1 = win.session.add_derived(sid, "rolling_mean", {"window": 5})
    d2 = win.session.add_derived(sid, "derivative", {})
    win.rebuild_panels()
    win.refresh_list()
    others = [x.sid for x in win.session.project.ordered()
             if x.sid not in (sid, d1.sid, d2.sid)]
    win.session.project.reorder([sid, d1.sid, d2.sid] + others)
    win.rebuild_panels()
    win.refresh_list()
    root = win.list.topLevelItem(0)
    assert root.data(0, Qt_UserRole_workaround(win)) == sid
    assert root.childCount() == 2
