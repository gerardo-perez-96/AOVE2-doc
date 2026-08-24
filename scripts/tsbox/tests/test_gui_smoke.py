"""Smoke test headless: no valida píxeles, valida que el cableado no explota."""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets  # noqa: E402

from tsbox.mainwindow import MainWindow  # noqa: E402
from tsbox.model import KIND_DERIVATIVE, KIND_ROLLING_MEAN  # noqa: E402
from tsbox.viewbox import MODE_MARK, MODE_REGION  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def csv(tmp_path):
    n = 2000
    t = pd.date_range("2024-05-01", periods=n, freq="100ms").delete(range(800, 850))
    y = np.sin(np.arange(len(t)) / 30) + np.random.default_rng(0).normal(0, .05, len(t))
    y[300:320] = np.nan
    p = tmp_path / "senal.csv"
    pd.DataFrame({"ts": t, "a": y, "b": np.cos(np.arange(len(t)) / 50)}).to_csv(
        p, index=False)
    return p


def test_flujo_completo(qapp, csv):
    w = MainWindow()
    w.session.open(csv, "column", "ts")
    w.rebuild_panels()
    assert len(w.panels) == 2
    assert w.list.topLevelItemCount() == 2  # árbol: 2 series raíz, sin derivadas

    sid_a = w.session.project.by_name("a").sid
    sid_b = w.session.project.by_name("b").sid

    # los huecos se detectan en el panel
    assert "⚠" in w.panels[sid_a].missing_lbl.text()

    # anotar: región + marca, con deshacer
    w.set_mode(MODE_REGION)
    w.on_region_drawn(sid_a, 200.0, 100.0)          # invertida a propósito
    r = w.session.project.regions[0]
    assert r.t0 == 100.0 and r.t1 == 200.0
    w.set_mode(MODE_MARK)
    w.on_mark_drawn(sid_b, 150.0)
    assert w.table.rowCount() == 2
    w.undo.undo()
    assert len(w.session.project.marks) == 0
    w.undo.redo()
    assert len(w.session.project.marks) == 1

    # derivadas: ambas tienen SIEMPRE panel propio (plegado bajo el original
    # en el árbol). overlay=True además la dibuja superpuesta en el panel
    # del padre -- son dos cosas independientes, no alternativas.
    mm = w.session.add_derived(sid_a, KIND_ROLLING_MEAN, {"window": 21}, overlay=True)
    dv = w.session.add_derived(sid_a, KIND_DERIVATIVE, {}, overlay=False)
    w.rebuild_panels()
    w.refresh_list()
    assert len(w.panels) == 4                       # a, b, mm, dv
    assert mm.sid in w.panels                        # panel propio, plegado
    assert mm.sid in w.panels[sid_a]._overlay_curves  # y también superpuesta
    assert dv.sid in w.panels
    assert dv.sid not in w.panels[sid_a]._overlay_curves
    root = w._tree_items[sid_a]
    assert root.childCount() == 2

    # reordenar: mover la última a la primera posición
    order = [s.sid for s in w.session.project.ordered()]
    w.session.project.reorder([order[-1]] + order[:-1])
    assert w.session.project.ordered()[0].sid == order[-1]

    # zoom sincronizado
    w.set_x_range(120.0, 180.0)
    for p in w.panels.values():
        x0, x1 = p.vb.viewRange()[0]
        assert abs(x0 - 120) < 1 and abs(x1 - 180) < 1

    # estadísticas de la ventana visible, no de toda la serie
    w.panels[sid_a].update_stats()
    assert "μ" in w.panels[sid_a].stats_lbl.text()
    w.a_stats.setChecked(True)
    assert w.panels[sid_a]._stat_items

    # guardado
    w.session.dirty = True
    w.save(manual=True)
    assert w.session.json_path.exists()
    assert w.session.json_path.name == "senal.json"

    # visibilidad
    w.hide_series(sid_b)
    assert not w.panels[sid_b].isVisible()

    # borrar anotación desde la tabla
    w.table.selectRow(0)
    w.delete_selected_annotation()
    assert len(w.session.project.regions) + len(w.session.project.marks) == 1

    # cerrar con cambios pendientes abre un modal: en headless eso cuelga,
    # así que confirmamos que la app se considera sucia y guardamos antes.
    assert w.session.dirty
    w.save(manual=True)
    w.close()


def test_reapertura_conserva_todo(qapp, csv):
    w = MainWindow()
    w.session.open(csv, "column", "ts")
    sid = w.session.project.by_name("a").sid
    w.rebuild_panels()
    w.on_region_drawn(sid, 10.0, 20.0)
    w.session.add_derived(sid, KIND_ROLLING_MEAN, {"window": 9}, overlay=True)
    w.save(manual=True)
    w.close()

    w2 = MainWindow()
    w2.session.open(csv, "column", "ts")
    w2.rebuild_panels()
    assert len(w2.session.project.regions) == 1
    assert any(s.params.get("window") == 9 for s in w2.session.project.series)
    assert not w2.session.warnings
    w2.close()
