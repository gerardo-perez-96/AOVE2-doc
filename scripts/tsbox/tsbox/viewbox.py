"""ViewBox con máquina de estados de gestos.

El conflicto de arrastres se resuelve aquí, no con parches en cada panel:
  NAV     -> arrastre izq = pan          (pyqtgraph por defecto)
  REGION  -> arrastre izq = dibujar región
  MARK    -> click izq    = poner marca
La rueda hace zoom SIEMPRE, en los tres modos. Reordenar paneles no ocurre
nunca sobre el lienzo: se hace en la lista lateral.
"""
from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore
from PySide6.QtCore import Qt

MODE_NAV = "nav"
MODE_REGION = "region"
MODE_MARK = "mark"

CURSORS = {
    MODE_NAV: Qt.OpenHandCursor,
    MODE_REGION: Qt.SplitHCursor,
    MODE_MARK: Qt.CrossCursor,
}


class EditViewBox(pg.ViewBox):
    sigDrawRegion = QtCore.Signal(float, float, bool)  # x0, x1, terminado
    sigMark = QtCore.Signal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = MODE_NAV
        self.setCursor(CURSORS[MODE_NAV])

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.setCursor(CURSORS.get(mode, Qt.ArrowCursor))

    def mouseDragEvent(self, ev, axis=None):
        if (self.mode == MODE_REGION and axis is None
                and ev.button() == Qt.MouseButton.LeftButton):
            ev.accept()
            x0 = self.mapToView(ev.buttonDownPos()).x()
            x1 = self.mapToView(ev.pos()).x()
            self.sigDrawRegion.emit(min(x0, x1), max(x0, x1), ev.isFinish())
            return
        super().mouseDragEvent(ev, axis=axis)

    def mouseClickEvent(self, ev):
        if (self.mode == MODE_MARK
                and ev.button() == Qt.MouseButton.LeftButton):
            ev.accept()
            self.sigMark.emit(self.mapToView(ev.pos()).x())
            return
        super().mouseClickEvent(ev)
