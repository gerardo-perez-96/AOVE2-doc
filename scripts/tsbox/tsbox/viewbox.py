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
MODE_GLOBAL_REGION = "global_region"
MODE_GLOBAL_MARK = "global_mark"

CURSORS = {
    MODE_NAV: Qt.OpenHandCursor,
    MODE_REGION: Qt.SplitHCursor,
    MODE_MARK: Qt.CrossCursor,
    MODE_GLOBAL_REGION: Qt.SizeHorCursor,
    MODE_GLOBAL_MARK: Qt.CrossCursor,
}


class EditViewBox(pg.ViewBox):
    sigDrawRegion = QtCore.Signal(float, float, bool)  # x0, x1, terminado
    sigDrawGlobalRegion = QtCore.Signal(float, float, bool)  # idem, toda la vista
    sigMark = QtCore.Signal(float)
    sigGlobalMark = QtCore.Signal(float)                # click en modo Marca global
    sigPanStep = QtCore.Signal(int)                    # -1 atrás, +1 adelante
    sigNavClick = QtCore.Signal(float)                  # click simple en modo Navegar

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
        if (self.mode == MODE_GLOBAL_REGION and axis is None
                and ev.button() == Qt.MouseButton.LeftButton):
            ev.accept()
            x0 = self.mapToView(ev.buttonDownPos()).x()
            x1 = self.mapToView(ev.pos()).x()
            self.sigDrawGlobalRegion.emit(min(x0, x1), max(x0, x1), ev.isFinish())
            return
        super().mouseDragEvent(ev, axis=axis)

    def mouseClickEvent(self, ev):
        # Botones laterales del ratón (atrás/adelante, "back"/"forward"):
        # desplazan la vista SIEMPRE, en cualquier modo. No son un gesto de
        # edición -- son navegación, igual que la rueda, así que se atienden
        # antes de mirar self.mode.
        btn = ev.button()
        if btn == Qt.MouseButton.BackButton:
            ev.accept()
            self.sigPanStep.emit(-1)
            return
        if btn == Qt.MouseButton.ForwardButton:
            ev.accept()
            self.sigPanStep.emit(1)
            return
        if (self.mode == MODE_MARK and btn == Qt.MouseButton.LeftButton):
            ev.accept()
            self.sigMark.emit(self.mapToView(ev.pos()).x())
            return
        if (self.mode == MODE_GLOBAL_MARK and btn == Qt.MouseButton.LeftButton):
            ev.accept()
            self.sigGlobalMark.emit(self.mapToView(ev.pos()).x())
            return
        if self.mode == MODE_NAV and btn == Qt.MouseButton.LeftButton:
            # El ViewBox base no hace NADA con el click izquierdo en modo
            # navegar (solo gestiona el derecho, para el menú contextual),
            # así que interceptarlo aquí no pisa ningún comportamiento
            # existente. mouseClickEvent solo dispara sin arrastre de por
            # medio -- ya viene distinguido de pan/zoom por la propia
            # GraphicsScene de pyqtgraph.
            ev.accept()
            self.sigNavClick.emit(self.mapToView(ev.pos()).x())
            return
        super().mouseClickEvent(ev)
