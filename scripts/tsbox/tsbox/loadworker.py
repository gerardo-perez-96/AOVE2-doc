"""Carga en un hilo aparte.

Sin esto, leer 400 MB congela la ventana ~10 s sin pintar nada: Qt no procesa
eventos mientras pandas trabaja. El usuario no distingue "tarda" de "colgado",
y es exactamente la misma queja.
"""
from __future__ import annotations

from PySide6 import QtCore


class LoadWorker(QtCore.QObject):
    sigProgress = QtCore.Signal(str, float)
    sigDone = QtCore.Signal(bool, str)      # ok, mensaje de error

    def __init__(self, session, kwargs: dict):
        super().__init__()
        self.session = session
        self.kwargs = kwargs
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _progress(self, msg: str, frac: float) -> None:
        if self._cancel:
            raise InterruptedError("Carga cancelada por el usuario.")
        self.sigProgress.emit(msg, frac)

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.session.open(progress=self._progress, **self.kwargs)
            self.sigDone.emit(True, "")
        except InterruptedError as e:
            self.sigDone.emit(False, str(e))
        except Exception as e:
            self.sigDone.emit(False, f"{type(e).__name__}: {e}")
