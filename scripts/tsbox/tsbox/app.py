from __future__ import annotations

import sys
from pathlib import Path

import pyqtgraph as pg
from PySide6 import QtWidgets


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    pg.setConfigOptions(antialias=False, background="#1e1e1e", foreground="#d0d0d0")

    app = QtWidgets.QApplication(argv)
    app.setStyle("Fusion")

    from .mainwindow import MainWindow
    win = MainWindow()
    win.show()

    if len(argv) > 1 and Path(argv[1]).exists():
        from .dialogs import LoadDialog
        dlg = LoadDialog(Path(argv[1]), win)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            win.session.open(argv[1], dlg.x_mode(), dlg.x_column(),
                             dlg.max_samples(), dlg.sample_policy(),
                             dlg.selected_columns())
            win.rebuild_panels()
            win._set_enabled(True)
            win.setWindowTitle(f"tsbox — {Path(argv[1]).name}")

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
