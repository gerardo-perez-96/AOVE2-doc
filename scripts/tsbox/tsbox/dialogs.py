from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from . import loader, longformat
from .model import (KIND_DERIVATIVE, KIND_ROLLING_MEAN, KIND_ROLLING_STD,
                    Project, SeriesDef)


class LoadDialog(QtWidgets.QDialog):
    """Qué columnas cargar, qué va en el eje X y cuántas muestras."""

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.setWindowTitle(f"Abrir {self.path.name}")
        self.resize(560, 560)
        self.preview = loader.peek_columns(self.path)

        lay = QtWidgets.QVBoxLayout(self)

        # --- eje X
        gx = QtWidgets.QGroupBox("Eje X")
        fx = QtWidgets.QGridLayout(gx)
        self.rb_col = QtWidgets.QRadioButton("Usar una columna")
        self.rb_idx = QtWidgets.QRadioButton("Índice de muestra (0…N−1)")
        self.cmb_x = QtWidgets.QComboBox()
        self.cmb_x.addItems([str(c) for c in self.preview.columns])
        guess = loader.guess_x_column(self.preview)
        if guess is not None:
            self.cmb_x.setCurrentText(str(guess))
            self.rb_col.setChecked(True)
        else:
            self.rb_idx.setChecked(True)
        self.cmb_x.setEnabled(self.rb_col.isChecked())
        self.rb_col.toggled.connect(self.cmb_x.setEnabled)
        self.rb_col.toggled.connect(self._refresh_table)
        self.cmb_x.currentIndexChanged.connect(lambda *_: self._refresh_table())
        fx.addWidget(self.rb_col, 0, 0)
        fx.addWidget(self.cmb_x, 0, 1)
        fx.addWidget(self.rb_idx, 1, 0, 1, 2)
        lay.addWidget(gx)

        # --- muestras
        gs = QtWidgets.QGroupBox("Límite de muestras")
        fs = QtWidgets.QHBoxLayout(gs)
        self.chk_limit = QtWidgets.QCheckBox("Limitar a")
        self.spin_max = QtWidgets.QSpinBox()
        self.spin_max.setRange(10, 100_000_000)
        self.spin_max.setValue(200_000)
        self.spin_max.setSingleStep(10_000)
        self.spin_max.setGroupSeparatorShown(True)
        self.spin_max.setEnabled(False)
        self.cmb_policy = QtWidgets.QComboBox()
        self.cmb_policy.addItem("primeras N muestras", "truncate")
        self.cmb_policy.addItem("decimar (introduce aliasing)", "decimate")
        self.cmb_policy.setEnabled(False)
        self.chk_limit.toggled.connect(self.spin_max.setEnabled)
        self.chk_limit.toggled.connect(self.cmb_policy.setEnabled)
        self.chk_limit.toggled.connect(lambda *_: self._refresh_cost())
        self.spin_max.valueChanged.connect(lambda *_: self._refresh_cost())
        self.cmb_policy.currentIndexChanged.connect(lambda *_: self._refresh_cost())
        fs.addWidget(self.chk_limit)
        fs.addWidget(self.spin_max)
        fs.addWidget(self.cmb_policy)
        fs.addStretch(1)
        lay.addWidget(gs)

        self.chk_f32 = QtWidgets.QCheckBox(
            "Guardar en float32 (mitad de RAM)")
        self.chk_f32.setChecked(True)
        self.chk_f32.setToolTip(
            "7 dígitos significativos. Sobra para dibujar y para estadísticas.\n"
            "Desactívalo si tus valores superan ~1e7 y los decimales importan.\n"
            "El eje X siempre va en float64 pase lo que pase.")
        self.chk_f32.toggled.connect(lambda *_: self._refresh_cost())
        lay.addWidget(self.chk_f32)

        self.cost = QtWidgets.QLabel("")
        self.cost.setWordWrap(True)
        self.cost.setStyleSheet("padding:4px; border-radius:3px;")
        lay.addWidget(self.cost)

        # --- formato largo (varias entidades apiladas en el mismo eje X)
        self.gb_long = QtWidgets.QGroupBox("Formato largo detectado")
        gl = QtWidgets.QVBoxLayout(self.gb_long)
        self.long_msg = QtWidgets.QLabel("")
        self.long_msg.setWordWrap(True)
        self.long_msg.setStyleSheet("color:#FFCC80;")
        gl.addWidget(self.long_msg)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Separar por"))
        self.cmb_group = QtWidgets.QComboBox()
        self.cmb_group.currentIndexChanged.connect(self._on_group_changed)
        row.addWidget(self.cmb_group, 1)
        gl.addLayout(row)
        self.rb_pivot = QtWidgets.QRadioButton(
            "Pivotar: una serie por (señal × entidad)  [recomendado]")
        self.rb_filter = QtWidgets.QRadioButton("Cargar solo una entidad:")
        self.rb_raw = QtWidgets.QRadioButton(
            "Cargar en crudo (entrelazado — las medias mezclarán entidades)")
        self.rb_pivot.setChecked(True)
        self.cmb_value = QtWidgets.QComboBox()
        self.cmb_value.setEnabled(False)
        self.rb_filter.toggled.connect(self.cmb_value.setEnabled)
        for r in (self.rb_pivot, self.rb_filter, self.rb_raw):
            r.toggled.connect(lambda *_: self._refresh_cost())
        gl.addWidget(self.rb_pivot)
        rowf = QtWidgets.QHBoxLayout()
        rowf.addWidget(self.rb_filter)
        rowf.addWidget(self.cmb_value, 1)
        gl.addLayout(rowf)
        gl.addWidget(self.rb_raw)
        self.gb_long.setVisible(False)
        lay.addWidget(self.gb_long)

        # --- columnas
        lay.addWidget(QtWidgets.QLabel("Series a cargar:"))
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Serie", "Tipo", "Ejemplo"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table, 1)

        row = QtWidgets.QHBoxLayout()
        b_all = QtWidgets.QPushButton("Todas")
        b_none = QtWidgets.QPushButton("Ninguna")
        b_all.clicked.connect(lambda: self._check_all(True))
        b_none.clicked.connect(lambda: self._check_all(False))
        row.addWidget(b_all)
        row.addWidget(b_none)
        row.addStretch(1)
        lay.addLayout(row)

        self.note = QtWidgets.QLabel("")
        self.note.setStyleSheet("color:#888")
        lay.addWidget(self.note)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.table.itemChanged.connect(lambda *_: self._refresh_cost())
        self._refresh_table()
        self._detect_long()
        self._refresh_cost()

    def _detect_long(self) -> None:
        """Se detecta sobre la vista previa (500 filas), que basta para ver la
        repetición del eje X. No hace falta leer el fichero entero."""
        xcol = self.x_column()
        self.long = longformat.detect(self.preview, xcol)
        self.gb_long.setVisible(self.long.is_long)
        if not self.long.is_long:
            return
        self.long_msg.setText(self.long.message)
        self.cmb_group.blockSignals(True)
        self.cmb_group.clear()
        self.cmb_group.addItems(self.long.group_columns)
        self.cmb_group.blockSignals(False)
        self._on_group_changed()

    def _on_group_changed(self) -> None:
        col = self.cmb_group.currentText()
        self.cmb_value.clear()
        if col and col in self.preview.columns:
            self.cmb_value.addItems(longformat.group_values(self.preview, col))
        self._refresh_cost()

    # --- resultados del bloque de formato largo
    def long_mode(self) -> str:
        if not getattr(self, "long", None) or not self.long.is_long:
            return "raw"
        if self.rb_pivot.isChecked():
            return "pivot"
        if self.rb_filter.isChecked():
            return "filter"
        return "raw"

    def group_column(self):
        m = self.long_mode()
        return self.cmb_group.currentText() if m in ("pivot", "filter") else None

    def group_value(self):
        return self.cmb_value.currentText() if self.long_mode() == "filter" else None

    def _refresh_cost(self) -> None:
        """Estima filas y RAM ANTES de leer. Sin esto abres un fichero de 4 GB
        sin enterarte hasta que el sistema empieza a hacer swap."""
        try:
            total = loader.estimate_rows(self.path)
        except Exception:
            total = None
        ncols = max(1, len(self.selected_columns()))
        nrows, step, _ = loader.plan_sampling(
            self.path, self.max_samples(), self.sample_policy())
        if total:
            eff = min(total, nrows) if nrows else (total // max(1, step))
        else:
            eff = self.max_samples() or 0
        width = 4 if self.chk_f32.isChecked() else 8
        mb = (eff * ncols * width + eff * 8) / 1e6

        mode = self.long_mode()
        if mode == "pivot" and getattr(self, "long", None) and self.long.n_groups:
            g = self.long.n_groups
            eff = max(1, eff // g)
            ncols = ncols * g
            mb = (eff * ncols * width + eff * 8) / 1e6
        elif mode == "filter" and getattr(self, "long", None) and self.long.n_groups:
            eff = max(1, eff // self.long.n_groups)
            mb = (eff * ncols * width + eff * 8) / 1e6

        txt = (f"≈{total:,} filas en el fichero. " if total else "")
        txt += f"Se cargarán ≈{eff:,} × {ncols} columnas ≈ {mb:,.0f} MB de RAM."
        if mode == "pivot":
            txt += "  (pivotado)"
        elif mode == "filter":
            txt += f"  (solo «{self.cmb_value.currentText()}»)"
        if step > 1:
            txt += f"  (1 de cada {step} filas)"
        if mb > 2000:
            txt += "\n\nEsto no te va a caber cómodamente. Limita muestras o columnas."
            self.cost.setStyleSheet("padding:4px; background:#5D1F1F; color:#FFCDD2;")
        elif mb > 600:
            txt += "\n\nVa a tardar. Considera limitar muestras para explorar primero."
            self.cost.setStyleSheet("padding:4px; background:#4E3B12; color:#FFE0B2;")
        else:
            self.cost.setStyleSheet("padding:4px; color:#9E9E9E;")
        self.cost.setText(txt)

    def float32(self) -> bool:
        return self.chk_f32.isChecked()

    def _check_all(self, state: bool) -> None:
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None:
                it.setCheckState(Qt.Checked if state else Qt.Unchecked)

    def _refresh_table(self) -> None:
        self.table.blockSignals(True)   # poblar no debe disparar itemChanged
        try:
            self._fill_table()
        finally:
            self.table.blockSignals(False)
        if hasattr(self, "gb_long"):
            self._detect_long()
        if hasattr(self, "cost"):
            self._refresh_cost()

    def _fill_table(self) -> None:
        xcol = self.x_column()
        num = loader.numeric_columns(self.preview)
        cols = [c for c in self.preview.columns if c != xcol]
        self.table.setRowCount(len(cols))
        for r, c in enumerate(cols):
            numeric = c in num
            it = QtWidgets.QTableWidgetItem(str(c))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if numeric else Qt.Unchecked)
            if not numeric:
                it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(
                str(self.preview[c].dtype) + ("" if numeric else "  (no numérica)")))
            sample = self.preview[c].dropna().head(1)
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(
                "" if sample.empty else str(sample.iloc[0])))
        self.table.resizeColumnsToContents()
        self.note.setText(
            f"Vista previa de {len(self.preview)} filas. "
            "Las columnas no numéricas no se pueden dibujar.")

    # --- resultados
    def x_mode(self) -> str:
        return "column" if self.rb_col.isChecked() else "index"

    def x_column(self):
        return self.cmb_x.currentText() if self.rb_col.isChecked() else None

    def max_samples(self):
        return self.spin_max.value() if self.chk_limit.isChecked() else None

    def sample_policy(self) -> str:
        return self.cmb_policy.currentData()

    def selected_columns(self) -> list[str]:
        out = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None and it.checkState() == Qt.Checked:
                out.append(it.text())
        return out


class DerivedDialog(QtWidgets.QDialog):
    """Filtro de media móvil, desviación móvil o derivada."""

    KINDS = [("Media móvil", KIND_ROLLING_MEAN),
             ("Desviación típica móvil", KIND_ROLLING_STD),
             ("Derivada d/dx", KIND_DERIVATIVE)]

    def __init__(self, project: Project, default_sid: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nueva serie derivada")
        self.project = project
        lay = QtWidgets.QFormLayout(self)

        self.cmb_parent = QtWidgets.QComboBox()
        for s in project.ordered():
            self.cmb_parent.addItem(s.name, s.sid)
        if default_sid:
            i = self.cmb_parent.findData(default_sid)
            if i >= 0:
                self.cmb_parent.setCurrentIndex(i)
        lay.addRow("Sobre la serie", self.cmb_parent)

        self.cmb_kind = QtWidgets.QComboBox()
        for label, kind in self.KINDS:
            self.cmb_kind.addItem(label, kind)
        lay.addRow("Operación", self.cmb_kind)

        self.spin_win = QtWidgets.QSpinBox()
        self.spin_win.setRange(2, 1_000_000)
        self.spin_win.setValue(25)
        lay.addRow("Ventana (muestras)", self.spin_win)

        self.chk_center = QtWidgets.QCheckBox("Ventana centrada (sin retardo)")
        self.chk_center.setChecked(True)
        lay.addRow("", self.chk_center)

        self.chk_overlay = QtWidgets.QCheckBox(
            "Superponer sobre la original (eje Y derecho)")
        self.chk_overlay.setChecked(True)
        lay.addRow("", self.chk_overlay)

        self.hint = QtWidgets.QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#888")
        lay.addRow(self.hint)
        self.cmb_kind.currentIndexChanged.connect(self._update_hint)
        self._update_hint()

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def _update_hint(self) -> None:
        kind = self.cmb_kind.currentData()
        needs_window = kind != KIND_DERIVATIVE
        self.spin_win.setEnabled(needs_window)
        self.chk_center.setEnabled(needs_window)
        if kind == KIND_DERIVATIVE:
            self.hint.setText(
                "La derivada amplifica el ruido. Si la señal es ruidosa, "
                "filtra primero con una media móvil y deriva sobre ese resultado.")
        else:
            self.hint.setText(
                "La ventana se mide en muestras, no en segundos. Con muestreo "
                "irregular su duración real varía a lo largo de la señal.")

    def result_params(self) -> tuple[str, str, dict, bool]:
        kind = self.cmb_kind.currentData()
        params: dict = {}
        if kind != KIND_DERIVATIVE:
            params = {"window": self.spin_win.value(),
                      "center": self.chk_center.isChecked()}
        return (self.cmb_parent.currentData(), kind, params,
                self.chk_overlay.isChecked())
