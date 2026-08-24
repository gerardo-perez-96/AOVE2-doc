"""Ventana de análisis: histogramas, matriz de correlación, ACF/PACF y CCF.

Todo se calcula sobre la VENTANA VISIBLE si marcas la casilla. Es lo que
casi siempre quieres: la correlación global de una señal con tendencia te
dice poco, la del tramo que estás mirando te dice mucho.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import analysis as A

SIG_BG = QtGui.QColor("#1B5E20")
SIG_RAW_BG = QtGui.QColor("#4E342E")
NEG = QtGui.QColor("#EF5350")
POS = QtGui.QColor("#42A5F5")


def heat(r: float) -> QtGui.QColor:
    """Azul = positiva, rojo = negativa, intensidad = |r|."""
    if not np.isfinite(r):
        return QtGui.QColor("#2A2A2A")
    base = POS if r >= 0 else NEG
    c = QtGui.QColor(base)
    c.setAlpha(int(np.clip(abs(r), 0, 1) * 190))
    return c


class SeriesPicker(QtWidgets.QListWidget):
    """Multiselección de series con checkboxes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

    def populate(self, session, checked: set[str] | None = None) -> None:
        prev = checked if checked is not None else set(self.checked())
        self.clear()
        for s in session.project.ordered():
            it = QtWidgets.QListWidgetItem(s.name)
            it.setData(Qt.UserRole, s.sid)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if (not prev or s.sid in prev)
                             else Qt.Unchecked)
            it.setForeground(QtGui.QColor(s.color or "#DDD"))
            self.addItem(it)

    def checked(self) -> list[str]:
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())
                if self.item(i).checkState() == Qt.Checked]


class AnalysisWindow(QtWidgets.QDialog):
    def __init__(self, session, main, parent=None):
        super().__init__(parent)
        self.session = session
        self.main = main
        self.setWindowTitle("Análisis")
        self.resize(1180, 820)
        self.setWindowFlags(self.windowFlags() | Qt.Window)

        root = QtWidgets.QVBoxLayout(self)

        # --- barra común
        bar = QtWidgets.QHBoxLayout()
        self.chk_window = QtWidgets.QCheckBox("Solo la ventana visible")
        self.chk_window.setChecked(True)
        self.chk_window.setToolTip(
            "Calcula sobre el rango X que estás viendo, no sobre todo el fichero.")
        b_ref = QtWidgets.QPushButton("Recalcular  (F5)")
        b_ref.clicked.connect(self.refresh)
        bar.addWidget(self.chk_window)
        bar.addStretch(1)
        bar.addWidget(b_ref)
        root.addLayout(bar)

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)
        self.tabs.addTab(self._tab_hist(), "Histograma / modos")
        self.tabs.addTab(self._tab_matrix(), "Matriz de correlación")
        self.tabs.addTab(self._tab_acf(), "ACF / PACF")
        self.tabs.addTab(self._tab_ccf(), "Correlación con desfase")

        self.note = QtWidgets.QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color:#FFB74D; padding:4px;")
        root.addWidget(self.note)

        QtGui.QShortcut(QtGui.QKeySequence("F5"), self, activated=self.refresh)
        self.tabs.currentChanged.connect(lambda _: self.refresh())

    # ------------------------------------------------------------------
    def _xrange(self):
        if not self.chk_window.isChecked():
            return None, None
        for p in self.main.panels.values():
            if p.isVisible():
                return tuple(p.plot.viewRange()[0])
        return None, None

    def data(self, sid: str) -> np.ndarray:
        x0, x1 = self._xrange()
        return A.slice_window(self.session.x, self.session.values(sid), x0, x1)

    def name(self, sid: str) -> str:
        s = self.session.project.by_id(sid)
        return s.name if s else sid

    def color(self, sid: str) -> str:
        s = self.session.project.by_id(sid)
        return (s.color if s else None) or "#4C9AFF"

    def _sid_combo(self, combo: QtWidgets.QComboBox, keep=True) -> None:
        cur = combo.currentData() if keep else None
        combo.blockSignals(True)
        combo.clear()
        for s in self.session.project.ordered():
            combo.addItem(s.name, s.sid)
        if cur is not None:
            i = combo.findData(cur)
            if i >= 0:
                combo.setCurrentIndex(i)
        combo.blockSignals(False)

    # ================================================== histograma
    def _tab_hist(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Series"))
        self.h_pick = SeriesPicker()
        self.h_pick.itemChanged.connect(lambda _: self.refresh_hist())
        side.addWidget(self.h_pick, 1)

        f = QtWidgets.QFormLayout()
        self.h_bins = QtWidgets.QSpinBox()
        self.h_bins.setRange(0, 2000)
        self.h_bins.setSpecialValueText("auto (F-D)")
        self.h_bins.setValue(0)
        self.h_bins.valueChanged.connect(lambda _: self.refresh_hist())
        f.addRow("Bins", self.h_bins)

        self.h_prom = QtWidgets.QDoubleSpinBox()
        self.h_prom.setRange(0.01, 0.9)
        self.h_prom.setSingleStep(0.02)
        self.h_prom.setValue(0.08)
        self.h_prom.setToolTip("Prominencia mínima de un modo, en fracción del "
                               "pico máximo. Súbelo si ves modos fantasma.")
        self.h_prom.valueChanged.connect(lambda _: self.refresh_hist())
        f.addRow("Prominencia modo", self.h_prom)

        self.h_norm = QtWidgets.QCheckBox("Densidad + KDE")
        self.h_norm.setChecked(True)
        self.h_norm.toggled.connect(lambda _: self.refresh_hist())
        f.addRow(self.h_norm)
        side.addLayout(f)

        self.h_info = QtWidgets.QPlainTextEdit()
        self.h_info.setReadOnly(True)
        self.h_info.setMaximumHeight(190)
        self.h_info.setStyleSheet("font-family:monospace; font-size:11px;")
        side.addWidget(self.h_info)

        cont = QtWidgets.QWidget()
        cont.setLayout(side)
        cont.setMaximumWidth(300)
        lay.addWidget(cont)

        self.h_layout = pg.GraphicsLayoutWidget()
        lay.addWidget(self.h_layout, 1)
        return w

    def refresh_hist(self) -> None:
        self.h_layout.clear()
        self.h_info.clear()
        sids = self.h_pick.checked()[:8]
        lines = []
        for i, sid in enumerate(sids):
            y = self.data(sid)
            bins = self.h_bins.value() or "auto"
            h = A.histogram(y, bins, kde=self.h_norm.isChecked(),
                            prominence=self.h_prom.value())
            pl = self.h_layout.addPlot(row=i, col=0, title=self.name(sid))
            pl.setLabel("bottom", "valor")
            pl.showGrid(x=True, y=True, alpha=0.2)
            if h.n == 0:
                lines.append(f"{self.name(sid)}: sin datos válidos")
                continue

            widths = np.diff(h.edges)
            dens = h.counts / (h.counts.sum() * widths) if self.h_norm.isChecked() \
                else h.counts.astype(float)
            c = QtGui.QColor(self.color(sid))
            c.setAlpha(150)
            pl.addItem(pg.BarGraphItem(x=h.centers, height=dens,
                                       width=widths * 0.95, brush=c, pen=None))
            if h.kde_y.size:
                pl.plot(h.kde_x, h.kde_y, pen=pg.mkPen("#FFFFFF", width=2))
            for m in h.modes:
                pl.addItem(pg.InfiniteLine(m, angle=90, pen=pg.mkPen(
                    "#FFD54F", width=2, style=Qt.DashLine)))

            v = y[np.isfinite(y)]
            desc = (f"{self.name(sid)}\n"
                    f"  n={h.n}  faltan={h.n_nan}  bins={len(h.counts)} ({h.bin_rule})\n"
                    f"  μ={v.mean():.4g}  σ={v.std(ddof=1):.4g}  "
                    f"mediana={np.median(v):.4g}\n"
                    f"  asimetría={float(_skew(v)):.3g}  curtosis={float(_kurt(v)):.3g}")
            if len(h.modes):
                mm = "  ".join(f"{m:.4g}({w:.2f})" for m, w in
                               zip(h.modes, h.mode_weights))
                desc += f"\n  modos: {mm}"
                if len(h.modes) > 1:
                    desc += "  <- multimodal: puede haber regímenes distintos"
            lines.append(desc)
        self.h_info.setPlainText("\n\n".join(lines))

    # ================================================== matriz
    def _tab_matrix(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        top = QtWidgets.QHBoxLayout()
        self.m_method = QtWidgets.QComboBox()
        self.m_method.addItems(["pearson", "spearman", "kendall"])
        self.m_method.currentIndexChanged.connect(lambda _: self.refresh_matrix())
        self.m_alpha = QtWidgets.QDoubleSpinBox()
        self.m_alpha.setRange(0.001, 0.2)
        self.m_alpha.setDecimals(3)
        self.m_alpha.setSingleStep(0.01)
        self.m_alpha.setValue(0.05)
        self.m_alpha.valueChanged.connect(lambda _: self.refresh_matrix())
        self.m_adj = QtWidgets.QCheckBox("Corregir por autocorrelación (n efectivo)")
        self.m_adj.setChecked(True)
        self.m_adj.setToolTip(
            "Sin esto, dos series con inercia salen siempre significativas.")
        self.m_adj.toggled.connect(lambda _: self.refresh_matrix())
        self.m_show = QtWidgets.QComboBox()
        self.m_show.addItems(["r", "p", "n efectivo"])
        self.m_show.currentIndexChanged.connect(lambda _: self.refresh_matrix())

        for lbl, wid in (("Método", self.m_method), ("α", self.m_alpha),
                         ("Mostrar", self.m_show)):
            top.addWidget(QtWidgets.QLabel(lbl))
            top.addWidget(wid)
        top.addWidget(self.m_adj)
        top.addStretch(1)
        lay.addLayout(top)

        split = QtWidgets.QSplitter(Qt.Horizontal)
        self.m_pick = SeriesPicker()
        self.m_pick.itemChanged.connect(lambda _: self.refresh_matrix())
        self.m_pick.setMaximumWidth(240)
        split.addWidget(self.m_pick)

        self.m_table = QtWidgets.QTableWidget()
        self.m_table.setAlternatingRowColors(False)
        split.addWidget(self.m_table)
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)

        leg = QtWidgets.QLabel(
            "Fondo verde = significativo tras corrección FDR (fiable).   "
            "Marrón = solo p<α en crudo (sospechoso con muchos tests).   "
            "Doble clic en una celda para abrirla en la pestaña de desfase.")
        leg.setStyleSheet("color:#9E9E9E; font-size:11px;")
        lay.addWidget(leg)
        self.m_table.cellDoubleClicked.connect(self._matrix_to_ccf)
        return w

    def refresh_matrix(self) -> None:
        sids = self.m_pick.checked()[:40]
        self._m_sids = sids
        t = self.m_table
        t.clear()
        if len(sids) < 2:
            t.setRowCount(0)
            t.setColumnCount(0)
            return
        data = {self.name(s): self.data(s) for s in sids}
        M = A.corr_matrix(data, self.m_method.currentText(),
                          self.m_alpha.value(), self.m_adj.isChecked())
        self._M = M
        k = len(M.names)
        t.setRowCount(k)
        t.setColumnCount(k)
        t.setHorizontalHeaderLabels(M.names)
        t.setVerticalHeaderLabels(M.names)
        mode = self.m_show.currentIndex()

        for i in range(k):
            for j in range(k):
                if mode == 0:
                    val, txt = M.r[i, j], _fmt(M.r[i, j], 3)
                elif mode == 1:
                    val, txt = M.r[i, j], ("—" if i == j else _fmt_p(M.p[i, j]))
                else:
                    val, txt = M.r[i, j], ("—" if i == j else _fmt(M.n_eff[i, j], 0))
                it = QtWidgets.QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignCenter)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if i == j:
                    it.setBackground(QtGui.QColor("#37474F"))
                elif M.sig_fdr[i, j]:
                    it.setBackground(SIG_BG)
                    it.setFont(_bold())
                elif M.sig_raw[i, j]:
                    it.setBackground(SIG_RAW_BG)
                else:
                    it.setBackground(heat(val))
                if i != j:
                    it.setToolTip(
                        f"{M.names[i]} vs {M.names[j]}\n"
                        f"r = {_fmt(M.r[i,j],4)}\np = {_fmt_p(M.p[i,j])}\n"
                        f"n = {M.n[i,j]}   n efectivo = {_fmt(M.n_eff[i,j],0)}\n"
                        f"{'SIGNIFICATIVO (FDR)' if M.sig_fdr[i,j] else ''}")
                t.setItem(i, j, it)
        t.resizeColumnsToContents()
        self.note.setText("  ".join(M.notes))

    def _matrix_to_ccf(self, i: int, j: int) -> None:
        if i == j or not getattr(self, "_m_sids", None):
            return
        self.c_x.setCurrentIndex(self.c_x.findData(self._m_sids[i]))
        self.c_y.setCurrentIndex(self.c_y.findData(self._m_sids[j]))
        self.tabs.setCurrentIndex(3)

    # ================================================== ACF / PACF
    def _tab_acf(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        self.a_sid = QtWidgets.QComboBox()
        self.a_sid.currentIndexChanged.connect(lambda _: self.refresh_acf())
        self.a_lags = QtWidgets.QSpinBox()
        self.a_lags.setRange(5, 5000)
        self.a_lags.setValue(50)
        self.a_lags.valueChanged.connect(lambda _: self.refresh_acf())
        self.a_bart = QtWidgets.QCheckBox("Banda de Bartlett")
        self.a_bart.setChecked(True)
        self.a_bart.toggled.connect(lambda _: self.refresh_acf())
        self.a_diff = QtWidgets.QCheckBox("Diferenciar (1−B)")
        self.a_diff.setToolTip("Quita la tendencia. Si la ACF decae lentísimo, "
                               "la serie no es estacionaria y esto lo arregla.")
        self.a_diff.toggled.connect(lambda _: self.refresh_acf())
        for lbl, wid in (("Serie", self.a_sid), ("Lags", self.a_lags)):
            top.addWidget(QtWidgets.QLabel(lbl))
            top.addWidget(wid)
        top.addWidget(self.a_bart)
        top.addWidget(self.a_diff)
        top.addStretch(1)
        lay.addLayout(top)

        self.a_layout = pg.GraphicsLayoutWidget()
        lay.addWidget(self.a_layout, 1)
        self.a_info = QtWidgets.QLabel("")
        self.a_info.setWordWrap(True)
        self.a_info.setStyleSheet("font-family:monospace; font-size:11px;")
        lay.addWidget(self.a_info)
        return w

    def refresh_acf(self) -> None:
        self.a_layout.clear()
        sid = self.a_sid.currentData()
        if sid is None:
            return
        y = A.fill_gaps(self.data(sid))
        if self.a_diff.isChecked() and y.size > 1:
            y = np.diff(y)
        if y.size < 10:
            self.a_info.setText("Pocas muestras.")
            return
        nl = min(self.a_lags.value(), y.size - 1)
        r = A.acf(y, nl)
        p = A.pacf(y, nl)
        lags = np.arange(r.size)
        cr = A.acf_conf(r, y.size, bartlett=self.a_bart.isChecked())
        cp = A.pacf_conf(p.size, y.size)

        for row, (vals, conf, title) in enumerate(
                [(r, cr, "ACF — autocorrelación"),
                 (p, cp, "PACF — autocorrelación parcial")]):
            pl = self.a_layout.addPlot(row=row, col=0, title=title)
            pl.showGrid(x=True, y=True, alpha=0.2)
            pl.setLabel("bottom", "lag (muestras)")
            pl.setYRange(-1.05, 1.05)
            band = pg.FillBetweenItem(
                pg.PlotDataItem(lags, conf), pg.PlotDataItem(lags, -conf),
                brush=pg.mkBrush(80, 140, 255, 55))
            pl.addItem(band)
            sig = np.abs(vals) > conf
            pl.addItem(pg.BarGraphItem(x=lags[~sig], height=vals[~sig], width=0.6,
                                       brush="#607D8B"))
            pl.addItem(pg.BarGraphItem(x=lags[sig], height=vals[sig], width=0.6,
                                       brush="#66BB6A"))
            pl.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#888")))

        q, pv = A.ljung_box(y, min(20, y.size // 4))
        first = np.flatnonzero(np.abs(r[1:]) <= cr[1:])
        txt = (f"n={y.size}   Ljung-Box Q={q:.1f}  p={_fmt_p(pv)}   ")
        txt += ("la serie NO es ruido blanco: tiene memoria."
                if np.isfinite(pv) and pv < 0.05
                else "compatible con ruido blanco.")
        if first.size:
            txt += f"   La ACF entra en banda en el lag {int(first[0])+1}."
        sigp = np.flatnonzero(np.abs(p[1:]) > cp[1:])[:6]
        if sigp.size:
            txt += ("   PACF significativa en lags "
                    + ", ".join(str(int(i) + 1) for i in sigp)
                    + " -> orden AR sugerido ≈ " + str(int(sigp[-1]) + 1))
        self.a_info.setText(txt)

    # ================================================== CCF
    def _tab_ccf(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        top = QtWidgets.QHBoxLayout()
        self.c_x = QtWidgets.QComboBox()
        self.c_y = QtWidgets.QComboBox()
        for c in (self.c_x, self.c_y):
            c.currentIndexChanged.connect(lambda _: self.refresh_ccf())
        self.c_maxlag = QtWidgets.QSpinBox()
        self.c_maxlag.setRange(1, 100000)
        self.c_maxlag.setValue(50)
        self.c_maxlag.valueChanged.connect(lambda _: self.refresh_ccf())
        self.c_pw = QtWidgets.QCheckBox("Pre-blanquear")
        self.c_pw.setChecked(True)
        self.c_pw.setToolTip("Quita la inercia de X antes de comparar. Sin esto, "
                             "dos señales lentas correlacionan aunque no tengan "
                             "nada que ver.")
        self.c_pw.toggled.connect(lambda _: self.refresh_ccf())
        top.addWidget(QtWidgets.QLabel("X (predictor)"))
        top.addWidget(self.c_x, 1)
        top.addWidget(QtWidgets.QLabel("Y (target)"))
        top.addWidget(self.c_y, 1)
        top.addWidget(QtWidgets.QLabel("máx lag"))
        top.addWidget(self.c_maxlag)
        top.addWidget(self.c_pw)
        lay.addLayout(top)

        act = QtWidgets.QHBoxLayout()
        self.c_lag = QtWidgets.QSpinBox()
        self.c_lag.setRange(-100000, 100000)
        self.c_lag.valueChanged.connect(self._draw_overlay)
        b_best = QtWidgets.QPushButton("Ir al mejor lag")
        b_best.clicked.connect(self._goto_best)
        b_make = QtWidgets.QPushButton("Crear serie X desplazada")
        b_make.clicked.connect(self._materialize_lag)
        b_all = QtWidgets.QPushButton("Todas contra este target…")
        b_all.clicked.connect(self._scan_all)
        act.addWidget(QtWidgets.QLabel("Lag manual"))
        act.addWidget(self.c_lag)
        act.addWidget(b_best)
        act.addWidget(b_make)
        act.addWidget(b_all)
        act.addStretch(1)
        lay.addLayout(act)

        self.c_layout = pg.GraphicsLayoutWidget()
        lay.addWidget(self.c_layout, 2)
        self.c_info = QtWidgets.QLabel("")
        self.c_info.setWordWrap(True)
        self.c_info.setStyleSheet("font-family:monospace; font-size:11px;")
        lay.addWidget(self.c_info)

        self.c_table = QtWidgets.QTableWidget(0, 6)
        self.c_table.setHorizontalHeaderLabels(
            ["Serie", "Mejor lag", "r", "p (corregido)", "n", "Significativo"])
        self.c_table.horizontalHeader().setStretchLastSection(True)
        self.c_table.setMaximumHeight(190)
        lay.addWidget(self.c_table)
        return w

    def refresh_ccf(self) -> None:
        self.c_layout.clear()
        sx, sy = self.c_x.currentData(), self.c_y.currentData()
        if sx is None or sy is None or sx == sy:
            self.c_info.setText("Elige dos series distintas.")
            return
        x, y = self.data(sx), self.data(sy)
        res = A.ccf(x, y, self.c_maxlag.value(), self.c_pw.isChecked())
        self._ccf = res

        pl = self.c_layout.addPlot(row=0, col=0, title="CCF   (lag>0: X adelanta a Y)")
        pl.showGrid(x=True, y=True, alpha=0.2)
        pl.setLabel("bottom", "lag (muestras)")
        band = pg.FillBetweenItem(
            pg.PlotDataItem(res.lags, np.full(res.lags.size, res.conf)),
            pg.PlotDataItem(res.lags, np.full(res.lags.size, -res.conf)),
            brush=pg.mkBrush(80, 140, 255, 55))
        pl.addItem(band)
        sig = np.abs(res.ccf) > res.conf
        pl.addItem(pg.BarGraphItem(x=res.lags[~sig], height=res.ccf[~sig],
                                   width=0.7, brush="#607D8B"))
        pl.addItem(pg.BarGraphItem(x=res.lags[sig], height=res.ccf[sig],
                                   width=0.7, brush="#66BB6A"))
        pl.addItem(pg.InfiniteLine(res.best_lag, angle=90,
                                   pen=pg.mkPen("#FFD54F", width=2,
                                                style=Qt.DashLine)))
        pl.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#888")))

        self.c_lag.blockSignals(True)
        self.c_lag.setValue(res.best_lag)
        self.c_lag.blockSignals(False)
        self._draw_overlay()

        from .gaps import median_step
        step = median_step(self.session.x)
        secs = res.best_lag * step
        unit = "s" if self.session.project.source.x_is_datetime else "unidades de X"
        txt = (f"mejor lag = {res.best_lag} muestras ({secs:.6g} {unit})   "
               f"r = {res.best_r:.4f}   p = {_fmt_p(res.best_p)} "
               f"(corregido por probar {res.lags.size} lags)   n = {res.n}   "
               f"banda ±{res.conf:.3f}")
        if res.prewhitened and res.ar_order:
            txt += f"   pre-blanqueado AR({res.ar_order})"
        self.c_info.setText(txt)
        self.note.setText("  ".join(res.notes))

    def _draw_overlay(self) -> None:
        if len(self.c_layout.ci.items) < 1:
            return
        for it in list(self.c_layout.ci.items):
            if getattr(it, "_overlay", False):
                self.c_layout.removeItem(it)
        sx, sy = self.c_x.currentData(), self.c_y.currentData()
        if sx is None or sy is None or sx == sy:
            return
        lag = self.c_lag.value()
        x, y = self.data(sx), self.data(sy)
        n = min(x.size, y.size)
        xs = A.shift(x[:n], lag)
        t = np.arange(n)
        pl = self.c_layout.addPlot(row=1, col=0,
                                   title=f"{self.name(sx)} desplazada {lag} vs "
                                         f"{self.name(sy)}")
        pl._overlay = True
        pl.showGrid(x=True, y=True, alpha=0.2)
        pl.addLegend(offset=(10, 10))
        pl.plot(t, _z(y[:n]), pen=pg.mkPen(self.color(sy), width=2),
                name=self.name(sy) + " (target)")
        pl.plot(t, _z(xs), pen=pg.mkPen(self.color(sx), width=1,
                                        style=Qt.DashLine),
                name=self.name(sx) + f" (lag {lag})")
        pl.plot(t, _z(x[:n]), pen=pg.mkPen("#666", width=1),
                name=self.name(sx) + " (original)")
        d = A.lagged_corr(x[:n], y[:n], lag)
        pl.setTitle(f"Superpuestas y normalizadas (z-score) · lag={lag} · "
                    f"r={_fmt(d['r'],4)} p={_fmt_p(d['p'])} n_ef={_fmt(d['n_eff'],0)}")

    def _goto_best(self) -> None:
        if getattr(self, "_ccf", None):
            self.c_lag.setValue(self._ccf.best_lag)

    def _materialize_lag(self) -> None:
        """Crea una serie derivada X(t−lag) para poder pintarla y anotarla
        junto al target en la ventana principal."""
        sx = self.c_x.currentData()
        if sx is None:
            return
        lag = self.c_lag.value()
        from .model import KIND_LAG
        self.session.add_derived(sx, KIND_LAG, {"lag": int(lag)}, overlay=False)
        self.main.rebuild_panels()
        self.main.refresh_list()
        self.refresh()

    def _scan_all(self) -> None:
        """Todas las series contra el target Y: mejor lag de cada una."""
        sy = self.c_y.currentData()
        if sy is None:
            return
        y = self.data(sy)
        rows = []
        for s in self.session.project.ordered():
            if s.sid == sy:
                continue
            res = A.ccf(self.data(s.sid), y, self.c_maxlag.value(),
                        self.c_pw.isChecked())
            rows.append((s.name, res))
        ps = np.array([r.best_p for _, r in rows])
        sig = A.benjamini_hochberg(ps, 0.05)
        order = np.argsort([-abs(r.best_r) for _, r in rows])

        t = self.c_table
        t.setRowCount(len(rows))
        for row, k in enumerate(order):
            name, res = rows[k]
            vals = [name, str(res.best_lag), f"{res.best_r:.4f}",
                    _fmt_p(res.best_p), str(res.n),
                    "sí (FDR)" if sig[k] else "no"]
            for col, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if sig[k]:
                    it.setBackground(SIG_BG)
                t.setItem(row, col, it)
        t.resizeColumnsToContents()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        if self.session.df is None:
            return
        self.h_pick.populate(self.session)
        self.m_pick.populate(self.session)
        self._sid_combo(self.a_sid)
        self._sid_combo(self.c_x)
        self._sid_combo(self.c_y)
        self.note.setText("")
        i = self.tabs.currentIndex()
        (self.refresh_hist, self.refresh_matrix, self.refresh_acf,
         self.refresh_ccf)[i]()


# ---------------------------------------------------------------- helpers
def _bold() -> QtGui.QFont:
    f = QtGui.QFont()
    f.setBold(True)
    return f


def _z(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    v = y[np.isfinite(y)]
    if v.size < 2 or v.std() == 0:
        return y
    return (y - v.mean()) / v.std()


def _fmt(v, d=3) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def _fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _skew(v):
    from scipy import stats
    return stats.skew(v) if v.size > 2 else np.nan


def _kurt(v):
    from scipy import stats
    return stats.kurtosis(v) if v.size > 3 else np.nan
