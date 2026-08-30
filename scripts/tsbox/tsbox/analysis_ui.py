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
from .transforms import fmt_x

SIG_BG = QtGui.QColor("#1B5E20")
SIG_RAW_BG = QtGui.QColor("#4E342E")
NEG = QtGui.QColor("#EF5350")
POS = QtGui.QColor("#42A5F5")


HIST_TILE_W = 260   # tamaño BASE de cada mini-histograma: compacto y
HIST_TILE_H = 170   # predecible, no se estira aunque haya pocas series
HIST_MAX_COLS = 4
HIST_MAX_TILES = 64   # tope de histogramas dibujados a la vez


def _hist_grid_cols(n: int) -> int:
    import math
    return max(1, min(HIST_MAX_COLS, math.ceil(math.sqrt(max(1, n)))))


def hist_grid(n: int, cols: int = 0, rows: int = 0) -> tuple[int, int]:
    """Rejilla MxN para n histogramas.

    Con cols/rows a 0 se decide sola (comportamiento de siempre: casi
    cuadrada, tope HIST_MAX_COLS). Si el usuario fija una de las dos, la
    otra se deduce para que quepan TODAS las series -- nunca se recorta
    la rejilla por debajo de lo necesario, porque una serie marcada que
    no se dibuja es peor que hacer scroll. Si fija las dos y no caben,
    manda el número de columnas y se añaden filas.
    """
    n = max(1, int(n))
    cols = max(0, int(cols))
    rows = max(0, int(rows))
    if cols <= 0 and rows <= 0:
        cols = _hist_grid_cols(n)
    elif cols <= 0:
        cols = -(-n // rows)          # ceil: filas fijas, columnas las que hagan falta
    ncols = max(1, cols)
    nrows = max(1, max(rows, -(-n // ncols)))
    return ncols, nrows


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
        """Sin selección previa, no se marca NADA por defecto. Marcar todo
        de entrada hace ilegible el histograma o la matriz nada más abrir
        la ventana -- mejor que el usuario elija qué mirar."""
        prev = checked if checked is not None else set(self.checked())
        self.clear()
        for s in session.project.ordered():
            it = QtWidgets.QListWidgetItem(s.name)
            it.setData(Qt.UserRole, s.sid)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if s.sid in prev else Qt.Unchecked)
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
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setSizeGripEnabled(True)
        # Arranca ocupando ~85% de la pantalla (con un mínimo razonable en
        # monitores pequeños) en vez de un 1180x820 fijo: la matriz de
        # correlación y una rejilla de histogramas grandes necesitan sitio.
        self.setMinimumSize(900, 600)
        scr = QtGui.QGuiApplication.primaryScreen()
        av = scr.availableGeometry() if scr else QtCore.QRect(0, 0, 1400, 900)
        self.resize(max(1180, int(av.width() * 0.85)),
                    max(820, int(av.height() * 0.85)))

        root = QtWidgets.QVBoxLayout(self)

        # --- barra común
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Sección:"))
        self.section_combo = QtWidgets.QComboBox()
        self.section_combo.setMinimumWidth(260)
        self.section_combo.setToolTip(
            "Restringe TODO el análisis (las cuatro pestañas) a un tramo "
            "concreto. 'Ventana visible' es lo que estés viendo ahora mismo "
            "en el gráfico; las demás son las regiones y zonas globales que "
            "ya tengas marcadas -- crea una zona global 'arranque' en la "
            "ventana principal y aparecerá aquí.")
        self.section_combo.currentIndexChanged.connect(lambda _: self.refresh())
        bar.addWidget(self.section_combo)
        self.section_lbl = QtWidgets.QLabel("")
        self.section_lbl.setStyleSheet("color:#888;")
        bar.addWidget(self.section_lbl)
        b_ref = QtWidgets.QPushButton("Recalcular  (F5)")
        b_ref.clicked.connect(self.refresh)
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
    def refresh_sections(self) -> None:
        """Repuebla el selector de secciones con lo que haya marcado en la
        ventana principal ahora mismo: regiones, zonas globales, y las dos
        opciones fijas (todo el dataset / ventana visible). Se hace en cada
        refresh() para que una zona creada mientras el análisis está
        abierto aparezca sin tener que cerrar y reabrir la ventana.
        """
        combo = self.section_combo
        cur = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Todo el dataset", None)
        combo.addItem("Ventana visible", "window")

        is_dt = self.session.project.source.x_is_datetime
        proj = self.session.project
        sections = []
        for g in proj.global_regions:
            label = g.label or "(sin nombre)"
            sections.append((g.t0, f"{label}  · global", (g.t0, g.t1)))
        for r in proj.regions:
            s = proj.by_id(r.sid)
            label = r.label or "(sin nombre)"
            sections.append((r.t0, f"{label}  · {s.name if s else '?'}",
                            (r.t0, r.t1)))
        for _, text, rng in sorted(sections, key=lambda t: t[0]):
            combo.addItem(text, rng)

        idx = combo.findData(cur) if not isinstance(cur, tuple) else -1
        if isinstance(cur, tuple):
            for i in range(combo.count()):
                if combo.itemData(i) == cur:
                    idx = i
                    break
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

        rng = combo.currentData()
        if isinstance(rng, tuple):
            t0, t1 = rng
            self.section_lbl.setText(
                f"({fmt_x(t0, is_dt)} → {fmt_x(t1, is_dt)})")
        else:
            self.section_lbl.setText("")

    def _xrange(self):
        rng = self.section_combo.currentData()
        if rng is None:
            return None, None
        if rng == "window":
            for p in self.main.panels.values():
                if p.isVisible():
                    return tuple(p.plot.viewRange()[0])
            return None, None
        return rng   # (t0, t1) de una región o zona global concreta

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

        # --- tamaño de cada histograma y rejilla MxN elegida a mano
        self.h_zoom = QtWidgets.QSlider(Qt.Horizontal)
        self.h_zoom.setRange(50, 400)       # % sobre el tile base 260x170
        self.h_zoom.setValue(100)
        self.h_zoom.setToolTip(
            "Tamaño de cada histograma, en % del tamaño base. Súbelo para "
            "ver bien una distribución concreta: los que no quepan se "
            "alcanzan con la barra de scroll.")
        self.h_zoom_lbl = QtWidgets.QLabel("100%")
        self.h_zoom.valueChanged.connect(
            lambda v: (self.h_zoom_lbl.setText(f"{v}%"), self.refresh_hist()))
        zrow = QtWidgets.QHBoxLayout()
        zrow.addWidget(self.h_zoom, 1)
        zrow.addWidget(self.h_zoom_lbl)
        f.addRow("Tamaño", zrow)

        self.h_cols = QtWidgets.QSpinBox()
        self.h_cols.setRange(0, 16)
        self.h_cols.setSpecialValueText("auto")
        self.h_cols.setToolTip(
            "Columnas de la rejilla. 'auto' la reparte casi cuadrada.")
        self.h_cols.valueChanged.connect(lambda _: self.refresh_hist())
        self.h_rows = QtWidgets.QSpinBox()
        self.h_rows.setRange(0, 16)
        self.h_rows.setSpecialValueText("auto")
        self.h_rows.setToolTip(
            "Filas de la rejilla. Si fijas filas y columnas y no caben "
            "todas las series, se añaden filas: nunca se deja de dibujar "
            "una serie marcada.")
        self.h_rows.valueChanged.connect(lambda _: self.refresh_hist())
        grow = QtWidgets.QHBoxLayout()
        grow.addWidget(self.h_rows)
        grow.addWidget(QtWidgets.QLabel("x"))
        grow.addWidget(self.h_cols)
        f.addRow("Rejilla (filas x col)", grow)

        self.h_fit = QtWidgets.QCheckBox("Ajustar al ancho de la ventana")
        self.h_fit.setToolTip(
            "Reparte el ancho disponible entre las columnas en vez de usar "
            "el tamaño fijo. Con pocas series los histogramas se hacen "
            "grandes y no hay scroll horizontal.")
        self.h_fit.toggled.connect(lambda _: self.refresh_hist())
        f.addRow(self.h_fit)
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

        # GraphicsLayoutWidget SIN scroll se estira para llenar el hueco
        # disponible: con 2-3 series cada histograma se vuelve enorme (eso
        # es lo "ilegible de lo grande"); con muchas, se aplasta al revés.
        # Dentro de un QScrollArea, cada mini-histograma mantiene un tamaño
        # FIJO y compacto (260x170) sea cual sea el número de series, y se
        # scrollea si no caben todas -- ni gigante ni aplastado.
        self.h_layout = pg.GraphicsLayoutWidget()
        self.h_scroll = QtWidgets.QScrollArea()
        self.h_scroll.setWidgetResizable(True)
        self.h_scroll.setWidget(self.h_layout)
        lay.addWidget(self.h_scroll, 1)
        return w

    def _hist_tile_size(self, ncols: int) -> tuple[int, int]:
        """Tamaño de cada tile: el base escalado por el slider, o repartido
        a lo ancho del viewport si se ha marcado 'ajustar al ancho'. En
        modo ajuste se conserva la proporción del tile base para que un
        histograma ancho no salga con 20 px de alto."""
        k = self.h_zoom.value() / 100.0
        tw, th = int(round(HIST_TILE_W * k)), int(round(HIST_TILE_H * k))
        if self.h_fit.isChecked():
            avail = self.h_scroll.viewport().width() - 24
            if avail > 0:
                tw = max(120, avail // max(1, ncols))
                th = max(90, int(round(tw * HIST_TILE_H / HIST_TILE_W * k)))
        return tw, th

    def refresh_hist(self) -> None:
        self.h_layout.clear()
        self.h_info.clear()
        sids = self.h_pick.checked()[:HIST_MAX_TILES]
        ncols, nrows = hist_grid(len(sids), self.h_cols.value(),
                                 self.h_rows.value())
        tile_w, tile_h = self._hist_tile_size(ncols)
        lines = []
        for i, sid in enumerate(sids):
            y = self.data(sid)
            bins = self.h_bins.value() or "auto"
            h = A.histogram(y, bins, kde=self.h_norm.isChecked(),
                            prominence=self.h_prom.value())
            pl = self.h_layout.addPlot(row=i // ncols, col=i % ncols,
                                       title=self.name(sid))
            # Tamaño explícito por tile: el base HIST_TILE_* escalado por
            # el slider (o repartido a lo ancho si se pide ajustar). Fijar
            # min y max evita que el layout lo estire para llenar hueco.
            pl.setMaximumWidth(tile_w)
            pl.setMinimumWidth(tile_w)
            pl.setMaximumHeight(tile_h)
            pl.setMinimumHeight(tile_h)
            pl.setTitle(self.name(sid), size="9pt")
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

        # QScrollArea con setWidgetResizable(True) estira el widget interior
        # para llenar el viewport si este no tiene tamaño propio -- eso
        # anulaba el ancho/alto fijo de cada tile en cuanto sobraba hueco
        # (con 1-2 series, se veían gigantes otra vez). Fijar aquí el
        # tamaño del GraphicsLayoutWidget al contenido real del grid hace
        # que el scroll area lo respete: ni se estira de más, ni se
        # aplasta -- solo aparece scrollbar si de verdad no cabe.
        used_rows = (len(sids) + ncols - 1) // ncols if sids else 0
        used_cols = min(ncols, len(sids)) if sids else 0
        pad = 24
        self.h_layout.setFixedSize(
            max(1, used_cols * tile_w + pad),
            max(1, used_rows * tile_h + pad))

    def resizeEvent(self, ev):   # noqa: N802 (API de Qt)
        """En modo 'ajustar al ancho' el tamaño del tile depende del
        viewport, así que hay que recalcular al redimensionar la ventana."""
        super().resizeEvent(ev)
        if getattr(self, "h_fit", None) is not None and self.h_fit.isChecked() \
                and self.tabs.currentIndex() == 0:
            self.refresh_hist()

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
        self.refresh_sections()
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
