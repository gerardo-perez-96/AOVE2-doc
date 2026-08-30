from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import commands as cmd
from . import store
from .analysis_ui import AnalysisWindow
from .dialogs import AddColumnsDialog, DerivedDialog, LoadDialog
from .loadworker import LoadWorker
from .model import GlobalRegion, Group, Mark, Note, Region, new_id
from .panel import YMODE_FULL, YMODE_WINDOW, SeriesPanel
from .session import Session
from .transforms import fmt_x
from .viewbox import MODE_MARK, MODE_NAV, MODE_REGION, MODE_GLOBAL_REGION

AUTOSAVE_MS = 30_000





class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("tsbox")
        self.resize(1400, 900)
        self.session = Session()
        self.undo = QtGui.QUndoStack(self)
        self.panels: dict[str, SeriesPanel] = {}
        self._active_sid: str | None = None
        self._mode = MODE_NAV
        self.analysis: AnalysisWindow | None = None
        self._syncing = False

        self._build_ui()
        self._build_actions()
        self.autosave = QtCore.QTimer(self)
        self.autosave.setInterval(AUTOSAVE_MS)
        self.autosave.timeout.connect(self._autosave_tick)
        self.autosave.start()
        self._set_enabled(False)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.splitter = QtWidgets.QSplitter(Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.splitter)
        self.setCentralWidget(scroll)

        # --- dock izquierdo: series
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.addWidget(QtWidgets.QLabel(
            "<b>Series</b><br><span style='color:#888'>marca para mostrar · "
            "arrastra para reordenar</span>"))
        self.list = QtWidgets.QTreeWidget()
        self.list.setHeaderHidden(True)
        self.list.setRootIsDecorated(True)
        self.list.setIndentation(14)
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.currentItemChanged.connect(self._on_item_selected)
        self.list.model().rowsMoved.connect(self._on_rows_moved)
        self.list.itemExpanded.connect(lambda *_: self.refresh_visibility())
        self.list.itemCollapsed.connect(lambda *_: self.refresh_visibility())
        lv.addWidget(self.list, 1)
        row = QtWidgets.QHBoxLayout()
        b_add = QtWidgets.QPushButton("Derivar…")
        b_add.clicked.connect(self.add_derived)
        b_col = QtWidgets.QPushButton("Color")
        b_col.clicked.connect(self.pick_color)
        b_del = QtWidgets.QPushButton("Eliminar")
        b_del.clicked.connect(self.delete_series)
        b_restore = QtWidgets.QPushButton("Añadir señal…")
        b_restore.setToolTip("Recupera del fichero de origen una columna "
                             "que no está en la lista (p.ej. tras un "
                             "Eliminar).")
        b_restore.clicked.connect(self.add_raw_signal)
        b_none = QtWidgets.QPushButton("Ocultar todas")
        b_none.setToolTip("Desmarca todas las series de golpe, sin tener "
                          "que ir una a una.")
        b_none.clicked.connect(self.hide_all_series)
        for b in (b_add, b_col, b_del, b_restore, b_none):
            row.addWidget(b)
        lv.addLayout(row)

        # --- grupos: mostrar/ocultar varias series de golpe. No es la
        # jerarquía de derivadas (eso ya lo resuelve el árbol) -- un grupo
        # puede juntar series sin ningún parentesco entre sí.
        lv.addWidget(QtWidgets.QLabel(
            "<b>Grupos</b><br><span style='color:#888'>selecciona varias "
            "series arriba y pulsa Agrupar</span>"))
        self.group_list = QtWidgets.QListWidget()
        self.group_list.setMaximumHeight(110)
        self.group_list.itemChanged.connect(self._on_group_item_changed)
        self.group_list.itemDoubleClicked.connect(
            lambda _: self.rename_group())
        lv.addWidget(self.group_list)
        grow = QtWidgets.QHBoxLayout()
        b_group = QtWidgets.QPushButton("Agrupar seleccionadas…")
        b_group.setToolTip(
            "Selecciona varias series en la lista de arriba (Ctrl/Shift + "
            "click) y pulsa aquí para crear un grupo con ellas.")
        b_group.clicked.connect(self.create_group)
        b_rename_g = QtWidgets.QPushButton("Renombrar")
        b_rename_g.clicked.connect(self.rename_group)
        b_ungroup = QtWidgets.QPushButton("Desagrupar")
        b_ungroup.setToolTip("Borra el grupo. Las series no se tocan.")
        b_ungroup.clicked.connect(self.delete_group)
        for b in (b_group, b_rename_g, b_ungroup):
            grow.addWidget(b)
        lv.addLayout(grow)

        dock_l = QtWidgets.QDockWidget("Series", self)
        dock_l.setWidget(left)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_l)

        # --- dock derecho: anotaciones (regiones/marcas) + notas libres
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Serie", "Tipo", "Inicio", "Fin", "Duración", "Etiqueta"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self._on_table_edited)
        self.table.itemDoubleClicked.connect(self._on_table_double_clicked)
        tab_ann = QtWidgets.QWidget()
        av = QtWidgets.QVBoxLayout(tab_ann)
        av.setContentsMargins(6, 6, 6, 6)
        av.addWidget(self.table, 1)
        b_deln = QtWidgets.QPushButton("Borrar seleccionada  (Supr)")
        b_deln.clicked.connect(self.delete_selected_annotation)
        av.addWidget(b_deln)

        # Notas: texto libre, sin necesidad de dibujar nada sobre el gráfico.
        # Ctrl+N (desde cualquier sitio, en cualquier modo de edición) trae
        # el foco aquí para escribir sin soltar el hilo de lo que se está
        # mirando. El contexto (series visibles + rango X) se captura solo.
        tab_notes = QtWidgets.QWidget()
        nv = QtWidgets.QVBoxLayout(tab_notes)
        nv.setContentsMargins(6, 6, 6, 6)
        self.note_list = QtWidgets.QListWidget()
        self.note_list.itemDoubleClicked.connect(self._on_note_double_clicked)
        nv.addWidget(self.note_list, 1)
        entry = QtWidgets.QHBoxLayout()
        self.note_input = QtWidgets.QLineEdit()
        self.note_input.setPlaceholderText(
            "Apunta algo (Ctrl+N para venir aquí) · Enter para guardar")
        self.note_input.returnPressed.connect(self.add_note)
        b_addnote = QtWidgets.QPushButton("Añadir")
        b_addnote.clicked.connect(self.add_note)
        entry.addWidget(self.note_input, 1)
        entry.addWidget(b_addnote)
        nv.addLayout(entry)
        nbtn = QtWidgets.QHBoxLayout()
        b_editn = QtWidgets.QPushButton("Editar  (F2)")
        b_editn.setToolTip(
            "Edita el texto de la nota seleccionada, en la propia lista. "
            "El doble click sigue saltando al tramo que se veía al "
            "escribirla, que es lo que ya hacía antes.")
        b_editn.clicked.connect(self.edit_selected_note)
        nbtn.addWidget(b_editn)
        b_deln2 = QtWidgets.QPushButton("Borrar  (Supr)")
        b_deln2.clicked.connect(self.delete_selected_note)
        nbtn.addWidget(b_deln2)
        nv.addLayout(nbtn)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_Delete), self.note_list,
                        activated=self.delete_selected_note)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_F2), self.note_list,
                        activated=self.edit_selected_note)
        # La edición se confirma al terminar de escribir en el item. Se
        # conecta al delegate (no a itemChanged) para no confundir una
        # edición del usuario con los cambios que hace refresh_notes() al
        # repoblar la lista, que dispararían el mismo signal.
        self.note_list.itemDelegate().commitData.connect(self._on_note_edited)

        right = QtWidgets.QTabWidget()
        right.addTab(tab_ann, "Anotaciones")
        right.addTab(tab_notes, "Notas")
        self.tabs_right = right
        dock_r = QtWidgets.QDockWidget("Anotaciones y notas", self)
        dock_r.setWidget(right)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_r)
        self._dock_r = dock_r

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+N"), self,
                        activated=self.quick_note)

        self.status_msg = QtWidgets.QLabel("Abre un fichero para empezar.")
        self.status_pos = QtWidgets.QLabel("")
        self.statusBar().addWidget(self.status_msg, 1)
        self.statusBar().addPermanentWidget(self.status_pos)

    def _build_actions(self) -> None:
        tb = self.addToolBar("Principal")
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        m_file = self.menuBar().addMenu("&Archivo")
        m_edit = self.menuBar().addMenu("&Edición")
        m_view = self.menuBar().addMenu("&Ver")
        m_an = self.menuBar().addMenu("&Análisis")

        a_open = QtGui.QAction("Abrir…", self, shortcut="Ctrl+O",
                               triggered=self.open_file)
        a_save = QtGui.QAction("Guardar", self, shortcut="Ctrl+S",
                               triggered=lambda: self.save(manual=True))
        for a in (a_open, a_save):
            tb.addAction(a)
            m_file.addAction(a)
        m_file.addSeparator()
        m_file.addAction(QtGui.QAction("Salir", self, shortcut="Ctrl+Q",
                                       triggered=self.close))

        tb.addSeparator()
        self.mode_group = QtGui.QActionGroup(self)
        self.mode_group.setExclusive(True)
        for text, mode, key in (("Navegar", MODE_NAV, "N"),
                                ("Región", MODE_REGION, "R"),
                                ("Marca", MODE_MARK, "M"),
                                ("Zona global", MODE_GLOBAL_REGION, "G")):
            a = QtGui.QAction(text, self, checkable=True, shortcut=key)
            a.setData(mode)
            a.triggered.connect(lambda _=False, m=mode: self.set_mode(m))
            self.mode_group.addAction(a)
            tb.addAction(a)
            m_edit.addAction(a)
        self.mode_group.actions()[0].setChecked(True)

        m_edit.addSeparator()
        a_undo = self.undo.createUndoAction(self, "Deshacer")
        a_undo.setShortcut("Ctrl+Z")
        a_redo = self.undo.createRedoAction(self, "Rehacer")
        a_redo.setShortcut("Ctrl+Shift+Z")
        m_edit.addAction(a_undo)
        m_edit.addAction(a_redo)
        tb.addSeparator()
        tb.addAction(a_undo)
        tb.addAction(a_redo)

        tb.addSeparator()
        self.a_sync = QtGui.QAction("Zoom sincronizado", self, checkable=True,
                                    checked=True, shortcut="Ctrl+L")
        self.a_sync.toggled.connect(self._relink)
        self.a_gaps = QtGui.QAction("Marcar huecos", self, checkable=True,
                                    checked=True)
        self.a_gaps.toggled.connect(
            lambda on: [p.set_show_gaps(on) for p in self.panels.values()])
        self.a_stats = QtGui.QAction("Líneas μ/σ/min/max", self, checkable=True)
        self.a_stats.toggled.connect(
            lambda on: [p.set_show_stat_lines(on) for p in self.panels.values()])
        a_apply = QtGui.QAction("Aplicar zoom del panel activo al resto", self,
                                shortcut="Ctrl+Shift+L",
                                triggered=self.apply_zoom_to_all)
        a_reset = QtGui.QAction("Ver todo", self, shortcut="Ctrl+0",
                                triggered=self.reset_zoom)
        a_fity = QtGui.QAction("Ajustar Y a la señal", self, shortcut="Ctrl+Y",
                               triggered=self.autoscale_y)
        self.a_yfull = QtGui.QAction("Y fija al rango completo", self,
                                     checkable=True)
        self.a_yfull.setToolTip(
            "Desmarcado: la Y se ajusta a lo que se ve (la señal llena el panel).\n"
            "Marcado: la Y queda fija al min/max de toda la serie, para que al "
            "hacer zoom se note si un tramo está más alto o más bajo.")
        self.a_yfull.toggled.connect(self._set_y_mode)
        for a in (self.a_sync, self.a_gaps, self.a_stats, a_apply, a_reset,
                  a_fity, self.a_yfull):
            tb.addAction(a)
            m_view.addAction(a)

        tb.addSeparator()
        a_an = QtGui.QAction("Análisis estadístico…", self, shortcut="Ctrl+A",
                             triggered=self.open_analysis)
        tb.addAction(a_an)
        m_an.addAction(a_an)
        m_an.addAction(QtGui.QAction(
            "Informe de datos faltantes…", self, triggered=self.show_missing))

        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_Delete), self.table,
                        activated=self.delete_selected_annotation)

    def _set_enabled(self, on: bool) -> None:
        for w in (self.list, self.table):
            w.setEnabled(on)

    # ------------------------------------------------------------- abrir
    def open_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir datos", "",
            "Series temporales (*.csv *.tsv *.txt *.parquet *.pq);;Todos (*)")
        if not path:
            return
        dlg = LoadDialog(Path(path), self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        kw = dict(data_path=path, x_mode=dlg.x_mode(), x_column=dlg.x_column(),
                  max_samples=dlg.max_samples(), sample_policy=dlg.sample_policy(),
                  selected_columns=dlg.selected_columns(), float32=dlg.float32(),
                  long_mode=dlg.long_mode(), group_column=dlg.group_column(),
                  group_value=dlg.group_value())
        self._path = path

        self._prog = QtWidgets.QProgressDialog("Preparando...", "Cancelar", 0, 100, self)
        self._prog.setWindowTitle("Abriendo")
        self._prog.setWindowModality(Qt.WindowModal)
        self._prog.setMinimumDuration(300)
        self._prog.setAutoClose(False)
        self._prog.setAutoReset(False)

        self._thread = QtCore.QThread(self)
        self._worker = LoadWorker(self.session, kw)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.sigProgress.connect(self._on_load_progress)
        self._worker.sigDone.connect(self._on_load_done)
        self._prog.canceled.connect(self._worker.cancel)
        self._thread.start()

    @QtCore.Slot(str, float)
    def _on_load_progress(self, msg: str, frac: float) -> None:
        self._prog.setLabelText(msg)
        self._prog.setValue(int(frac * 100))

    @QtCore.Slot(bool, str)
    def _on_load_done(self, ok: bool, err: str) -> None:
        self._thread.quit()
        self._thread.wait()
        self._prog.reset()
        self._prog.close()
        if not ok:
            if "cancelada" not in err.lower():
                QtWidgets.QMessageBox.critical(self, "No se pudo abrir", err)
            return

        self.undo.clear()
        self.rebuild_panels()
        self._set_enabled(True)
        path = self._path
        self.setWindowTitle(f"tsbox \u2014 {Path(path).name}")
        self.status_msg.setText(
            f"{len(self.session.x):,} muestras \u00b7 "
            f"{len(self.session.project.series)} series \u00b7 "
            f"{self.session.memory_mb():.0f} MB en RAM \u00b7 "
            f"JSON: {self.session.json_path.name}")
        if self.session.warnings:
            QtWidgets.QMessageBox.warning(self, "Avisos al abrir",
                                          "\n\n".join(self.session.warnings))
        view = self.session.project.view.get("xrange")
        if view:
            self.set_x_range(*view)

    # ------------------------------------------------------------ paneles
    def panel_series(self):
        """Todas las series tienen panel propio, incluidas las derivadas.
        Su visibilidad la controla el checkbox del árbol; su posición, estar
        plegadas/expandidas bajo la original no oculta el panel por sí solo
        -- eso lo hace refresh_visibility()."""
        return self.session.project.ordered()

    def rebuild_panels(self) -> None:
        for p in self.panels.values():
            p.setParent(None)
            p.deleteLater()
        self.panels.clear()
        for s in self.panel_series():
            p = SeriesPanel(self.session, s)
            p.sigRegionDrawn.connect(self.on_region_drawn)
            p.sigGlobalRegionDrawn.connect(self.on_global_region_drawn)
            p.sigMarkDrawn.connect(self.on_mark_drawn)
            p.sigPanStep.connect(self.on_pan_step)
            p.sigReorderDrop.connect(self.on_panel_drop)
            p.sigRegionClicked.connect(self.on_region_clicked)
            p.sigDragStarted.connect(self._on_panel_drag_started)
            p.sigDragEnded.connect(self._on_panel_drag_ended)
            p.sigRegionEdited.connect(self.on_region_edited)
            p.sigGlobalRegionEdited.connect(self.on_global_region_edited)
            p.sigCloseRequested.connect(self.hide_series)
            p.sigCursor.connect(self.on_cursor)
            p.set_show_gaps(self.a_gaps.isChecked())
            p.set_show_stat_lines(self.a_stats.isChecked())
            p.set_mode(self._mode)
            self.splitter.addWidget(p)
            self.panels[s.sid] = p
        self.refresh_list()
        self.refresh_visibility()
        self.refresh_annotations()
        self.refresh_annotation_table()
        self.refresh_groups()
        self.refresh_notes()
        self._relink(self.a_sync.isChecked())
        self._apply_y_mode_to_new()

    def _apply_y_mode_to_new(self) -> None:
        mode = YMODE_FULL if self.a_yfull.isChecked() else YMODE_WINDOW
        for p in self.panels.values():
            p.set_y_mode(mode)

    def refresh_visibility(self) -> None:
        """Un panel se muestra si su checkbox está marcado Y, siendo una
        derivada, su nodo padre está expandido en el árbol. Plegar el
        original oculta sus derivadas sin tener que desmarcarlas una a una;
        al desplegar vuelven tal como estaban."""
        for sid, p in self.panels.items():
            sd = self.session.project.by_id(sid)
            visible = sd.visible
            if visible and sd.parent is not None:
                parent_item = self._tree_items.get(sd.parent) if hasattr(
                    self, "_tree_items") else None
                if parent_item is not None and not parent_item.isExpanded():
                    visible = False
            p.setVisible(visible)
        for p in self.panels.values():
            p.redraw_overlays()
        self._relink(self.a_sync.isChecked())
        self._refresh_x_axes()

    def _refresh_x_axes(self) -> None:
        """Solo el ultimo panel visible muestra el eje X. Los demas comparten
        el mismo rango, asi que su eje seria una copia identica repintada en
        cada frame: el 27% del coste de un pan con 18 paneles."""
        vis = [p for s in self.panel_series()
               if (p := self.panels.get(s.sid)) is not None and p.isVisibleTo(self)]
        shared = self.a_sync.isChecked() and len(vis) > 1
        for i, p in enumerate(vis):
            p.set_x_axis_visible((not shared) or i == len(vis) - 1)

    def _relink(self, on: bool) -> None:
        vis = [p for s in self.panel_series()
               if (p := self.panels.get(s.sid)) and p.isVisible()]
        if not vis:
            return
        master = vis[0]
        for p in vis[1:]:
            p.plot.setXLink(master.plot if on else None)

    def apply_zoom_to_all(self) -> None:
        p = self.panels.get(self._active_sid) or next(iter(self.panels.values()), None)
        if p is None:
            return
        (x0, x1), _ = p.vb.viewRange()
        self.set_x_range(x0, x1)

    def set_x_range(self, x0: float, x1: float) -> None:
        for p in self.panels.values():
            p.vb.setXRange(x0, x1, padding=0)

    PAN_STEP_FRACTION = 0.2   # 20% de la ventana visible por pulsación

    def on_pan_step(self, direction: int) -> None:
        """Botón lateral del ratón (atrás/adelante): desplaza la ventana un
        paso, manteniendo el zoom. Afecta a TODAS las series a la vez -- es
        navegación global, no una interacción de un panel suelto, así que no
        depende de si el zoom sincronizado está activado o no.

        Es un paso por click, no repetición mientras se mantiene pulsado
        (los botones laterales se pulsan y sueltan, no se mantienen como un
        joystick). Si en el uso real se queda corto, PAN_STEP_FRACTION es
        el único número que hay que tocar.
        """
        p = self.panels.get(self._active_sid) or next(iter(self.panels.values()), None)
        if p is None:
            return
        (x0, x1), _ = p.vb.viewRange()
        step = (x1 - x0) * self.PAN_STEP_FRACTION * direction
        self.set_x_range(x0 + step, x1 + step)

    def autoscale_y(self) -> None:
        for p in self.panels.values():
            p.autoscale_y()
        self.a_yfull.setChecked(False)

    def _set_y_mode(self, full: bool) -> None:
        mode = YMODE_FULL if full else YMODE_WINDOW
        for p in self.panels.values():
            p.set_y_mode(mode)

    def reset_zoom(self) -> None:
        if len(self.session.x):
            self.set_x_range(float(self.session.x[0]), float(self.session.x[-1]))

    # ----------------------------------------------------------- lista
    def refresh_list(self) -> None:
        """Arbol: cada serie de origen es una raiz; sus derivadas cuelgan
        debajo, plegadas por defecto. Antes una derivada solo podia salir de
        su panel superponiendose en el mismo grafico con eje Y secundario, que
        es justo lo que costaba leer con varias señales a la vez. Ahora cada
        derivada tiene su PROPIO panel, plegable bajo el original: se analiza
        aparte, con su propia escala Y, sin abarrotar la vista por defecto."""
        self.list.blockSignals(True)
        expanded = {sid for sid, it in self._tree_items.items()
                   if it.isExpanded()} if hasattr(self, "_tree_items") else set()
        self.list.clear()
        self._tree_items: dict[str, QtWidgets.QTreeWidgetItem] = {}

        roots = [s for s in self.session.project.ordered() if s.parent is None]
        for root in roots:
            it = self._make_item(root)
            self.list.addTopLevelItem(it)
            self._tree_items[root.sid] = it
            children = sorted(self.session.project.children_of(root.sid),
                              key=lambda c: c.order)
            for child in children:
                cit = self._make_item(child)
                it.addChild(cit)
                self._tree_items[child.sid] = cit
            if children:
                it.setExpanded(root.sid in expanded)
        self.list.blockSignals(False)

    def _make_item(self, s) -> QtWidgets.QTreeWidgetItem:
        label = s.describe() if s.is_derived else s.name
        it = QtWidgets.QTreeWidgetItem([label])
        it.setData(0, Qt.UserRole, s.sid)
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
        it.setCheckState(0, Qt.Checked if s.visible else Qt.Unchecked)
        it.setForeground(0, QtGui.QBrush(QtGui.QColor(s.color)))
        it.setToolTip(0, f"{s.name} · {s.describe()}" if s.is_derived else s.name)
        return it

    def _on_item_changed(self, item, col=0) -> None:
        sid = item.data(0, Qt.UserRole)
        s = self.session.project.by_id(sid)
        if s is None:
            return
        s.visible = item.checkState(0) == Qt.Checked
        self.session.dirty = True
        self.refresh_visibility()
        self.refresh_groups()   # el estado mixto de un grupo puede cambiar

    def _on_item_selected(self, item, _prev) -> None:
        if item:
            self._active_sid = item.data(0, Qt.UserRole)

    def _on_rows_moved(self, *_) -> None:
        order: list[str] = []
        for i in range(self.list.topLevelItemCount()):
            root = self.list.topLevelItem(i)
            order.append(root.data(0, Qt.UserRole))
            for j in range(root.childCount()):
                order.append(root.child(j).data(0, Qt.UserRole))
        self.undo.push(cmd.ReorderSeries(self, order))

    def _on_panel_drag_started(self) -> None:
        """Arrastrar un panel es click+arrastre sobre un icono diminuto, pero
        Qt entra en un bucle de eventos nativo aparte (drag.exec()) que sigue
        procesando la app mientras dura -- incluidos los repintados de los
        18 paneles bajo el cursor. Se congelan aquí para que ese trabajo no
        compita con el propio arrastre; setUpdatesEnabled es la técnica
        estándar de Qt para esto, la misma que usan Qt Designer/Creator al
        arrastrar widgets pesados."""
        self.splitter.setUpdatesEnabled(False)

    def _on_panel_drag_ended(self) -> None:
        self.splitter.setUpdatesEnabled(True)
        self.splitter.update()

    def on_panel_drop(self, dragged_sid: str, onto_sid: str, before: bool) -> None:
        """Reordenar arrastrando el panel EN LA INTERFAZ, no solo en la
        lista lateral. Mismo resultado final que _on_rows_moved: se
        recoloca en la lista GLOBAL de series (todas, visibles u ocultas,
        originales y derivadas), porque el orden es una única propiedad del
        proyecto -- la lista lateral y los paneles son dos vistas de lo
        mismo, no dos órdenes independientes.
        """
        order = [s.sid for s in self.session.project.ordered()]
        if dragged_sid not in order or onto_sid not in order:
            return
        order.remove(dragged_sid)
        idx = order.index(onto_sid)
        order.insert(idx if before else idx + 1, dragged_sid)
        self.undo.push(cmd.ReorderSeries(self, order))

    def hide_series(self, sid: str) -> None:
        s = self.session.project.by_id(sid)
        if s:
            s.visible = False
            self.session.dirty = True
            self.refresh_list()
            self.refresh_visibility()

    def hide_all_series(self) -> None:
        """Desmarca todas las series de golpe. Con muchas series, ir una a
        una en el árbol es justo el tipo de trabajo repetitivo que no
        debería hacer falta -- un solo botón, un solo refresco, no N.
        """
        if not self.session.project.series:
            return

        def _uncheck(item: QtWidgets.QTreeWidgetItem) -> None:
            item.setCheckState(0, Qt.Unchecked)
            for i in range(item.childCount()):
                _uncheck(item.child(i))

        self.list.blockSignals(True)
        for i in range(self.list.topLevelItemCount()):
            _uncheck(self.list.topLevelItem(i))
        self.list.blockSignals(False)

        for s in self.session.project.series:
            s.visible = False
        self.session.dirty = True
        self.refresh_visibility()
        self.refresh_groups()

    # -------------------------------------------------------------- grupos
    def _selected_series_sids(self) -> list[str]:
        sids = []
        for it in self.list.selectedItems():
            sid = it.data(0, Qt.UserRole)
            if sid and sid not in sids:
                sids.append(sid)
        return sids

    def create_group(self) -> None:
        sids = self._selected_series_sids()
        if not sids:
            QtWidgets.QMessageBox.information(
                self, "Agrupar",
                "Selecciona una o más series en la lista de arriba "
                "(Ctrl/Shift + click) antes de agrupar.")
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Nuevo grupo", "Nombre del grupo:")
        name = name.strip()
        if not ok or not name:
            return
        g = Group(gid=new_id("grp"), name=name, members=sids)
        self.undo.push(cmd.AddGroup(self, g))

    def rename_group(self) -> None:
        it = self.group_list.currentItem()
        if it is None:
            return
        gid = it.data(Qt.UserRole)
        g = next((x for x in self.session.project.groups if x.gid == gid), None)
        if g is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Renombrar grupo", "Nombre del grupo:", text=g.name)
        name = name.strip()
        if not ok or not name or name == g.name:
            return
        self.undo.push(cmd.RenameGroup(self, gid, name))

    def delete_group(self) -> None:
        it = self.group_list.currentItem()
        if it is None:
            return
        self.undo.push(cmd.DeleteGroup(self, it.data(Qt.UserRole)))

    def refresh_groups(self) -> None:
        """Repuebla la lista de grupos. El checkbox de cada uno refleja si
        TODOS sus miembros están visibles ahora mismo; si el estado es
        mixto (o ninguno visible), se muestra desmarcado con un "(N/M)" en
        el texto -- no se usa el tercer estado nativo de Qt (tristate) a
        propósito: su ciclo de click depende de detalles internos difíciles
        de testear, mientras que un checkbox binario de verdad + una pista
        en el texto es exactamente igual de claro y 100% predecible."""
        self.group_list.blockSignals(True)
        self.group_list.clear()
        proj = self.session.project
        for g in proj.groups:
            members = [m for sid in g.members if (m := proj.by_id(sid)) is not None]
            n_tot = len(members)
            n_vis = sum(1 for m in members if m.visible)
            all_visible = n_tot > 0 and n_vis == n_tot
            label = g.name if (n_tot == 0 or all_visible) else f"{g.name}  ({n_vis}/{n_tot})"
            it = QtWidgets.QListWidgetItem(label)
            it.setData(Qt.UserRole, g.gid)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if all_visible else Qt.Unchecked)
            names = ", ".join(m.name for m in members[:10])
            it.setToolTip(f"{n_tot} serie(s): {names}" +
                         ("…" if n_tot > 10 else ""))
            self.group_list.addItem(it)
        self.group_list.blockSignals(False)

    def _on_group_item_changed(self, item) -> None:
        gid = item.data(Qt.UserRole)
        visible = item.checkState() == Qt.Checked
        self.undo.push(cmd.SetGroupVisibility(self, gid, visible))

    def pick_color(self) -> None:
        s = self.session.project.by_id(self._active_sid)
        if not s:
            return
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(s.color), self)
        if c.isValid():
            s.color = c.name()
            self.session.dirty = True
            self.refresh_list()
            for p in self.panels.values():
                p.redraw()

    def delete_series(self) -> None:
        s = self.session.project.by_id(self._active_sid)
        if not s:
            return
        kids = len(self.session.project.children_of(s.sid))
        extra = f" y sus {kids} derivadas" if kids else ""
        if QtWidgets.QMessageBox.question(
                self, "Eliminar",
                f"Eliminar «{s.name}»{extra}, con sus anotaciones. No hay deshacer "
                f"para esto.") != QtWidgets.QMessageBox.Yes:
            return
        self.session.project.remove_series(s.sid)
        self.session.invalidate(s.sid)
        self.session.dirty = True
        self.rebuild_panels()

    def add_raw_signal(self) -> None:
        cols = self.session.available_columns()
        if not cols:
            QtWidgets.QMessageBox.information(
                self, "Añadir señal",
                "No hay columnas del fichero de origen pendientes de "
                "añadir: todas las numéricas ya están en la lista de "
                "series.")
            return
        dlg = AddColumnsDialog(cols, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        chosen = dlg.selected_columns()
        if not chosen:
            return
        for c in chosen:
            self.session.add_raw_series(c)
        self.rebuild_panels()

    def add_derived(self) -> None:
        if not self.session.project.series:
            return
        dlg = DerivedDialog(self.session.project, self._active_sid, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        parent, kind, params, overlay = dlg.result_params()
        # una derivada superpuesta solo tiene sentido sobre una serie con panel
        p = self.session.project.by_id(parent)
        if overlay and p.is_derived and p.overlay_on_parent:
            overlay = False
        self.session.add_derived(parent, kind, params, overlay)
        self.rebuild_panels()

    # ------------------------------------------------------ anotaciones
    def find_annotation(self, aid: str):
        for coll in (self.session.project.regions, self.session.project.marks,
                    self.session.project.global_regions):
            for a in coll:
                if a.aid == aid:
                    return a
        return None

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        for p in self.panels.values():
            p.set_mode(mode)
        hint = {MODE_NAV: "Navegar: arrastra para desplazar, rueda para zoom.",
                MODE_REGION: "Región: arrastra sobre la señal para etiquetar un tramo.",
                MODE_MARK: "Marca: click sobre la señal para poner una marca.",
                MODE_GLOBAL_REGION: "Zona global: arrastra en cualquier panel para "
                                    "marcar una franja en TODAS las series a la vez."}
        self.status_msg.setText(hint[mode])

    def on_region_drawn(self, sid: str, t0: float, t1: float) -> None:
        r = Region(aid=new_id("r"), sid=sid, t0=min(t0, t1), t1=max(t0, t1),
                   label="")
        self.undo.push(cmd.AddRegion(self, r))

    def on_global_region_drawn(self, t0: float, t1: float) -> None:
        r = GlobalRegion(aid=new_id("g"), t0=min(t0, t1), t1=max(t0, t1), label="")
        self.undo.push(cmd.AddGlobalRegion(self, r))

    def on_mark_drawn(self, sid: str, t: float) -> None:
        self.undo.push(cmd.AddMark(self, Mark(aid=new_id("m"), sid=sid, t=t)))

    def on_region_edited(self, aid: str, t0: float, t1: float) -> None:
        r = self.find_annotation(aid)
        if r is None or (abs(r.t0 - t0) < 1e-12 and abs(r.t1 - t1) < 1e-12):
            return
        self.undo.push(cmd.EditRegion(self, aid, t0, t1))

    def on_global_region_edited(self, aid: str, t0: float, t1: float) -> None:
        r = self.find_annotation(aid)
        if r is None or (abs(r.t0 - t0) < 1e-12 and abs(r.t1 - t1) < 1e-12):
            return
        self.undo.push(cmd.EditGlobalRegion(self, aid, t0, t1))

    def on_region_clicked(self, aid: str) -> None:
        """Click en una región/zona global sobre el gráfico: selecciona la
        fila correspondiente en la tabla de Anotaciones y trae esa pestaña
        al frente, para no tener que buscarla a mano entre docenas de filas.
        """
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None and it.data(Qt.UserRole) == aid:
                self.table.clearSelection()
                self.table.selectRow(r)
                self.table.scrollToItem(it)
                self._dock_r.show()
                self._dock_r.raise_()
                self.tabs_right.setCurrentWidget(self.tabs_right.widget(0))
                return

    def refresh_annotations(self) -> None:
        movable = self._mode == MODE_REGION
        movable_global = self._mode == MODE_GLOBAL_REGION
        for sid, p in self.panels.items():
            p.sync_annotations(
                [r for r in self.session.project.regions if r.sid == sid],
                [m for m in self.session.project.marks if m.sid == sid],
                movable)
            # las zonas globales se ven en TODOS los paneles, no solo el suyo
            p.sync_global_regions(self.session.project.global_regions,
                                  movable_global)

    def refresh_annotation_table(self) -> None:
        is_dt = self.session.project.source.x_is_datetime
        rows = ([("región", r) for r in self.session.project.regions] +
                [("marca", m) for m in self.session.project.marks] +
                [("zona global", g) for g in self.session.project.global_regions])
        rows.sort(key=lambda t: getattr(t[1], "t0", getattr(t[1], "t", 0.0)))
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for i, (kind, a) in enumerate(rows):
            if kind == "zona global":
                name = "— todas las series —"
            else:
                s = self.session.project.by_id(a.sid)
                name = s.name if s else "?"
            is_point = kind == "marca"
            t0 = a.t if is_point else a.t0
            t1 = a.t if is_point else a.t1
            vals = [name, kind, fmt_x(t0, is_dt), fmt_x(t1, is_dt),
                    "" if is_point else f"{t1 - t0:.6g}"]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                it.setData(Qt.UserRole, a.aid)
                self.table.setItem(i, c, it)
            lab = QtWidgets.QTableWidgetItem(a.label)
            lab.setData(Qt.UserRole, a.aid)
            self.table.setItem(i, 5, lab)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()

    def _on_table_edited(self, item) -> None:
        if item.column() != 5:
            return
        aid = item.data(Qt.UserRole)
        a = self.find_annotation(aid)
        if a is not None and a.label != item.text():
            self.undo.push(cmd.RenameAnnotation(self, aid, item.text()))

    def _on_table_double_clicked(self, item) -> None:
        a = self.find_annotation(item.data(Qt.UserRole))
        if a is None:
            return
        t0 = getattr(a, "t0", getattr(a, "t", 0.0))
        t1 = getattr(a, "t1", getattr(a, "t", 0.0))
        pad = max((t1 - t0) * 0.3, 1e-9)
        self.set_x_range(t0 - pad, t1 + pad)

    def delete_selected_annotation(self) -> None:
        rows = {i.row() for i in self.table.selectedIndexes()}
        aids = [self.table.item(r, 0).data(Qt.UserRole) for r in sorted(rows)]
        if not aids:
            return
        self.undo.beginMacro(f"Borrar {len(aids)} anotaciones")
        for aid in aids:
            a = self.find_annotation(aid)
            if a is None:
                continue
            if isinstance(a, GlobalRegion):
                self.undo.push(cmd.DeleteGlobalRegion(self, aid))
            elif isinstance(a, Region):
                self.undo.push(cmd.DeleteRegion(self, aid))
            else:
                self.undo.push(cmd.DeleteMark(self, aid))
        self.undo.endMacro()

    # ------------------------------------------------------------ notas
    def quick_note(self) -> None:
        """Trae el foco al cuadro de notas desde cualquier sitio y en
        cualquier modo de edición (navegar/región/marca), sin tocar el
        canvas. Ctrl+N. Es la vía rápida para apuntar algo al vuelo sin
        interrumpir lo que se está mirando para ir a dibujar una marca."""
        self._dock_r.show()
        self._dock_r.raise_()
        self.tabs_right.setCurrentWidget(self.tabs_right.widget(1))
        self.note_input.setFocus()
        self.note_input.selectAll()

    def _current_context(self) -> tuple[float | None, float | None, list[str]]:
        """Rango X visible y series mostradas ahora mismo, para adjuntarlo a
        una nota sin que el usuario tenga que señalar nada él mismo -- es
        exactamente lo que se pidió: anotar sin necesidad de marcar."""
        vis_sids = [s.sid for s in self.panel_series()
                   if (p := self.panels.get(s.sid)) is not None
                   and p.isVisibleTo(self)]
        p = self.panels.get(self._active_sid)
        if p is None or not p.isVisibleTo(self):
            p = next((self.panels[sid] for sid in vis_sids), None)
        if p is None:
            return None, None, vis_sids
        (x0, x1), _ = p.vb.viewRange()
        return float(x0), float(x1), vis_sids

    def add_note(self) -> None:
        text = self.note_input.text().strip()
        if not text:
            return
        x0, x1, series = self._current_context()
        note = Note(nid=new_id("n"), text=text, created_at=time.time(),
                    x0=x0, x1=x1, series=series)
        self.undo.push(cmd.AddNote(self, note))
        self.note_input.clear()

    def refresh_notes(self) -> None:
        self.note_list.clear()
        is_dt = self.session.project.source.x_is_datetime
        for n in sorted(self.session.project.notes, key=lambda n: n.created_at):
            when = dt.datetime.fromtimestamp(n.created_at).strftime("%d/%m %H:%M")
            names = ", ".join(
                s.name for sid in n.series
                if (s := self.session.project.by_id(sid)) is not None)
            where = (f"{fmt_x(n.x0, is_dt)} – {fmt_x(n.x1, is_dt)}"
                     if n.x0 is not None and n.x1 is not None else "")
            ctx = "  ·  ".join(c for c in (names, where) if c)
            it = QtWidgets.QListWidgetItem(f"[{when}]  {n.text}")
            it.setData(Qt.UserRole, n.nid)
            it.setToolTip(f"{ctx}\n\n{n.text}" if ctx else n.text)
            self.note_list.addItem(it)

    def edit_selected_note(self) -> None:
        """Abre el item seleccionado para escribir encima."""
        items = self.note_list.selectedItems()
        it = items[0] if items else self.note_list.currentItem()
        if it is None:
            return
        # Al editar se muestra SOLO el texto: el prefijo "[dd/mm HH:MM]" es
        # decoración de la lista, y dejarlo dentro del editor haría que el
        # usuario lo borrase o lo guardase como parte de la nota.
        n = self._note_by_id(it.data(Qt.UserRole))
        if n is None:
            return
        it.setFlags(it.flags() | Qt.ItemIsEditable)
        it.setText(n.text)
        self.note_list.editItem(it)

    def _note_by_id(self, nid):
        return next((n for n in self.session.project.notes if n.nid == nid), None)

    def _on_note_edited(self, editor) -> None:
        it = self.note_list.currentItem()
        if it is None:
            return
        nid = it.data(Qt.UserRole)
        n = self._note_by_id(nid)
        text = editor.text().strip() if hasattr(editor, "text") else ""
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        # Texto vacío = no cambiar nada. Borrar una nota es Supr, explícito;
        # vaciarla sin querer y perderla sería una sorpresa desagradable.
        if n is None or not text or text == n.text:
            self.refresh_notes()
            return
        self.undo.push(cmd.EditNote(self, nid, text))

    def delete_selected_note(self) -> None:
        nids = {i.data(Qt.UserRole) for i in self.note_list.selectedItems()}
        if not nids:
            return
        self.undo.beginMacro(f"Borrar {len(nids)} nota(s)")
        for nid in nids:
            self.undo.push(cmd.DeleteNote(self, nid))
        self.undo.endMacro()

    def _on_note_double_clicked(self, item) -> None:
        """Doble click en una nota: salta al rango que se estaba viendo
        cuando se escribió, igual que hace la tabla de anotaciones."""
        n = next((n for n in self.session.project.notes
                  if n.nid == item.data(Qt.UserRole)), None)
        if n is None or n.x0 is None or n.x1 is None:
            return
        pad = max((n.x1 - n.x0) * 0.3, 1e-9)
        self.set_x_range(n.x0 - pad, n.x1 + pad)

    def on_cursor(self, x: float, y: float) -> None:
        panel = self.sender()
        if isinstance(panel, SeriesPanel):
            self._active_sid = panel.sid
        self.status_pos.setText(
            f"x {fmt_x(x, self.session.project.source.x_is_datetime)}   y {y:.6g}")

    # -------------------------------------------------------- guardado
    def _capture_view(self) -> None:
        p = next((p for p in self.panels.values() if p.isVisible()), None)
        if p is not None:
            (x0, x1), _ = p.vb.viewRange()
            self.session.project.view["xrange"] = [float(x0), float(x1)]

    def save(self, manual: bool = False) -> None:
        if self.session.json_path is None:
            return
        self._capture_view()
        try:
            path = self.session.save()
        except FileExistsError as e:
            QtWidgets.QMessageBox.critical(self, "No se guarda", str(e))
            self.autosave.stop()
            return
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Error al guardar", str(e))
            return
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        self.status_msg.setText(
            f"{'Guardado' if manual else 'Autoguardado'} {stamp} → {path.name}")

    def _autosave_tick(self) -> None:
        if self.session.json_path and self.session.dirty:
            self.save(manual=False)

    def closeEvent(self, ev) -> None:
        if self.session.json_path and self.session.dirty:
            r = QtWidgets.QMessageBox.question(
                self, "Cambios sin guardar", "Guardar antes de salir.",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard |
                QtWidgets.QMessageBox.Cancel)
            if r == QtWidgets.QMessageBox.Cancel:
                ev.ignore()
                return
            if r == QtWidgets.QMessageBox.Save:
                self.save(manual=True)
        ev.accept()

    # -------------------------------------------------------- análisis
    def open_analysis(self) -> None:
        if self.session.df is None:
            QtWidgets.QMessageBox.information(self, "Análisis",
                                              "Abre un fichero primero.")
            return
        if self.analysis is None:
            self.analysis = AnalysisWindow(self.session, self, self)
        self.analysis.refresh()
        self.analysis.show()
        self.analysis.raise_()

    def show_missing(self) -> None:
        if self.session.df is None:
            return
        is_dt = self.session.project.source.x_is_datetime
        lines = []
        for s in self.session.project.ordered():
            rep = self.session.missing(s.sid)
            if not rep["n_nan"] and not rep["time_gaps"]:
                continue
            lines.append(f"{s.name}")
            if rep["n_nan"]:
                lines.append(f"   valores NaN: {rep['n_nan']}")
                for a, b in rep["nan_intervals"][:20]:
                    lines.append(f"      {fmt_x(a, is_dt)} → {fmt_x(b, is_dt)}")
                if len(rep["nan_intervals"]) > 20:
                    lines.append(f"      … y {len(rep['nan_intervals'])-20} más")
            if rep["time_gaps"]:
                lines.append(f"   saltos en el eje X: {len(rep['time_gaps'])} "
                             f"(≈{rep['muestras_perdidas_estimadas']} muestras "
                             f"perdidas, dt mediano {rep['dt_mediano']:.6g})")
                for a, b in rep["time_gaps"][:20]:
                    lines.append(f"      {fmt_x(a, is_dt)} → {fmt_x(b, is_dt)}")
            lines.append("")
        txt = "\n".join(lines) or "No falta ningún dato en ninguna serie."
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Datos faltantes")
        dlg.resize(760, 560)
        v = QtWidgets.QVBoxLayout(dlg)
        te = QtWidgets.QPlainTextEdit(txt)
        te.setReadOnly(True)
        te.setStyleSheet("font-family:monospace; font-size:11px;")
        v.addWidget(te)
        b = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        b.rejected.connect(dlg.reject)
        v.addWidget(b)
        dlg.exec()
