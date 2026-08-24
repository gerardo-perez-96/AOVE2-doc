"""Un panel = una serie base + sus derivadas superpuestas."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from .model import GlobalRegion, Mark, Region, SeriesDef
from .transforms import fmt_stats
from .viewbox import MODE_NAV, EditViewBox

GAP_BRUSH = pg.mkBrush(220, 40, 40, 55)
GAP_PEN = pg.mkPen(220, 40, 40, 120, width=1, style=Qt.DashLine)

YMODE_WINDOW = "window"   # Y se ajusta a lo visible (por defecto)
YMODE_FULL = "full"       # Y fija al rango de toda la serie
YMODE_MANUAL = "manual"   # el usuario ha fijado el zoom Y a mano
Y_PAD = 0.06              # 6% de margen: si la señal toca el borde, no se lee
GAP_DENSE_COVER = 0.25    # si los huecos tapan mas de esto, franja fina abajo
GAP_STRIP_FRAC = 0.07     # alto de esa franja, en fraccion del panel

MAX_GAP_ITEMS = 200          # por encima de esto, se dibuja vectorizado
GAP_DETAIL_LIMIT = 20_000    # por encima, se agrupan en bandas
PANEL_MIME = "application/x-tsbox-panel-sid"


class GapOverlay(pg.GraphicsObject):
    """Todos los tramos de datos faltantes en UN SOLO item grafico.

    Tres cosas que la version anterior hacia mal y que se veian en pantalla:

    1. Un LinearRegionItem por tramo. Con 44.000 tramos, 32 s por panel.
    2. Rectangulos de 2e12 de alto "para cubrir siempre el panel". Eso metia
       -1e12..1e12 en el autorange (la señal quedaba aplastada en una linea
       recta) y desbordaba el rasterizador de Qt, que pinta en punto fijo 26.6:
       de ahi el panel entero embadurnado de rojo.
    3. Agrupar los 44.000 tramos UNA VEZ, globalmente. Con 90 dias de datos y
       un limite de 20.000 bandas, la tolerancia de agrupado eran 390 s: al
       hacer zoom a un minuto seguias viendo las bandas gordas de antes, que
       ya no correspondian a ningun hueco real.

    Ahora: se guardan todos los tramos, y en cada repintado se seleccionan solo
    los del rango visible y se agrupan a resolucion de PIXEL. Al hacer zoom, las
    bandas se separan y se ven los huecos de verdad. Es lo mismo que hace un
    mapa: no dibuja cada calle cuando ves el pais entero, pero al acercarte
    aparecen.
    """

    def __init__(self, intervals, brush=GAP_BRUSH):
        super().__init__()
        self._brush = brush
        self.set_intervals(intervals)

    def set_intervals(self, intervals) -> None:
        iv = (np.asarray(intervals, dtype=np.float64) if len(intervals)
              else np.empty((0, 2)))
        if iv.size:
            iv = iv[np.argsort(iv[:, 0], kind="stable")]
        self._iv = iv
        self._n = len(iv)
        self._x0 = float(iv[:, 0].min()) if self._n else 0.0
        self._x1 = float(iv[:, 1].max()) if self._n else 0.0
        self._rects: list[QtCore.QRectF] = []
        self._key = None
        self._dense = False
        self._cover = 0.0
        self.prepareGeometryChange()
        self.update()

    # --- el autorange debe ignorarnos --------------------------------
    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        """pyqtgraph pregunta esto para el autorange. En Y devolvemos None:
        una franja de hueco es decoracion, no debe estirar la escala."""
        if self._n == 0:
            return [None, None]
        return [self._x0, self._x1] if ax == 0 else [None, None]

    def pixelPadding(self):
        return 0

    def _view(self):
        vb = self.getViewBox()
        if vb is None:
            return None
        (x0, x1), (y0, y1) = vb.viewRange()
        w = max(1.0, vb.width())
        if not all(np.isfinite(v) for v in (x0, x1, y0, y1)) or x1 <= x0:
            return None
        return x0, x1, y0, y1, w

    def viewRangeChanged(self):
        self.prepareGeometryChange()
        self.update()

    def _rebuild(self, x0, x1, w) -> None:
        """Recalcula la lista de rectangulos para el rango visible.

        Antes esto construia un QPicture y en cada frame se hacia play(). Medido:
        0.057 ms por repintado frente a 0.014 ms de un solo drawRects() con la
        lista. Con 18 paneles a 60 Hz esa diferencia es la que hace que el pan
        vaya a tirones. Un QPicture reejecuta la secuencia de comandos grabada;
        drawRects entrega el lote entero a Qt de una vez.
        """
        px = (x1 - x0) / w
        key = (round(x0 / px), round(x1 / px), int(w))
        if key == self._key:
            return
        self._key = key
        self._rects = []
        self._dense = False
        self._cover = 0.0
        if self._n == 0:
            return
        lo = np.searchsorted(self._iv[:, 1], x0 - px, side="left")
        hi = np.searchsorted(self._iv[:, 0], x1 + px, side="right")
        vis = self._iv[lo:hi]
        if vis.size == 0:
            return
        vis, _ = merge_intervals(vis, tol=px)

        # ¿Cuanto del ancho visible tapan? Si es mucho, pintar bandas de altura
        # completa deja el panel rojo y tapa la señal: justo lo que NO quieres.
        # Entonces se pasa a una franja fina abajo. Es la diferencia entre
        # subrayar tres frases de una pagina y pintarla entera de amarillo.
        cover = sum(max(b - a, px) for a, b in vis) / max(1e-12, x1 - x0)
        self._cover = float(min(1.0, cover))
        self._dense = self._cover > GAP_DENSE_COVER
        # coordenadas en Y unitaria (0..1); paint() las estira a la vista
        self._rects = [QtCore.QRectF(a, 0.0, max(b - a, px), 1.0)
                       for a, b in vis]

    def paint(self, p, *args):
        v = self._view()
        if v is None or self._n == 0:
            return
        x0, x1, y0, y1, w = v
        self._rebuild(x0, x1, w)
        if not self._rects:
            return
        p.save()
        if self._dense:
            p.translate(0.0, y0)
            p.scale(1.0, (y1 - y0) * GAP_STRIP_FRAC)   # franja fina abajo
        else:
            # Sin padding: paint() lee el viewRange actual en cada repintado,
            # asi que no hace falta pintar de mas "por si acaso". El padding de
            # 3x que habia antes triplicaba el area a rasterizar para nada.
            p.translate(0.0, y0)
            p.scale(1.0, y1 - y0)
        p.setPen(QtGui.QPen(Qt.NoPen))
        p.setBrush(self._brush)
        p.drawRects(self._rects)                       # UNA llamada, no N
        p.restore()

    def boundingRect(self):
        v = self._view()
        if v is None or self._n == 0:
            return QtCore.QRectF()
        x0, x1, y0, y1, _ = v
        if self._dense:
            return QtCore.QRectF(self._x0, y0, self._x1 - self._x0,
                                 (y1 - y0) * GAP_STRIP_FRAC)
        return QtCore.QRectF(self._x0, y0, self._x1 - self._x0, y1 - y0)


def merge_intervals(iv, tol: float = 0.0, max_out: int | None = None):
    """Fusiona tramos separados por menos de `tol`. Devuelve (lista, fusionados).

    Si se pasa `max_out`, se calcula una `tol` que baje de ese numero. La
    tolerancia normal es el ancho de un pixel: dos huecos que caen en el mismo
    pixel son un solo trazo, dibujarlos por separado no aporta nada.
    """
    a = np.asarray(iv, dtype=np.float64)
    if a.size == 0:
        return [], 0
    if a.ndim == 1:
        a = a.reshape(1, 2)
    a = a[np.argsort(a[:, 0], kind="stable")]
    if max_out is not None:
        if len(a) <= max_out:
            return [tuple(r) for r in a], 0
        tol = max(tol, (a[:, 1].max() - a[:, 0].min()) / max_out)
    if tol <= 0 and len(a) > 1:
        pass
    out, cur = [], [a[0, 0], a[0, 1]]
    for lo, hi in a[1:]:
        if lo - cur[1] <= tol:
            cur[1] = max(cur[1], hi)
        else:
            out.append((cur[0], cur[1]))
            cur = [lo, hi]
    out.append((cur[0], cur[1]))
    return out, len(a) - len(out)


class _DragHandle(QtWidgets.QLabel):
    """Icono de agarre para reordenar arrastrando el panel EN LA PROPIA
    INTERFAZ, sin tener que ir a la lista lateral. Solo esta zona concreta
    inicia el QDrag; el resto del panel sigue respondiendo a los gestos
    normales del ratón (pan, región, marca) sin ninguna interferencia,
    porque el drag-and-drop de Qt es un mecanismo aparte de los eventos de
    ratón que usa pyqtgraph, y solo se activa aquí, al pulsar el icono.
    """

    sigDragStarted = QtCore.Signal()
    sigDragEnded = QtCore.Signal()

    def __init__(self, sid: str, parent=None):
        super().__init__("⠿⠿", parent)
        self.sid = sid
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip("Arrastra para mover este panel")
        self.setStyleSheet("color:#888; font-size:11px; padding:0 4px;")

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            drag = QtGui.QDrag(self)
            mime = QtCore.QMimeData()
            mime.setData(PANEL_MIME, self.sid.encode("utf-8"))
            drag.setMimeData(mime)
            # Pixmap explícito y pequeño (solo el propio icono), no lo que
            # Qt decida por defecto. Sin esto, algunas plataformas generan
            # el feedback visual del arrastre renderizando el widget de
            # origen -- barato aquí porque el icono es diminuto, pero es
            # el tipo de detalle que en otras combinaciones sí importa, y
            # dejarlo explícito quita la incógnita en vez de confiar en el
            # comportamiento por defecto de cada plataforma.
            drag.setPixmap(self.grab())
            drag.setHotSpot(QtCore.QPoint(self.width() // 2, self.height() // 2))
            self.sigDragStarted.emit()
            try:
                drag.exec(Qt.MoveAction)
            finally:
                self.sigDragEnded.emit()
            return
        super().mousePressEvent(ev)


class SeriesPanel(QtWidgets.QFrame):
    sigRegionDrawn = QtCore.Signal(str, float, float)      # sid, t0, t1
    sigGlobalRegionDrawn = QtCore.Signal(float, float)     # t0, t1 (TODAS las series)
    sigMarkDrawn = QtCore.Signal(str, float)               # sid, t
    sigPanStep = QtCore.Signal(int)                        # -1 atrás, +1 adelante
    sigRegionEdited = QtCore.Signal(str, float, float)     # aid, t0, t1
    sigGlobalRegionEdited = QtCore.Signal(str, float, float)  # aid, t0, t1
    sigAnnotationMenu = QtCore.Signal(str, object)         # aid, globalPos
    sigCloseRequested = QtCore.Signal(str)
    sigRegionClicked = QtCore.Signal(str)                  # aid (región o zona global)
    sigCursor = QtCore.Signal(float, float)
    sigReorderDrop = QtCore.Signal(str, str, bool)  # sid_arrastrado, sid_destino, antes
    sigDragStarted = QtCore.Signal()
    sigDragEnded = QtCore.Signal()

    def __init__(self, session, sdef: SeriesDef, parent=None):
        super().__init__(parent)
        self.session = session
        self.sdef = sdef
        self.sid = sdef.sid
        self._region_items: dict[str, pg.LinearRegionItem] = {}
        self._global_region_items: dict[str, pg.LinearRegionItem] = {}
        self._mark_items: dict[str, pg.InfiniteLine] = {}
        self._gap_items: list[pg.LinearRegionItem] = []
        self._gap_overlay: GapOverlay | None = None
        self._stat_items: list[pg.InfiniteLine] = []
        self._overlay_curves: dict[str, pg.PlotDataItem] = {}
        self._draft: pg.LinearRegionItem | None = None
        self._show_gaps = True
        self._show_stat_lines = False
        self._ymode = YMODE_WINDOW
        self._x_init = False
        self._x_axis_on = True

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setMinimumHeight(140)
        self.setAcceptDrops(True)
        self._build_ui()
        self.redraw()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 4)
        lay.setSpacing(2)

        head = QtWidgets.QHBoxLayout()
        self.chip = QtWidgets.QLabel("  ")
        self.chip.setFixedSize(12, 12)
        self.chip.setStyleSheet(
            f"background:{self.sdef.color}; border-radius:3px;")
        self.title = QtWidgets.QLabel(f"<b>{self.sdef.name}</b>")
        self.stats_lbl = QtWidgets.QLabel("")
        self.stats_lbl.setStyleSheet("color:#888; font-family:monospace;")
        self.missing_lbl = QtWidgets.QLabel("")
        self.missing_lbl.setStyleSheet("color:#d04040;")
        btn_close = QtWidgets.QToolButton()
        btn_close.setText("✕")
        btn_close.setToolTip("Ocultar este panel")
        btn_close.clicked.connect(lambda: self.sigCloseRequested.emit(self.sid))

        self._drag_handle = _DragHandle(self.sid)
        self._drag_handle.sigDragStarted.connect(self.sigDragStarted.emit)
        self._drag_handle.sigDragEnded.connect(self.sigDragEnded.emit)
        head.addWidget(self._drag_handle)
        head.addWidget(self.chip)
        head.addWidget(self.title)
        head.addSpacing(8)
        head.addWidget(self.stats_lbl)
        head.addStretch(1)
        head.addWidget(self.missing_lbl)
        head.addWidget(btn_close)
        lay.addLayout(head)

        axis = {}
        if self.session.project.source.x_is_datetime:
            # utcOffset=0 es obligatorio, no cosmético: sin él, DateAxisItem
            # aplica la zona horaria LOCAL DEL SISTEMA a las etiquetas del
            # eje, mientras que fmt_x() (usado en la tabla de anotaciones,
            # las notas y los tooltips) formatea siempre en UTC. Resultado
            # medido: una región que la tabla marca como "termina a las
            # 11:01" aparecía en el eje como "12:01" en una máquina en
            # CET (UTC+1) -- la hora de pared del fichero de origen no
            # cambia, pero cada sitio de la interfaz la mostraba distinta.
            axis["bottom"] = pg.DateAxisItem(orientation="bottom", utcOffset=0)
        self.vb = EditViewBox()
        self.plot = pg.PlotWidget(viewBox=self.vb, axisItems=axis)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setClipToView(True)
        self.plot.setDownsampling(mode="peak", auto=True)
        # Y automatico sobre la ventana visible. Es lo que hace que la señal
        # llene el panel en vez de quedarse aplastada contra su propia media.
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.vb.setAutoVisible(y=True)
        self.vb.setDefaultPadding(Y_PAD)
        lay.addWidget(self.plot, 1)

        # Eje Y derecho para superponer derivadas con escalas distintas.
        # Se crea PEREZOSAMENTE: antes se creaba siempre, en los 18 paneles, y
        # cada uno iba setXLink-ado al principal. Con el eje X sincronizado eso
        # duplica la cascada de propagacion de rango en cada frame de pan, para
        # unos ViewBox que casi nunca tienen nada dentro.
        self.vb2 = None

        self.curve = self.plot.plot([], [], pen=pg.mkPen(self.sdef.color, width=1))

        self.vb.sigDrawRegion.connect(self._on_draw_region)
        self.vb.sigDrawGlobalRegion.connect(self._on_draw_global_region)
        self.vb.sigMark.connect(lambda x: self.sigMarkDrawn.emit(self.sid, x))
        self.vb.sigPanStep.connect(self.sigPanStep.emit)
        self.vb.sigNavClick.connect(self._on_nav_click)
        # Sin throttle, sigMouseMoved dispara a ritmo NATIVO del ratón (puede
        # ir muy por encima de 60 Hz) por cada panel bajo el cursor. Medido:
        # 34 us por llamada -- poco aislado, pero sin límite se acumula, y es
        # justo el tipo de trabajo de fondo que se nota como lag durante
        # cualquier movimiento continuo del ratón, arrastrar incluido. Es la
        # recomendación estándar de pyqtgraph para tracking de posición.
        self._mouse_proxy = pg.SignalProxy(
            self.plot.scene().sigMouseMoved, rateLimit=30,
            slot=self._on_mouse_moved_raw)

        self._stats_timer = QtCore.QTimer(self)
        self._stats_timer.setSingleShot(True)
        self._stats_timer.setInterval(150)          # debounce: sin esto el zoom pega tirones
        self._stats_timer.timeout.connect(self.update_stats)
        self.vb.sigXRangeChanged.connect(lambda *_: self._stats_timer.start())

    def _ensure_vb2(self) -> None:
        """Crea el ViewBox del eje derecho la primera vez que hace falta."""
        if self.vb2 is not None:
            return
        self.vb2 = pg.ViewBox()
        self.plot.plotItem.scene().addItem(self.vb2)
        self.plot.plotItem.getAxis("right").linkToView(self.vb2)
        self.vb2.setXLink(self.plot.plotItem)
        self.plot.plotItem.vb.sigResized.connect(self._sync_vb2)
        self._sync_vb2()

    def _sync_vb2(self) -> None:
        if self.vb2 is None:
            return
        self.vb2.setGeometry(self.plot.plotItem.vb.sceneBoundingRect())
        self.vb2.linkedViewChanged(self.plot.plotItem.vb, self.vb2.XAxis)

    def _on_nav_click(self, x: float) -> None:
        """Click simple en modo Navegar: si cae dentro de una región (propia
        de esta serie, o global), avisa hacia arriba para que la tabla de
        anotaciones seleccione esa fila. Si el click cae dentro de varias
        regiones solapadas, gana la más estrecha -- es la interpretación
        más específica, igual que hacer click en el elemento "de encima"
        en un editor gráfico normal.
        """
        aid = self._region_at(x)
        if aid is not None:
            self.sigRegionClicked.emit(aid)

    def _region_at(self, x: float) -> str | None:
        best_aid, best_width = None, float("inf")
        for items in (self._region_items, self._global_region_items):
            for aid, it in items.items():
                lo, hi = sorted(it.getRegion())
                if lo <= x <= hi and (hi - lo) < best_width:
                    best_aid, best_width = aid, hi - lo
        return best_aid

    def _on_mouse_moved_raw(self, args) -> None:        # SignalProxy entrega los argumentos originales envueltos en una
        # tupla -- es su convención, no un capricho nuestro.
        self._on_mouse_moved(args[0])

    def _on_mouse_moved(self, pos) -> None:
        if self.plot.sceneBoundingRect().contains(pos):
            p = self.vb.mapSceneToView(pos)
            self.sigCursor.emit(p.x(), p.y())

    # ------------------------------------------------------ drag & drop
    # Soltar sobre la mitad superior de este panel lo coloca ANTES;
    # sobre la mitad inferior, DESPUÉS. Es el mismo "arrastro la tercera
    # entre la segunda y la primera" que ya hacía la lista lateral, pero
    # ahora también funciona arrastrando el panel mismo en la interfaz.
    def dragEnterEvent(self, ev) -> None:
        if ev.mimeData().hasFormat(PANEL_MIME):
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev) -> None:
        if ev.mimeData().hasFormat(PANEL_MIME):
            ev.acceptProposedAction()

    def dropEvent(self, ev) -> None:
        if not ev.mimeData().hasFormat(PANEL_MIME):
            return
        dragged_sid = bytes(ev.mimeData().data(PANEL_MIME)).decode("utf-8")
        if dragged_sid == self.sid:
            return
        before = ev.position().y() < self.height() / 2
        self.sigReorderDrop.emit(dragged_sid, self.sid, before)
        ev.acceptProposedAction()

    # ------------------------------------------------------------- dibujo
    def redraw(self) -> None:
        x = self.session.x
        y = self.session.values(self.sid)
        self.curve.setData(x, y, connect="finite")
        self.curve.setPen(pg.mkPen(self.sdef.color, width=1))
        self.chip.setStyleSheet(f"background:{self.sdef.color}; border-radius:3px;")
        self.title.setText(f"<b>{self.sdef.name}</b>"
                           f"<span style='color:#888'> · {self.sdef.describe()}</span>")
        self.redraw_overlays()
        self.redraw_gaps()
        if not self._x_init:
            # Sin esto el panel arranca con el rango X por defecto (0..1), que
            # no contiene datos: la cabecera mostraba "n 1" hasta el primer zoom.
            self._x_init = True
            self.plot.autoRange(padding=Y_PAD)
            self.plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        self.apply_y_mode()
        self.update_stats()

    def redraw_overlays(self) -> None:
        """Derivadas marcadas como 'superponer': van al eje Y derecho, porque
        una derivada y su señal original no comparten rango numérico."""
        wanted = {s.sid: s for s in self.session.project.series
                  if s.parent == self.sid and s.overlay_on_parent and s.visible}
        if wanted:
            self._ensure_vb2()
        for sid, item in list(self._overlay_curves.items()):
            if sid not in wanted:
                if self.vb2 is not None:
                    self.vb2.removeItem(item)
                del self._overlay_curves[sid]
        for sid, sd in wanted.items():
            y = self.session.values(sid)
            if sid in self._overlay_curves:
                self._overlay_curves[sid].setData(self.session.x, y, connect="finite")
                self._overlay_curves[sid].setPen(pg.mkPen(sd.color, width=1,
                                                          style=Qt.DashLine))
            else:
                item = pg.PlotDataItem(self.session.x, y, connect="finite",
                                       pen=pg.mkPen(sd.color, width=1,
                                                    style=Qt.DashLine))
                self.vb2.addItem(item)
                self._overlay_curves[sid] = item
        self.plot.plotItem.showAxis("right", bool(wanted))
        self._sync_vb2()

    def set_x_axis_visible(self, on: bool) -> None:
        """Muestra u oculta el eje X de este panel.

        Con N paneles sincronizados en X, los N dibujan el MISMO eje de fechas
        en cada frame de pan. Medido con 18 paneles: AxisItem.paint era el 27%
        del tiempo de frame, todo el para repetir 18 veces las mismas etiquetas.
        Solo el panel de abajo necesita eje: los demas comparten rango.
        """
        if on == self._x_axis_on:
            return
        self._x_axis_on = on
        self.plot.plotItem.showAxis("bottom", on)

    # --------------------------------------------------------- escala Y
    def set_y_mode(self, mode: str) -> None:
        """window: Y sigue a lo que se ve (defecto).
        full:   Y fija al min/max de toda la serie, para comparar tramos.
        manual: el usuario manda."""
        self._ymode = mode
        self.apply_y_mode()

    def apply_y_mode(self) -> None:
        if self._ymode == YMODE_WINDOW:
            self.vb.setAutoVisible(y=True)
            self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        elif self._ymode == YMODE_FULL:
            self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            lo, hi = self.full_y_range()
            if lo is not None:
                self.vb.setYRange(lo, hi, padding=0)
        else:
            self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)

    def full_y_range(self):
        """min/max de la serie entera, con margen. None si no hay datos."""
        y = self.session.values(self.sid)
        v = y[np.isfinite(y)]
        if v.size == 0:
            return None, None
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:                       # señal constante: dale algo de aire
            d = abs(lo) * 0.05 or 1.0
            return lo - d, hi + d
        pad = (hi - lo) * Y_PAD
        return lo - pad, hi + pad

    def autoscale_y(self) -> None:
        """Vuelve al automatico. Es el boton 'Ajustar Y'."""
        self.set_y_mode(YMODE_WINDOW)
        self.plot.autoRange(padding=Y_PAD)
        self.plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)

    def set_show_gaps(self, on: bool) -> None:
        self._show_gaps = on
        self.redraw_gaps()

    def redraw_gaps(self) -> None:
        for it in self._gap_items:
            self.plot.removeItem(it)
        self._gap_items.clear()
        if self._gap_overlay is not None:
            self.plot.removeItem(self._gap_overlay)
            self._gap_overlay = None

        rep = self.session.missing(self.sid)
        iv = list(rep["time_gaps"]) + list(rep["nan_intervals"])

        bits = []
        if rep["n_nan"]:
            bits.append(f"{rep['n_nan']:,} sin valor")
        if rep["time_gaps"]:
            bits.append(f"{len(rep['time_gaps']):,} saltos "
                        f"(~{rep['muestras_perdidas_estimadas']:,} muestras)")
        self.missing_lbl.setText("\u26a0 " + " \u00b7 ".join(bits) if bits else "")
        if rep.get("x_duplicado"):
            self.missing_lbl.setText(
                self.missing_lbl.text() + "  \u00b7 eje X con valores repetidos")
            self.missing_lbl.setToolTip(
                "El eje X repite valores: hay varias entidades apiladas. La "
                "anchura de los huecos marcados no es fiable en este modo.")
        else:
            self.missing_lbl.setToolTip(
                f"{len(rep['nan_intervals']):,} tramos sin valor.\n"
                "Con poco zoom caen varios por pixel y se dibujan como franja "
                "fina abajo (pintarlos enteros taparia la señal).\n"
                "Haz zoom y se separan en huecos individuales.")

        if not self._show_gaps or not iv:
            return
        if len(iv) <= MAX_GAP_ITEMS:
            for t0, t1 in iv:
                it = pg.LinearRegionItem(values=(t0, t1), brush=GAP_BRUSH,
                                         pen=GAP_PEN, movable=False)
                it.setZValue(-100)
                self.plot.addItem(it)
                self._gap_items.append(it)
        else:
            ov = GapOverlay(iv)
            ov.setZValue(-100)
            self.plot.addItem(ov)
            self._gap_overlay = ov

    def set_show_stat_lines(self, on: bool) -> None:
        self._show_stat_lines = on
        self.update_stats()

    def update_stats(self) -> None:
        (x0, x1), _ = self.vb.viewRange()
        s = self.session.stats(self.sid, x0, x1)
        self.stats_lbl.setText(fmt_stats(s))
        for it in self._stat_items:
            self.plot.removeItem(it)
        self._stat_items.clear()
        if not self._show_stat_lines or not np.isfinite(s.get("mean", np.nan)):
            return
        specs = [(s["mean"], "#ffffff", Qt.SolidLine, "μ"),
                 (s["min"], "#66bb6a", Qt.DotLine, "min"),
                 (s["max"], "#ef5350", Qt.DotLine, "max")]
        if np.isfinite(s["std"]) and s["std"] > 0:
            specs += [(s["mean"] + s["std"], "#90a4ae", Qt.DashLine, "+σ"),
                      (s["mean"] - s["std"], "#90a4ae", Qt.DashLine, "−σ")]
        for val, color, style, lbl in specs:
            line = pg.InfiniteLine(pos=val, angle=0, movable=False,
                                   pen=pg.mkPen(color, width=1, style=style),
                                   label=lbl, labelOpts={"position": 0.02,
                                                         "color": color})
            line.setZValue(-50)
            self.plot.addItem(line)
            self._stat_items.append(line)

    # -------------------------------------------------------- anotaciones
    def set_mode(self, mode: str) -> None:
        self.vb.set_mode(mode)
        for it in self._region_items.values():
            it.setMovable(mode != MODE_NAV)
        for it in self._global_region_items.values():
            it.setMovable(mode != MODE_NAV)

    def _on_draw_region(self, x0: float, x1: float, finished: bool) -> None:
        if self._draft is None:
            self._draft = pg.LinearRegionItem(values=(x0, x1),
                                              brush=pg.mkBrush(255, 213, 79, 70))
            self._draft.setZValue(10)
            self.plot.addItem(self._draft)
        self._draft.setRegion((x0, x1))
        if finished:
            self.plot.removeItem(self._draft)
            self._draft = None
            if abs(x1 - x0) > 0:
                self.sigRegionDrawn.emit(self.sid, x0, x1)

    def _on_draw_global_region(self, x0: float, x1: float, finished: bool) -> None:
        """Igual que _on_draw_region, pero el borrador solo se ve en ESTE
        panel mientras arrastras (feedback inmediato); al soltar, se avisa
        hacia arriba sin sid -- mainwindow la crea una vez y la reparte a
        TODOS los paneles via sync_global_regions()."""
        if self._draft is None:
            self._draft = pg.LinearRegionItem(values=(x0, x1),
                                              brush=pg.mkBrush(124, 77, 255, 70),
                                              pen=pg.mkPen("#7C4DFF", width=1))
            self._draft.setZValue(10)
            self.plot.addItem(self._draft)
        self._draft.setRegion((x0, x1))
        if finished:
            self.plot.removeItem(self._draft)
            self._draft = None
            if abs(x1 - x0) > 0:
                self.sigGlobalRegionDrawn.emit(x0, x1)

    def sync_annotations(self, regions: list[Region], marks: list[Mark],
                         movable: bool) -> None:
        keep_r = {r.aid for r in regions}
        for aid in list(self._region_items):
            if aid not in keep_r:
                self.plot.removeItem(self._region_items.pop(aid))
        for r in regions:
            it = self._region_items.get(r.aid)
            if it is None:
                it = pg.LinearRegionItem(values=(r.t0, r.t1),
                                         brush=pg.mkBrush(QtGui.QColor(r.color).red(),
                                                          QtGui.QColor(r.color).green(),
                                                          QtGui.QColor(r.color).blue(), 70))
                it.setZValue(10)
                it.aid = r.aid
                it.sigRegionChangeFinished.connect(
                    lambda item=it: self.sigRegionEdited.emit(
                        item.aid, *sorted(item.getRegion())))
                self.plot.addItem(it)
                self._region_items[r.aid] = it
            it.setMovable(movable)
            if tuple(sorted(it.getRegion())) != (r.t0, r.t1):
                it.setRegion((r.t0, r.t1))

        keep_m = {m.aid for m in marks}
        for aid in list(self._mark_items):
            if aid not in keep_m:
                self.plot.removeItem(self._mark_items.pop(aid))
        for m in marks:
            if m.aid not in self._mark_items:
                line = pg.InfiniteLine(pos=m.t, angle=90, movable=False,
                                       pen=pg.mkPen(m.color, width=2),
                                       label=m.label or "",
                                       labelOpts={"position": 0.92,
                                                  "color": m.color})
                line.setZValue(20)
                self.plot.addItem(line)
                self._mark_items[m.aid] = line
            else:
                self._mark_items[m.aid].setPos(m.t)

    def sync_global_regions(self, regions: list[GlobalRegion], movable: bool) -> None:
        """Zonas globales: se pintan en TODOS los paneles a la vez, con Z
        más bajo que las regiones/marcas por serie (son contexto de fondo,
        no deben tapar una anotación puesta a propósito sobre una señal
        concreta)."""
        keep = {r.aid for r in regions}
        for aid in list(self._global_region_items):
            if aid not in keep:
                self.plot.removeItem(self._global_region_items.pop(aid))
        for r in regions:
            it = self._global_region_items.get(r.aid)
            if it is None:
                c = QtGui.QColor(r.color)
                it = pg.LinearRegionItem(
                    values=(r.t0, r.t1),
                    brush=pg.mkBrush(c.red(), c.green(), c.blue(), 45),
                    pen=pg.mkPen(r.color, width=1, style=Qt.DashLine))
                it.setZValue(5)
                it.aid = r.aid
                it.sigRegionChangeFinished.connect(
                    lambda item=it: self.sigGlobalRegionEdited.emit(
                        item.aid, *sorted(item.getRegion())))
                self.plot.addItem(it)
                self._global_region_items[r.aid] = it
            it.setMovable(movable)
            if tuple(sorted(it.getRegion())) != (r.t0, r.t1):
                it.setRegion((r.t0, r.t1))
