"""Modelo de datos. Sin dependencias de Qt: se puede testear en headless."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

SCHEMA_VERSION = 1

# Tipos de serie derivada. La receta se guarda, NUNCA los datos.
KIND_RAW = "raw"
KIND_ROLLING_MEAN = "rolling_mean"
KIND_ROLLING_STD = "rolling_std"
KIND_DERIVATIVE = "derivative"
KIND_LAG = "lag"
DERIVED_KINDS = (KIND_ROLLING_MEAN, KIND_ROLLING_STD, KIND_DERIVATIVE, KIND_LAG)

PALETTE = [
    "#4C9AFF", "#F5A623", "#7ED321", "#D0021B", "#BD10E0",
    "#50E3C2", "#B8E986", "#9013FE", "#F8E71C", "#FF6F61",
]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class SeriesDef:
    """Una serie: o una columna del fichero, o una receta sobre otra serie."""
    sid: str
    name: str
    kind: str = KIND_RAW
    column: Optional[str] = None      # solo kind == raw
    parent: Optional[str] = None      # solo derivadas
    params: dict = field(default_factory=dict)
    visible: bool = True
    color: Optional[str] = None
    order: int = 0
    overlay_on_parent: bool = False   # dibujar en el panel del padre, eje Y derecho
    show_stats_lines: bool = False
    hidden_reason: str = ""           # "" | "not_loaded" (ver session._prune_missing_columns)

    @property
    def is_derived(self) -> bool:
        return self.kind in DERIVED_KINDS

    def describe(self) -> str:
        if self.kind == KIND_ROLLING_MEAN:
            return f"media móvil w={self.params.get('window')}"
        if self.kind == KIND_ROLLING_STD:
            return f"desv. móvil w={self.params.get('window')}"
        if self.kind == KIND_DERIVATIVE:
            return "derivada d/dx"
        if self.kind == KIND_LAG:
            return f"desplazada {self.params.get('lag')} muestras"
        return "original"


@dataclass
class Region:
    aid: str
    sid: str
    t0: float
    t1: float
    label: str = ""
    color: str = "#FFD54F"

    def normalized(self) -> "Region":
        if self.t0 > self.t1:
            self.t0, self.t1 = self.t1, self.t0
        return self


@dataclass
class GlobalRegion:
    """Una franja que abarca TODO el dataset, dibujada en TODOS los paneles
    a la vez -- "zona de arranque", "parada programada"... No pertenece a
    una señal concreta, así que no vive dentro de Region (que siempre está
    atada a un sid): es una capa aparte que se ve en cada panel además de
    sus propias regiones/marcas por serie.

    exclude_from_stats: si está marcada, la ventana de Análisis quita estos
    instantes de TODOS los cálculos (histograma, boxplot, heatmap, matriz,
    ACF/CCF) sin tener que restringir la sección a mano cada vez. Pensado
    para tramos que no son un régimen de operación real -- arranque de
    máquina, parada, purga -- y que de otro modo contaminan la estadística
    del resto de la serie.
    """
    aid: str
    t0: float
    t1: float
    label: str = ""
    color: str = "#7C4DFF"
    exclude_from_stats: bool = False

    def normalized(self) -> "GlobalRegion":
        if self.t0 > self.t1:
            self.t0, self.t1 = self.t1, self.t0
        return self


@dataclass
class Mark:
    aid: str
    sid: str
    t: float
    label: str = ""
    color: str = "#FF5252"


@dataclass
class GlobalMark:
    """Un instante puntual marcado en TODOS los paneles a la vez -- el
    equivalente de GlobalRegion pero para un punto en vez de un tramo:
    "cambio de turno", "parada de emergencia"... No pertenece a una señal
    concreta, igual que GlobalRegion frente a Region.
    """
    aid: str
    t: float
    label: str = ""
    color: str = "#FF5252"


@dataclass
class Note:
    """Un apunte de texto libre, sin necesidad de marcar nada en el gráfico.

    Frente a Region/Mark, que exigen un gesto sobre el canvas (arrastrar o
    hacer click) para fijar un instante concreto, una Note se puede escribir
    de un tirón mientras exploras -- "esto huele a fallo de sensor",
    "revisar con el maestro" -- sin interrumpir lo que estás mirando para ir
    a dibujar algo. El contexto (qué series se veían y en qué rango) se
    captura solo, como ayuda para releerla luego; no hace falta señalarlo.
    """
    nid: str
    text: str
    created_at: float                       # reloj de pared (epoch, segundos)
    x0: Optional[float] = None              # rango visible al escribirla
    x1: Optional[float] = None
    series: list[str] = field(default_factory=list)  # sids visibles entonces


@dataclass
class SourceInfo:
    """Identidad del fichero de datos. Si esto no cuadra al abrir, avisamos."""
    path: str = ""
    quick_hash: str = ""
    size: int = 0
    mtime: float = 0.0
    x_mode: str = "index"           # "column" | "index"
    x_column: Optional[str] = None
    x_is_datetime: bool = False
    max_samples: Optional[int] = None
    sample_policy: str = "truncate"  # "truncate" | "decimate"
    long_mode: str = "raw"           # "raw" | "filter" | "pivot"
    group_column: Optional[str] = None
    group_value: Optional[str] = None


@dataclass
class Group:
    """Un grupo de series para mostrarlas u ocultarlas todas de golpe.

    No es jerarquía como parent/derivada (que siempre relaciona una serie
    con SU derivada): un grupo puede juntar series de origen distinto que
    no tienen ningún parentesco -- "todas las de temperatura", "línea A
    completa". Es puro atajo de visibilidad, nada más.
    """
    gid: str
    name: str
    members: list[str] = field(default_factory=list)   # sids


@dataclass
class Project:
    source: SourceInfo = field(default_factory=SourceInfo)
    series: list[SeriesDef] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)
    global_regions: list[GlobalRegion] = field(default_factory=list)
    global_marks: list[GlobalMark] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    groups: list[Group] = field(default_factory=list)
    view: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    # --- consultas -------------------------------------------------------
    def by_id(self, sid: str) -> Optional[SeriesDef]:
        return next((s for s in self.series if s.sid == sid), None)

    def by_name(self, name: str) -> Optional[SeriesDef]:
        return next((s for s in self.series if s.name == name), None)

    def children_of(self, sid: str) -> list[SeriesDef]:
        return [s for s in self.series if s.parent == sid]

    def ordered(self) -> list[SeriesDef]:
        return sorted(self.series, key=lambda s: s.order)

    def excluded_intervals(self) -> list[tuple[float, float]]:
        """Tramos (t0, t1) marcados como zona a excluir de las estadísticas."""
        return [(g.t0, g.t1) for g in self.global_regions if g.exclude_from_stats]

    def root_of(self, sid: str) -> SeriesDef:
        s = self.by_id(sid)
        seen = set()
        while s and s.parent and s.parent not in seen:
            seen.add(s.sid)
            nxt = self.by_id(s.parent)
            if nxt is None:
                break
            s = nxt
        return s

    # --- mutaciones ------------------------------------------------------
    def add_series(self, s: SeriesDef) -> SeriesDef:
        if s.color is None:
            s.color = PALETTE[len(self.series) % len(PALETTE)]
        if not s.order:
            if s.parent is not None:
                s.order = self._order_under(s.parent)
            else:
                s.order = (max((x.order for x in self.series), default=-1)) + 1
        self.series.append(s)
        return s

    def _order_under(self, parent_sid: str) -> int:
        """Hueco de orden justo DEBAJO del padre y de las derivadas que ya
        cuelguen de él.

        Antes toda serie nueva iba al final (`max(order) + 1`), así que una
        derivada que no se superpone sobre la original aparecía como último
        panel de la ventana, lejísimos de la señal de la que sale: para
        compararlas había que arrastrar el panel a mano cada vez. Lo natural
        es que salga pegada debajo de su original, que es donde la vas a
        mirar.

        Se desplaza el orden de todo lo que venga después para abrir hueco.
        Es O(n) sobre una lista de series, que son decenas, no millones.
        """
        parent = self.by_id(parent_sid)
        if parent is None:
            return (max((x.order for x in self.series), default=-1)) + 1
        # El final del bloque del padre: él y toda su descendencia ya creada.
        block = {parent.sid}
        changed = True
        while changed:
            changed = False
            for x in self.series:
                if x.parent in block and x.sid not in block:
                    block.add(x.sid)
                    changed = True
        slot = max(x.order for x in self.series if x.sid in block) + 1
        for x in self.series:
            if x.order >= slot:
                x.order += 1
        return slot

    def remove_series(self, sid: str) -> list[str]:
        """Borra la serie y toda su descendencia. Devuelve los ids borrados."""
        doomed, stack = [], [sid]
        while stack:
            cur = stack.pop()
            doomed.append(cur)
            stack.extend(c.sid for c in self.children_of(cur))
        self.series = [s for s in self.series if s.sid not in doomed]
        self.regions = [r for r in self.regions if r.sid not in doomed]
        self.marks = [m for m in self.marks if m.sid not in doomed]
        # Una nota puede referenciar varias series a la vez (es contexto, no
        # pertenencia): si una desaparece, se limpia de la lista en vez de
        # borrar la nota entera -- el texto sigue siendo válido.
        for n in self.notes:
            n.series = [sid for sid in n.series if sid not in doomed]
        # Un grupo sin miembros no tiene ningún propósito -- se limpia solo,
        # a diferencia de una nota (cuyo texto sigue teniendo sentido aunque
        # pierda todas sus referencias).
        for g in self.groups:
            g.members = [sid for sid in g.members if sid not in doomed]
        self.groups = [g for g in self.groups if g.members]
        self.renumber()
        return doomed

    def reorder(self, sids_in_order: list[str]) -> None:
        pos = {sid: i for i, sid in enumerate(sids_in_order)}
        for s in self.series:
            if s.sid in pos:
                s.order = pos[s.sid]
        self.renumber()

    def renumber(self) -> None:
        for i, s in enumerate(self.ordered()):
            s.order = i

    # --- serialización ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source": asdict(self.source),
            "series": [asdict(s) for s in self.series],
            "regions": [asdict(r) for r in self.regions],
            "marks": [asdict(m) for m in self.marks],
            "global_regions": [asdict(g) for g in self.global_regions],
            "global_marks": [asdict(m) for m in self.global_marks],
            "notes": [asdict(n) for n in self.notes],
            "groups": [asdict(g) for g in self.groups],
            "view": self.view,
        }

    @staticmethod
    def from_dict(d: dict) -> "Project":
        ver = d.get("schema_version", 0)
        if ver > SCHEMA_VERSION:
            raise ValueError(
                f"El JSON usa schema_version={ver} y esta versión entiende "
                f"hasta {SCHEMA_VERSION}. Actualiza la herramienta."
            )
        def pick(cls, raw):
            allowed = cls.__dataclass_fields__.keys()
            return cls(**{k: v for k, v in raw.items() if k in allowed})

        return Project(
            source=pick(SourceInfo, d.get("source", {})),
            series=[pick(SeriesDef, s) for s in d.get("series", [])],
            regions=[pick(Region, r) for r in d.get("regions", [])],
            marks=[pick(Mark, m) for m in d.get("marks", [])],
            global_regions=[pick(GlobalRegion, g) for g in d.get("global_regions", [])],
            global_marks=[pick(GlobalMark, m) for m in d.get("global_marks", [])],
            notes=[pick(Note, n) for n in d.get("notes", [])],
            groups=[pick(Group, g) for g in d.get("groups", [])],
            view=d.get("view", {}),
            schema_version=SCHEMA_VERSION,
        )
