"""Estado en memoria: proyecto + tabla + caché de arrays calculados. Sin Qt."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import gaps as gapmod
from . import loader, longformat, store, transforms
from .model import KIND_RAW, Project, SeriesDef, SourceInfo, new_id


class Session:
    def __init__(self) -> None:
        self.project = Project()
        self.df: Optional[pd.DataFrame] = None
        self.x: np.ndarray = np.empty(0)
        self.data_path: Optional[Path] = None
        self.json_path: Optional[Path] = None
        self._cache: dict[str, np.ndarray] = {}
        self._gap_cache: dict[str, dict] = {}
        self.dirty = False
        self.warnings: list[str] = []
        self.long = None

    # ------------------------------------------------------------------
    def open(self, data_path: str | Path, x_mode: str, x_column: Optional[str],
             max_samples: Optional[int] = None, sample_policy: str = "truncate",
             selected_columns: Optional[list[str]] = None,
             float32: bool = True, progress=None,
             long_mode: str = "raw", group_column: Optional[str] = None,
             group_value=None) -> None:
        """Abre el fichero. Los filtros se empujan al lector: si pides 3 de 50
        columnas y 200k de 3M filas, se leen 3 columnas y 200k filas."""
        p = Path(data_path)
        self.warnings.clear()          # antes de nada: luego se van llenando

        # Columnas a pedir al lector: las elegidas + la del eje X.
        want = None
        if selected_columns:
            want = list(dict.fromkeys(
                list(selected_columns) + ([x_column] if x_column else [])))

        nrows, step, total = loader.plan_sampling(p, max_samples, sample_policy)
        df = loader.read_table(p, columns=want, nrows=nrows, decimate_step=step,
                               float32=float32, progress=progress)

        # Formato largo: varias entidades apiladas sobre el mismo eje X.
        self.long = longformat.detect(df, x_column)
        if long_mode != "raw" and group_column and group_column in df.columns:
            if progress:
                progress(f"Separando por «{group_column}»...", 0.62)
            if long_mode == "filter" and group_value is not None:
                df = longformat.filter_group(df, group_column, group_value)
                self.warnings.append(
                    f"Cargada solo la entidad «{group_value}» de "
                    f"«{group_column}». Las demás no están en memoria.")
            elif long_mode == "pivot":
                df = longformat.pivot_wide(df, x_column, group_column)
                self.warnings.append(
                    f"Fichero pivotado: una columna por (señal × «{group_column}»). "
                    f"{len(df.columns)-1} series sobre un eje X sin duplicados.")
        elif self.long.is_long and long_mode == "raw":
            self.warnings.append(
                "AVISO: " + self.long.message +
                "\n\nLo has cargado en crudo. Las estadísticas y la detección "
                "de huecos mezclarán las entidades.")

        if progress:
            progress("Construyendo eje X...", 0.65)
        x, is_dt = loader.build_x(df, x_mode, x_column)
        if progress:
            progress("Ordenando y saneando...", 0.8)
        x, df = loader.sanitize_axis(x, df)
        if progress:
            progress("Listo", 0.95)

        self.df, self.x, self.data_path = df, x, p
        self.json_path = store.sidecar_path(p)
        self._cache.clear()
        self._gap_cache.clear()

        src = SourceInfo(
            path=str(p), quick_hash=loader.quick_hash(p), size=p.stat().st_size,
            mtime=p.stat().st_mtime, x_mode=x_mode, x_column=x_column,
            x_is_datetime=is_dt, max_samples=max_samples, sample_policy=sample_policy,
            long_mode=long_mode, group_column=group_column,
            group_value=None if group_value is None else str(group_value),
        )

        existing = None
        if self.json_path.exists() and store.is_ours(self.json_path):
            try:
                existing = store.load(self.json_path)
            except Exception as e:
                self.warnings.append(f"El JSON existente no se pudo leer: {e}")

        if existing is not None:
            msg = store.check_source(existing, p)
            if msg:
                self.warnings.append(msg)
            self.project = existing
            self.project.source = src
            try:
                file_cols = set(loader.peek_columns(p, n=5).columns)
            except Exception:
                file_cols = set(df.columns)   # degradación razonable, no un crash
            self._prune_missing_columns(file_cols)
        else:
            self.project = Project(source=src)
            cols = [c for c in (selected_columns or loader.numeric_columns(df))
                    if c in df.columns]
            for c in cols:
                if c == x_column:
                    continue
                self.project.add_series(SeriesDef(sid=new_id("s"), name=c,
                                                  kind=KIND_RAW, column=c))
        self.project.renumber()
        self.dirty = False

    def _prune_missing_columns(self, file_columns: set[str]) -> None:
        """Elimina del proyecto solo las series cuya columna ya NO EXISTE EN
        EL FICHERO. Antes se comparaba contra self.df.columns, que es la vista
        ya filtrada por 'columnas a cargar' de ESTA apertura concreta: si abrías
        pidiendo menos columnas que la vez anterior, el resto se borraba del
        proyecto para siempre, con el mensaje falso de que "ya no está en el
        fichero" cuando sí estaba. Se guarda así, y en la siguiente apertura
        (aunque pidieras todas las columnas) ya no volvían: el borrado era
        permanente porque quedaba escrito en el sidecar.

        Ahora la única fuente de verdad es la cabecera real del CSV/parquet,
        no lo que se haya pedido cargar esta vez. Si una columna no se cargó
        porque no estaba marcada, la serie se conserva oculta, no se destruye.
        """
        assert self.df is not None
        missing = [s for s in self.project.series
                   if s.kind == KIND_RAW and s.column not in file_columns]
        for s in missing:
            self.warnings.append(f"La columna '{s.column}' ya no está en el "
                                 f"fichero de origen; se elimina la serie "
                                 f"'{s.name}' y sus anotaciones.")
            self.project.remove_series(s.sid)

        # Si una columna que se guardó como "oculta por no cargarse" vuelve a
        # estar en el df de esta apertura, se reactiva. La marca de "por qué
        # está oculta" vive en el propio SeriesDef (hidden_reason), no en
        # memoria de la sesión: así sobrevive a cerrar y volver a abrir, que
        # es exactamente cuando hace falta.
        for s in self.project.series:
            if (s.kind == KIND_RAW and s.column in self.df.columns
                    and s.hidden_reason == "not_loaded"):
                s.visible = True
                s.hidden_reason = ""

        not_loaded = [s for s in self.project.series
                     if s.kind == KIND_RAW and s.column in file_columns
                     and s.column not in self.df.columns]
        for s in not_loaded:
            s.visible = False
            s.hidden_reason = "not_loaded"
        if not_loaded:
            names = ", ".join(f"«{s.name}»" for s in not_loaded[:6])
            more = "" if len(not_loaded) <= 6 else f" y {len(not_loaded)-6} más"
            self.warnings.append(
                f"No se han cargado esta vez (no estaban marcadas): "
                f"{names}{more}. Siguen en el proyecto, ocultas.")

    # ------------------------------------------------------------------
    def values(self, sid: str) -> np.ndarray:
        """Array de la serie. Las derivadas se recalculan desde la receta."""
        if sid in self._cache:
            return self._cache[sid]
        s = self.project.by_id(sid)
        if s is None or self.df is None:
            return np.empty(0)
        if s.kind == KIND_RAW:
            if s.column not in self.df.columns:
                # Serie que existe en el proyecto pero no se cargó esta vez
                # (no estaba marcada en el diálogo). No es un error: se
                # devuelve NaN en vez de reventar con KeyError, y el panel
                # puede avisar "no cargada" en lugar de desaparecer.
                y = np.full(len(self.x), np.nan, dtype=np.float32)
                self._cache[sid] = y
                return y
            col = self.df[s.column]
            if pd.api.types.is_numeric_dtype(col):
                y = col.to_numpy()                     # sin copia si ya es float
                if y.dtype not in (np.float32, np.float64):
                    y = y.astype(np.float32, copy=False)
            else:
                y = pd.to_numeric(col, errors="coerce").to_numpy(np.float32)
        else:
            parent = self.values(s.parent)
            y = transforms.apply_recipe(s.kind, s.params, self.x, parent)
            if y.dtype == np.float64 and parent.dtype == np.float32:
                y = y.astype(np.float32, copy=False)
        self._cache[sid] = y
        return y

    def memory_mb(self) -> float:
        """RAM aproximada del fichero cargado + caches. Para el aviso del
        dialogo: 3M filas x 50 columnas en float64 son 1.2 GB solo de datos."""
        m = 0.0
        if self.df is not None:
            m += float(self.df.memory_usage(deep=False).sum())
        m += float(self.x.nbytes)
        m += sum(float(a.nbytes) for a in self._cache.values())
        return m / 1e6

    def invalidate(self, sid: str) -> None:
        """Invalida la serie y su descendencia."""
        stack = [sid]
        while stack:
            cur = stack.pop()
            self._cache.pop(cur, None)
            self._gap_cache.pop(cur, None)
            stack.extend(c.sid for c in self.project.children_of(cur))

    def missing(self, sid: str, factor: float = 1.8) -> dict:
        if sid not in self._gap_cache:
            self._gap_cache[sid] = gapmod.missing_report(self.x, self.values(sid), factor)
        return self._gap_cache[sid]

    def stats(self, sid: str, x0=None, x1=None) -> dict:
        return transforms.window_stats(self.x, self.values(sid), x0, x1)

    # ------------------------------------------------------------------
    def available_columns(self) -> list[str]:
        """Columnas numéricas del fichero ya cargado en memoria que no están
        siendo usadas por ninguna serie de origen: candidatas para recuperar
        con add_raw_series() tras un "Eliminar", sin volver a leer el fichero
        (el CSV completo ya vive en self.df desde la apertura)."""
        if self.df is None:
            return []
        used = {s.column for s in self.project.series if s.kind == KIND_RAW}
        xcol = self.project.source.x_column
        return [c for c in loader.numeric_columns(self.df)
                if c not in used and c != xcol]

    def add_raw_series(self, column: str) -> SeriesDef:
        """Vuelve a dar de alta una columna del CSV como serie de origen,
        p.ej. tras borrarla con "Eliminar". Los datos ya están en self.df:
        no hace falta releer el fichero."""
        s = SeriesDef(sid=new_id("s"), name=column, kind=KIND_RAW, column=column)
        self.project.add_series(s)
        self.dirty = True
        return s

    # ------------------------------------------------------------------
    def add_derived(self, parent_sid: str, kind: str, params: dict,
                    overlay: bool = False) -> SeriesDef:
        parent = self.project.by_id(parent_sid)
        name = f"{parent.name} · {kind}"
        if "window" in params:
            name += f"({params['window']})"
        s = SeriesDef(sid=new_id("s"), name=name, kind=kind, parent=parent_sid,
                      params=dict(params), overlay_on_parent=overlay)
        self.project.add_series(s)
        self.dirty = True
        return s

    def save(self) -> Optional[Path]:
        if self.json_path is None:
            return None
        p = store.save(self.project, self.json_path)
        self.dirty = False
        return p
