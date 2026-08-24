"""Lectura de ficheros y construcción del eje X. Sin Qt.

PRINCIPIO: no leer lo que no se va a usar. Todo filtro (columnas, número de
muestras) se empuja al lector, no se aplica después. Leer 3 millones de filas
para quedarte con 200.000 es tirar tiempo y RAM a la basura.
"""
from __future__ import annotations

import csv as _csv
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

CSV_EXT = {".csv", ".txt", ".tsv"}
PARQUET_EXT = {".parquet", ".pq"}

Progress = Optional[Callable[[str, float], None]]   # (mensaje, 0..1)


class LoadError(RuntimeError):
    pass


def _tick(cb: Progress, msg: str, frac: float) -> None:
    if cb is not None:
        cb(msg, frac)


# ----------------------------------------------------------------------
# delimitador
# ----------------------------------------------------------------------
def sniff_sep(path: Path) -> str:
    """Detecta el separador leyendo SOLO la cabecera.

    Antes esto se delegaba en pandas pasando sep=None, lo que activa el parser
    de Python: 6.5x mas lento y 2.6x mas RAM que el parser en C. En un fichero
    de 400 MB eso es la diferencia entre 8 segundos y quedarse sin memoria.
    """
    head = ""
    if path.suffix.lower() == ".tsv":
        return "\t"
    try:
        with open(path, "r", newline="", errors="replace") as f:
            head = f.read(64 * 1024)
        if not head:
            return ","
        return _csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except Exception:
        first = head.splitlines()[0] if head else ""
        return max(",;\t|", key=first.count) if first else ","


# ----------------------------------------------------------------------
# inspeccion barata
# ----------------------------------------------------------------------
def peek_columns(path: str | Path, n: int = 500) -> pd.DataFrame:
    """Muestra para el dialogo de carga. Nunca toca el fichero entero."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in CSV_EXT:
        return pd.read_csv(p, nrows=n, sep=sniff_sep(p))
    if ext in PARQUET_EXT:
        import pyarrow.parquet as pq
        f = pq.ParquetFile(p)
        try:
            return next(f.iter_batches(batch_size=n)).to_pandas()
        except StopIteration:
            return f.schema_arrow.empty_table().to_pandas()
    raise LoadError(f"Extension no soportada: {ext}")


def count_rows(path: str | Path) -> Optional[int]:
    """Filas totales sin parsear. Parquet es metadato (instantaneo);
    CSV es contar saltos de linea en binario."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in PARQUET_EXT:
        import pyarrow.parquet as pq
        return pq.ParquetFile(p).metadata.num_rows
    if ext in CSV_EXT:
        n = 0
        with open(p, "rb") as f:
            while True:
                chunk = f.read(8 << 20)
                if not chunk:
                    break
                n += chunk.count(b"\n")
        return max(0, n - 1)
    return None


def estimate_rows(path: str | Path) -> Optional[int]:
    """Estimacion instantanea por tamano medio de linea, para avisar en el
    dialogo sin bloquear: contar lineas de 4 GB tambien cuesta."""
    p = Path(path)
    if p.suffix.lower() in PARQUET_EXT:
        return count_rows(p)
    size = p.stat().st_size
    with open(p, "rb") as f:
        sample = f.read(1 << 20)
    lines = sample.count(b"\n")
    if lines < 2:
        return None
    return int(size / (len(sample) / lines))


# ----------------------------------------------------------------------
# lectura con filtros empujados al lector
# ----------------------------------------------------------------------
def read_table(path: str | Path,
               columns: Optional[list[str]] = None,
               nrows: Optional[int] = None,
               decimate_step: int = 1,
               float32: bool = False,
               progress: Progress = None) -> pd.DataFrame:
    """Lee aplicando los filtros DURANTE la lectura.

    columns        columnas a cargar (incluye ya la del eje X). None = todas.
    nrows          leer solo las primeras N filas del fichero.
    decimate_step  quedarse con 1 de cada N filas sin materializar el resto.
    float32        mitad de RAM. Para dibujar y para estadisticas sobra; si tus
                   valores pasan de ~1e7 con decimales significativos, no.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in CSV_EXT:
        return _read_csv(p, columns, nrows, decimate_step, float32, progress)
    if ext in PARQUET_EXT:
        return _read_parquet(p, columns, nrows, decimate_step, float32, progress)
    raise LoadError(f"Extension no soportada: {ext}")


def _downcast(df: pd.DataFrame, on: bool) -> pd.DataFrame:
    if not on:
        return df
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]) and df[c].dtype != np.float32:
            df[c] = df[c].astype(np.float32)
    return df


def _read_csv(p, columns, nrows, step, float32, cb) -> pd.DataFrame:
    sep = sniff_sep(p)
    kw = dict(sep=sep, engine="c")
    if columns:
        kw["usecols"] = columns

    if step <= 1:
        _tick(cb, "Leyendo CSV...", 0.05)
        df = pd.read_csv(p, nrows=nrows, **kw)
        _tick(cb, "Leido", 0.6)
        return _downcast(df.reset_index(drop=True), float32)

    total = nrows or estimate_rows(p) or 0
    chunks, seen = [], 0
    _tick(cb, "Leyendo y decimando...", 0.05)
    for ch in pd.read_csv(p, chunksize=1_000_000, **kw):
        take = ch.iloc[(step - seen % step) % step::step]
        chunks.append(_downcast(take, float32))
        seen += len(ch)
        if total:
            _tick(cb, f"Leyendo... {seen:,}/{total:,} filas",
                  0.05 + 0.55 * min(1.0, seen / total))
        if nrows and seen >= nrows:
            break
    if not chunks:
        return pd.read_csv(p, nrows=0, **kw)
    return pd.concat(chunks, ignore_index=True)


def _read_parquet(p, columns, nrows, step, float32, cb) -> pd.DataFrame:
    import pyarrow.parquet as pq
    f = pq.ParquetFile(p)
    if columns:
        have = set(f.schema_arrow.names)
        columns = [c for c in columns if c in have]

    if step <= 1 and nrows is None:
        _tick(cb, "Leyendo parquet...", 0.05)
        df = f.read(columns=columns).to_pandas()
        _tick(cb, "Leido", 0.6)
        return _downcast(df, float32)

    total = f.metadata.num_rows
    chunks, seen = [], 0
    _tick(cb, "Leyendo parquet...", 0.05)
    for batch in f.iter_batches(batch_size=500_000, columns=columns):
        ch = batch.to_pandas()
        take = ch.iloc[(step - seen % step) % step::step] if step > 1 else ch
        chunks.append(_downcast(take, float32))
        seen += len(ch)
        _tick(cb, f"Leyendo... {seen:,}/{total:,} filas",
              0.05 + 0.55 * min(1.0, seen / max(1, total)))
        if nrows and seen >= nrows:
            break
    if not chunks:
        return f.schema_arrow.empty_table().to_pandas()
    df = pd.concat(chunks, ignore_index=True)
    return df.iloc[:nrows].reset_index(drop=True) if (nrows and step <= 1) else df


def plan_sampling(path: str | Path, max_samples: Optional[int],
                  policy: str) -> tuple[Optional[int], int, Optional[int]]:
    """Traduce el limite de muestras a argumentos del lector.

    Devuelve (nrows, decimate_step, total_estimado). Antes esto se aplicaba
    DESPUES de leer, o sea que no ahorraba absolutamente nada.
    """
    total = estimate_rows(path)
    if not max_samples or (total and total <= max_samples):
        return None, 1, total
    if policy == "decimate":
        if not total:
            return None, 1, None
        return None, max(1, int(np.ceil(total / max_samples))), total
    return int(max_samples), 1, total


# ----------------------------------------------------------------------
# eje X
# ----------------------------------------------------------------------
def is_datetime_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    sample = s.dropna().head(20)
    if sample.empty:
        return False
    try:
        return bool(pd.to_datetime(sample, errors="raise",
                                   format="mixed").notna().all())
    except Exception:
        return False


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def guess_x_column(df: pd.DataFrame) -> Optional[str]:
    hints = ("time", "timestamp", "date", "datetime", "t", "fecha", "hora", "ts")
    for c in df.columns:
        if str(c).lower() in hints or is_datetime_like(df[c]):
            return c
    for c in numeric_columns(df):
        col = df[c].dropna()
        if len(col) > 2 and col.is_monotonic_increasing:
            return c
    return None


def _to_datetime_fast(col: pd.Series) -> pd.Series:
    """Deduce el formato con una fila y lo aplica fijo al resto.

    format="mixed" reintenta formato POR FILA: en millones de filas son
    minutos. Con formato fijo pandas usa la ruta vectorizada en C.
    """
    if pd.api.types.is_datetime64_any_dtype(col):
        return col
    sample = col.dropna().head(200)
    if sample.empty:
        return pd.to_datetime(col, errors="coerce")
    fmt = None
    try:
        from pandas._libs.tslibs.parsing import guess_datetime_format
        fmt = guess_datetime_format(str(sample.iloc[0]))
    except Exception:
        fmt = None
    if fmt:
        try:
            out = pd.to_datetime(col, errors="coerce", format=fmt)
            if out.notna().sum() >= 0.99 * col.notna().sum():
                return out
        except Exception:
            pass
    return pd.to_datetime(col, errors="coerce", format="mixed")


def build_x(df: pd.DataFrame, x_mode: str, x_column: Optional[str]
            ) -> tuple[np.ndarray, bool]:
    """Devuelve (x float64, es_datetime). Datetime -> segundos epoch (UTC).
    X siempre float64: en float32 dos instantes a 100 ms colapsan en el mismo
    valor y el zoom deja de funcionar."""
    if x_mode == "index" or not x_column:
        return np.arange(len(df), dtype=np.float64), False
    if x_column not in df.columns:
        raise LoadError(f"La columna X '{x_column}' no esta en el fichero.")

    col = df[x_column]
    if is_datetime_like(col):
        dt = _to_datetime_fast(col)
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        nat = dt.isna().to_numpy()
        x = dt.to_numpy(dtype="datetime64[ns]").astype("int64").astype(np.float64) / 1e9
        x[nat] = np.nan
        return x, True

    return pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64), False


def sanitize_axis(x: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Quita filas con X invalido y ordena por X. Sin esto todo lo demas miente
    (y setClipToView de pyqtgraph exige X creciente o dibuja basura)."""
    ok = np.isfinite(x)
    if not ok.all():
        x, df = x[ok], df.loc[ok].reset_index(drop=True)
    if len(x) > 1 and not np.all(np.diff(x) >= 0):
        idx = np.argsort(x, kind="stable")
        x, df = x[idx], df.take(idx).reset_index(drop=True)
    return x, df


# --- compatibilidad: se aplicaba despues de leer, ya no se usa en la ruta viva
def apply_sampling(df: pd.DataFrame, max_samples: Optional[int],
                   policy: str = "truncate") -> pd.DataFrame:
    if not max_samples or len(df) <= max_samples:
        return df
    if policy == "decimate":
        step = int(np.ceil(len(df) / max_samples))
        return df.iloc[::step].reset_index(drop=True)
    return df.iloc[:max_samples].reset_index(drop=True)


# ----------------------------------------------------------------------
def quick_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hash barato: tamano + primer y ultimo MB. Hashear 2 GB al abrir es absurdo."""
    import hashlib
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(0, size - chunk), os.SEEK_SET)
            h.update(f.read(chunk))
    return h.hexdigest()[:32]
