"""Detección de formato largo (long / stacked).

El caso: un fichero donde cada instante aparece varias veces porque hay varias
entidades (reactores, sensores, máquinas) apiladas en las mismas columnas. Es
el formato normal de exportación de un historiador industrial, y es veneno para
un visor de series temporales:

  timestamp            reactor_id  reactor_temp
  2024-01-01 00:00:00  A_R1        181.1
  2024-01-01 00:00:00  A_R2        190.4     <- MISMO instante, otro reactor
  2024-01-01 00:00:00  A_R3        188.7

Si lo pintas tal cual, la "serie" salta entre reactores en cada muestra. El
salto medio entre muestras consecutivas sale mayor que la variación real de la
señal: estás dibujando ruido de entrelazado, no un proceso. Y la media global
es la media de seis máquinas distintas, un número que ninguna de las seis tiene.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LongFormat:
    is_long: bool
    x_column: str
    group_columns: list[str]      # candidatas, la mejor primero
    n_groups: int
    repeats: float                # veces que se repite cada instante
    n_rows: int
    message: str = ""

    def explain(self) -> str:
        return self.message


def _repeat_factor(x: pd.Series) -> tuple[float, int]:
    n = len(x)
    uniq = x.nunique()
    return (n / uniq if uniq else 1.0), uniq


def detect(df: pd.DataFrame, x_column: str | None,
           max_groups: int = 200) -> LongFormat:
    """¿Hay varias entidades apiladas sobre el mismo eje X?

    Criterio: el eje X repite valores, y existe alguna columna categórica cuyo
    número de valores distintos coincide con esa repetición. Esa columna es la
    que separa entidades.
    """
    if x_column is None or x_column not in df.columns or df.empty:
        return LongFormat(False, x_column or "", [], 0, 1.0, len(df))

    rep, uniq_x = _repeat_factor(df[x_column])
    if rep < 1.5:
        return LongFormat(False, x_column, [], 0, rep, len(df))

    cands: list[tuple[float, str, int]] = []
    for c in df.columns:
        if c == x_column:
            continue
        col = df[c]
        if pd.api.types.is_float_dtype(col):
            continue
        k = col.nunique(dropna=False)
        if not (2 <= k <= max_groups):
            continue
        # Buena candidata: (x, c) identifica una fila única, y k ≈ repeticiones
        score = -abs(k - rep)
        pairs = df.groupby([x_column, c], observed=True, dropna=False).size()
        if pairs.max() == 1:
            score += 100                     # separa perfectamente
        cands.append((score, c, k))

    cands.sort(reverse=True)
    groups = [c for _, c, _ in cands]
    n_groups = cands[0][2] if cands else 0

    if not groups:
        msg = (f"El eje X repite cada valor {rep:.1f} veces "
               f"({len(df):,} filas para {uniq_x:,} instantes distintos) y no se "
               f"ha encontrado ninguna columna que separe las entidades.\n\n"
               f"Si dibujas esto tal cual, la curva saltará entre entidades en "
               f"cada muestra y lo que verás será entrelazado, no la señal.")
        return LongFormat(True, x_column, [], 0, rep, len(df), msg)

    msg = (f"Este fichero está en formato LARGO: {len(df):,} filas para "
           f"{uniq_x:,} instantes distintos.\n\n"
           f"La columna «{groups[0]}» tiene {n_groups} valores distintos: son "
           f"{n_groups} entidades apiladas sobre el mismo eje de tiempo.\n\n"
           f"Si lo cargas tal cual, cada muestra consecutiva será una entidad "
           f"distinta, no el instante siguiente. Las medias, la correlación y "
           f"la detección de huecos mezclarán las {n_groups} y no describirán a "
           f"ninguna.")
    return LongFormat(True, x_column, groups, n_groups, rep, len(df), msg)


def group_values(df: pd.DataFrame, column: str, limit: int = 200) -> list[str]:
    vals = pd.unique(df[column].dropna())
    return [str(v) for v in vals[:limit]]


def filter_group(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """Se queda con una entidad. Devuelve una serie temporal de verdad."""
    return df.loc[df[column].astype(str) == str(value)].reset_index(drop=True)


def pivot_wide(df: pd.DataFrame, x_column: str, group_column: str,
               value_columns: list[str] | None = None,
               sep: str = "·") -> pd.DataFrame:
    """Convierte a formato ancho: una columna por (señal, entidad).

    6 reactores × 3 señales -> 18 columnas, cada una una serie temporal real
    sobre un eje X sin duplicados. Es lo que hay que hacer para comparar
    entidades entre sí.
    """
    if value_columns is None:
        value_columns = [c for c in df.columns
                         if c not in (x_column, group_column)
                         and pd.api.types.is_numeric_dtype(df[c])]
    out = df.pivot_table(index=x_column, columns=group_column,
                         values=value_columns, aggfunc="first",
                         observed=True, sort=True)
    out.columns = [f"{a}{sep}{b}" for a, b in out.columns]
    return out.reset_index()


def estimate_wide_columns(df: pd.DataFrame, x_column: str,
                          group_column: str) -> tuple[int, int]:
    """(columnas resultantes, filas resultantes) sin hacer el pivot.
    Pivotar 200 entidades × 30 señales son 6.000 columnas: mejor avisar antes."""
    nvals = sum(1 for c in df.columns
                if c not in (x_column, group_column)
                and pd.api.types.is_numeric_dtype(df[c]))
    return nvals * df[group_column].nunique(), df[x_column].nunique()
