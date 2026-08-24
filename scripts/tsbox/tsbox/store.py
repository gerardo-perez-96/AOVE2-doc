"""Guardado del sidecar JSON.

Reglas que no se negocian:
  - escritura atómica (tmp + os.replace): un cuelgue no te deja un JSON a medias
  - .bak rotativo antes de sobrescribir
  - se rechaza pisar un .json que no sea nuestro
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from .loader import quick_hash
from .model import Project

MAGIC = "tsbox"


def sidecar_path(data_path: str | Path) -> Path:
    """data.csv -> data.json, como pediste. Ojo: colisiona si ya tienes un
    data.json que no es nuestro; por eso is_ours() antes de escribir."""
    p = Path(data_path)
    return p.with_suffix(".json")


def is_ours(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists():
        return True
    try:
        with open(p, "r", encoding="utf-8") as f:
            head = json.load(f)
        return head.get("app") == MAGIC
    except Exception:
        return False


def save(project: Project, path: str | Path) -> Path:
    p = Path(path)
    if not is_ours(p):
        raise FileExistsError(
            f"'{p.name}' ya existe y no lo generó esta herramienta. "
            f"No lo piso. Renombra el fichero de datos o borra ese JSON."
        )
    payload = {"app": MAGIC, **project.to_dict()}
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists():
        try:
            shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
        except OSError:
            pass

    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tsbox_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def load(path: str | Path) -> Project:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if d.get("app") != MAGIC:
        raise ValueError(f"'{Path(path).name}' no es un proyecto de esta herramienta.")
    return Project.from_dict(d)


def check_source(project: Project, data_path: str | Path) -> str | None:
    """Devuelve None si el fichero coincide, o un texto de aviso si no.
    Sin esto, renombrar el CSV te deja las etiquetas apuntando al vacío."""
    src = project.source
    if not src.quick_hash:
        return None
    p = Path(data_path)
    try:
        actual = quick_hash(p)
        size = p.stat().st_size
    except OSError as e:
        return f"No se puede leer el fichero de datos: {e}"
    if actual != src.quick_hash:
        return (f"El fichero ha cambiado desde el último guardado "
                f"(tamaño {src.size} → {size}). Las anotaciones pueden estar "
                f"desplazadas respecto a los datos actuales.")
    return None
