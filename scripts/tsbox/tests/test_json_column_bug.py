"""El bug que reportó el usuario: cargar con menos columnas marcadas
borraba esas series del proyecto PARA SIEMPRE, incluso si en la siguiente
apertura se volvían a pedir todas.
"""
import numpy as np
import pandas as pd
import pytest

from tsbox.session import Session


@pytest.fixture
def csv(tmp_path):
    n = 2000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "a": rng.standard_normal(n), "b": rng.standard_normal(n),
        "c": rng.standard_normal(n), "d": rng.standard_normal(n),
    })
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    return p


def test_cargar_menos_columnas_no_borra_las_demas_del_proyecto(csv):
    s1 = Session()
    s1.open(csv, "column", "ts", float32=True)
    assert {x.name for x in s1.project.series} == {"a", "b", "c", "d"}
    s1.save()

    s2 = Session()
    s2.open(csv, "column", "ts", float32=True, selected_columns=["a"])
    assert {x.name for x in s2.project.series} == {"a", "b", "c", "d"}, (
        "las columnas no marcadas se han BORRADO del proyecto en vez de "
        "quedar ocultas")
    assert not any("ya no está en el fichero" in w for w in s2.warnings), (
        "aviso falso: la columna sigue en el fichero, solo no se cargó")
    s2.save()

    s3 = Session()
    s3.open(csv, "column", "ts", float32=True)     # todas otra vez
    assert {x.name for x in s3.project.series} == {"a", "b", "c", "d"}, (
        "las series descartadas en el paso 2 no han vuelto: el borrado fue "
        "permanente")


def test_columna_no_cargada_queda_oculta_no_visible(csv):
    s = Session()
    s.open(csv, "column", "ts", float32=True)
    s.save()
    s2 = Session()
    s2.open(csv, "column", "ts", float32=True, selected_columns=["a", "b"])
    hidden = [x for x in s2.project.series if x.name in ("c", "d")]
    assert len(hidden) == 2
    assert all(not x.visible for x in hidden)
    assert any("no estaban marcadas" in w for w in s2.warnings)


def test_values_de_columna_no_cargada_no_revienta(csv):
    s = Session()
    s.open(csv, "column", "ts", float32=True)
    s.save()
    s2 = Session()
    s2.open(csv, "column", "ts", float32=True, selected_columns=["a"])
    sid_c = s2.project.by_name("c").sid
    y = s2.values(sid_c)                     # no debe lanzar KeyError
    assert len(y) == len(s2.x)
    assert np.isnan(y).all()


def test_columna_realmente_eliminada_si_se_borra_del_json(csv):
    """El caso correcto de borrado: la columna deja de existir en el fichero
    de verdad. Ahí sí debe desaparecer del proyecto."""
    s1 = Session()
    s1.open(csv, "column", "ts", float32=True)
    s1.save()

    df = pd.read_csv(csv)
    df = df.drop(columns=["d"])
    df.to_csv(csv, index=False)

    s2 = Session()
    s2.open(csv, "column", "ts", float32=True)
    assert {x.name for x in s2.project.series} == {"a", "b", "c"}
    assert any("se elimina la serie 'd'" in w for w in s2.warnings)


def test_primera_apertura_con_pocas_columnas_solo_registra_esas(csv):
    """Comportamiento correcto de la PRIMERA apertura (sin JSON previo): el
    proyecto nace con lo que se pidió. No hay nada más que "recordar" todavía,
    así que no tiene sentido inventar series de columnas que nunca se han visto."""
    s1 = Session()
    s1.open(csv, "column", "ts", float32=True, selected_columns=["a"])
    assert {x.name for x in s1.project.series} == {"a"}


def test_reabrir_reactivando_columna_la_vuelve_a_mostrar(csv):
    """El caso que sí importa: proyecto ya existente (JSON guardado con TODAS
    las columnas), se reabre pidiendo menos, luego se pide 'b' de nuevo."""
    s1 = Session()
    s1.open(csv, "column", "ts", float32=True)     # todas -> las registra
    s1.save()
    s2 = Session()
    s2.open(csv, "column", "ts", float32=True, selected_columns=["a"])
    s2.save()
    s3 = Session()
    s3.open(csv, "column", "ts", float32=True)      # todas otra vez
    b = s3.project.by_name("b")
    assert b is not None
    assert b.visible is True                        # nunca se marcó no-visible
