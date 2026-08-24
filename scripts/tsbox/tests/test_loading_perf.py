"""Los filtros deben aplicarse DURANTE la lectura, no después.
Si estos tests pasan pero el fichero grande sigue tardando, la regresión está
en otro sitio; si fallan, se ha reintroducido la lectura completa.
"""
import time

import numpy as np
import pandas as pd
import pytest

from tsbox import loader
from tsbox.session import Session


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    d = tmp_path_factory.mktemp("big")
    n, k = 200_000, 10
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=n, freq="100ms")})
    for i in range(k):
        df[f"s{i}"] = np.cumsum(rng.standard_normal(n))
    csv, pq = d / "big.csv", d / "big.parquet"
    df.to_csv(csv, index=False)
    df.to_parquet(pq, index=False)
    return csv, pq, n, k


# --- separador -------------------------------------------------------
@pytest.mark.parametrize("sep", [",", ";", "\t", "|"])
def test_sniff_sep(tmp_path, sep):
    p = tmp_path / "x.csv"
    p.write_text(sep.join(["a", "b", "c"]) + "\n" +
                 "\n".join(sep.join("123") for _ in range(5)))
    assert loader.sniff_sep(p) == sep
    assert list(loader.read_table(p).columns) == ["a", "b", "c"]


def test_no_usa_el_parser_de_python(monkeypatch, big):
    """El parser de Python es 6.5x más lento y usa 2.6x más RAM.
    Si alguien vuelve a poner engine='python', esto revienta."""
    csv, _, _, _ = big
    real = pd.read_csv

    def guard(*a, **kw):
        assert kw.get("engine") != "python", "regresión: engine='python'"
        assert kw.get("sep") is not None, "regresión: sep=None fuerza el parser Python"
        return real(*a, **kw)

    monkeypatch.setattr(pd, "read_csv", guard)
    loader.read_table(csv, nrows=100)


# --- pushdown de filtros ---------------------------------------------
def test_nrows_no_lee_el_fichero_entero(big):
    csv, _, n, _ = big
    t = time.perf_counter()
    df = loader.read_table(csv, nrows=1000)
    corto = time.perf_counter() - t
    assert len(df) == 1000

    t = time.perf_counter()
    loader.read_table(csv)
    largo = time.perf_counter() - t
    assert corto < largo / 3, "nrows no se está empujando al lector"


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_columns_pushdown(big, fmt):
    csv, pq, _, _ = big
    p = csv if fmt == "csv" else pq
    df = loader.read_table(p, columns=["ts", "s0", "s3"])
    assert list(df.columns) == ["ts", "s0", "s3"]


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_decimate_step(big, fmt):
    csv, pq, n, _ = big
    p = csv if fmt == "csv" else pq
    full = loader.read_table(p, columns=["s0"])["s0"].to_numpy()
    dec = loader.read_table(p, columns=["s0"], decimate_step=10)["s0"].to_numpy()
    assert len(dec) == pytest.approx(n / 10, rel=0.01)
    # y son exactamente las muestras 0, 10, 20... no un remuestreo inventado
    assert np.allclose(dec[:50], full[::10][:50])


def test_decimate_no_se_desalinea_entre_chunks(tmp_path):
    """El decimado por trozos debe mantener la rejilla global; si cada chunk
    empieza en 0, se duplican muestras en las costuras."""
    n = 2_500_000
    p = tmp_path / "seq.csv"
    pd.DataFrame({"i": np.arange(n)}).to_csv(p, index=False)
    got = loader.read_table(p, decimate_step=7)["i"].to_numpy()
    assert np.array_equal(got, np.arange(0, n, 7))


def test_float32_reduce_memoria(big):
    csv, _, _, _ = big
    a = loader.read_table(csv, columns=["s0", "s1"])
    b = loader.read_table(csv, columns=["s0", "s1"], float32=True)
    assert b["s0"].dtype == np.float32
    assert b.memory_usage(deep=False).sum() < a.memory_usage(deep=False).sum() * 0.6


# --- plan de muestreo -------------------------------------------------
def test_plan_sampling(big):
    csv, _, n, _ = big
    nrows, step, total = loader.plan_sampling(csv, 20_000, "truncate")
    assert nrows == 20_000 and step == 1
    nrows, step, total = loader.plan_sampling(csv, 20_000, "decimate")
    assert nrows is None and step >= 9
    nrows, step, _ = loader.plan_sampling(csv, None, "truncate")
    assert nrows is None and step == 1
    # si el límite es mayor que el fichero, no se toca nada
    nrows, step, _ = loader.plan_sampling(csv, 10_000_000, "truncate")
    assert nrows is None and step == 1


def test_estimate_rows_razonable(big):
    csv, pq, n, _ = big
    assert abs(loader.estimate_rows(csv) - n) / n < 0.05
    assert loader.estimate_rows(pq) == n
    assert loader.count_rows(csv) == n


# --- fechas -----------------------------------------------------------
def test_parseo_fechas_usa_formato_fijo(big):
    csv, _, n, _ = big
    df = loader.read_table(csv, columns=["ts"])
    t = time.perf_counter()
    x, is_dt = loader.build_x(df, "column", "ts")
    dur = time.perf_counter() - t
    assert is_dt and len(x) == n
    assert x.dtype == np.float64
    assert dur < 2.0, "el parseo de fechas se ha vuelto a ir a format='mixed'"
    assert np.all(np.diff(x) > 0)


def test_x_siempre_float64_aunque_pidas_float32(big):
    """En float32 dos instantes a 100 ms colapsan y el zoom deja de servir."""
    csv, _, _, _ = big
    s = Session()
    s.open(csv, "column", "ts", float32=True)
    assert s.x.dtype == np.float64
    assert len(np.unique(s.x)) == len(s.x)


# --- session end to end ------------------------------------------------
def test_session_solo_carga_lo_pedido(big):
    csv, _, n, k = big
    s = Session()
    s.open(csv, "column", "ts", max_samples=5_000,
           selected_columns=["s0", "s1"], float32=True)
    assert len(s.x) == 5_000
    assert len(s.project.series) == 2
    assert list(s.df.columns) == ["ts", "s0", "s1"]
    assert s.memory_mb() < 5


def test_session_decimando(big):
    csv, _, n, _ = big
    s = Session()
    s.open(csv, "column", "ts", max_samples=10_000, sample_policy="decimate",
           selected_columns=["s0"], float32=True)
    assert 9_000 < len(s.x) < 11_000
    assert np.all(np.diff(s.x) > 0)


def test_progreso_se_reporta(big):
    csv, _, _, _ = big
    seen = []
    s = Session()
    s.open(csv, "column", "ts", selected_columns=["s0"],
           progress=lambda m, f: seen.append((m, f)))
    assert len(seen) >= 3
    assert 0.0 <= min(f for _, f in seen) and max(f for _, f in seen) <= 1.0
    assert seen == sorted(seen, key=lambda t: t[1])


def test_cancelar_propaga(big):
    csv, _, _, _ = big

    def boom(msg, frac):
        raise InterruptedError("cancelado")

    s = Session()
    with pytest.raises(InterruptedError):
        s.open(csv, "column", "ts", progress=boom)


# --- diálogo de carga: no debe tocar el fichero entero -----------------
def test_dialogo_no_lee_el_fichero_entero(big, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt
    from tsbox.dialogs import LoadDialog

    csv, _, n, k = big
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    t = time.perf_counter()
    d = LoadDialog(csv)
    assert time.perf_counter() - t < 1.5, "el diálogo está leyendo el fichero entero"
    assert "filas" in d.cost.text()

    # poblar la tabla no debe reventar por señales a medio construir
    d._check_all(False)
    d.table.item(0, 0).setCheckState(Qt.Checked)
    assert d.selected_columns() == ["s0"]
    assert d.float32() is True
