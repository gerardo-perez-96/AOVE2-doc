import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsbox import gaps, loader, store, transforms
from tsbox.model import KIND_DERIVATIVE, KIND_ROLLING_MEAN, Project, SeriesDef, new_id
from tsbox.session import Session


@pytest.fixture
def csv(tmp_path) -> Path:
    n = 500
    t = pd.date_range("2024-01-01", periods=n, freq="1s")
    t = t.delete(range(200, 230))              # hueco temporal real
    y = np.sin(np.arange(len(t)) / 20.0)
    y[100:110] = np.nan                        # valores ausentes
    df = pd.DataFrame({"ts": t, "sig": y, "otra": np.arange(len(t), dtype=float),
                       "texto": ["a"] * len(t)})
    p = tmp_path / "datos.csv"
    df.to_csv(p, index=False)
    return p


def test_gaps_detecta_nan_y_saltos():
    x = np.arange(100, dtype=float)
    x = np.delete(x, range(40, 50))
    y = np.ones(len(x))
    y[10:15] = np.nan
    r = gaps.missing_report(x, y)
    assert r["n_nan"] == 5
    assert len(r["nan_intervals"]) == 1
    assert len(r["time_gaps"]) == 1
    assert r["muestras_perdidas_estimadas"] == 10


def test_derivada_y_media_movil():
    x = np.arange(0, 10, 0.1)
    y = 3 * x
    d = transforms.derivative(x, y)
    assert np.allclose(d, 3.0)
    m = transforms.rolling_mean(np.array([1.0, 2, 3, 4, 5]), 3)
    assert len(m) == 5 and np.isfinite(m).all()


def test_butterworth_lowpass_atenua_alta_frecuencia():
    fs = 100.0
    t = np.arange(2000) / fs
    slow = np.sin(2 * np.pi * 1.0 * t)      # 1 Hz, dentro de la banda de paso
    fast = np.sin(2 * np.pi * 30.0 * t)     # 30 Hz, muy por encima del corte
    out = transforms.butterworth(t, slow + fast, btype="low", order=4, cutoff=5.0)
    assert np.corrcoef(out[200:], slow[200:])[0, 1] > 0.95
    assert np.std(out[200:] - slow[200:]) < 0.3


def test_butterworth_highpass_quita_tendencia():
    fs = 100.0
    t = np.arange(2000) / fs
    trend = 0.01 * t
    fast = np.sin(2 * np.pi * 20.0 * t)
    out = transforms.butterworth(t, trend + fast, btype="high", order=4, cutoff=2.0)
    assert abs(np.mean(out[200:])) < 0.2


def test_butterworth_hueco_no_contamina_el_resto():
    fs = 50.0
    t = np.arange(1000) / fs
    y = np.sin(2 * np.pi * 1.0 * t)
    y[400:410] = np.nan
    out = transforms.butterworth(t, y, btype="low", order=4, cutoff=5.0)
    assert np.isnan(out[400:410]).all()
    assert np.isfinite(out[:300]).all()
    assert np.isfinite(out[500:]).all()


def test_butterworth_fase_cero_no_desplaza_causal_si():
    fs = 200.0
    t = np.arange(4000) / fs
    y = np.sin(2 * np.pi * 2.0 * t)
    zero = transforms.butterworth(t, y, btype="low", order=4, cutoff=10.0,
                                  zero_phase=True)
    causal = transforms.butterworth(t, y, btype="low", order=4, cutoff=10.0,
                                    zero_phase=False)
    seg = slice(500, 3500)
    n = seg.stop - seg.start

    def lag(out):
        c = np.correlate(out[seg] - out[seg].mean(), y[seg] - y[seg].mean(),
                         mode="full")
        return int(np.argmax(c)) - (n - 1)

    assert abs(lag(zero)) <= 1
    assert lag(causal) > 2


def test_butterworth_bandpass_conserva_solo_la_banda():
    fs = 100.0
    t = np.arange(3000) / fs
    low, mid, high = (np.sin(2 * np.pi * f * t) for f in (2.0, 15.0, 40.0))
    out = transforms.butterworth(t, low + mid + high, btype="bandpass",
                                 order=4, cutoff=10.0, cutoff2=20.0)
    assert np.corrcoef(out[300:], mid[300:])[0, 1] > 0.9
    assert np.std(out[300:] - mid[300:]) < 0.4


def test_butterworth_corte_fuera_de_rango_no_revienta():
    x = np.arange(200.0)
    y = np.sin(x)
    out = transforms.butterworth(x, y, btype="low", order=4, cutoff=1e9)
    assert out.shape == y.shape   # se clippea a < Nyquist o devuelve NaN, no lanza


def test_stats_ventana():
    x = np.arange(100, dtype=float)
    y = x.copy()
    s = transforms.window_stats(x, y, 10, 20)
    assert 9 <= s["min"] <= 10 and 20 <= s["max"] <= 21


def test_x_index_y_max_muestras(csv):
    s = Session()
    s.open(csv, x_mode="index", x_column=None, max_samples=50)
    assert len(s.x) == 50 and s.x[0] == 0
    assert s.project.by_name("texto") is None      # no numérica, fuera


def test_x_datetime(csv):
    s = Session()
    s.open(csv, x_mode="column", x_column="ts")
    assert s.project.source.x_is_datetime
    assert s.x[1] - s.x[0] == pytest.approx(1.0)
    rep = s.missing(s.project.by_name("sig").sid)
    assert rep["n_nan"] == 10 and len(rep["time_gaps"]) == 1


def test_derivada_es_receta_no_datos(csv, tmp_path):
    s = Session()
    s.open(csv, x_mode="column", x_column="ts")
    base = s.project.by_name("sig")
    d = s.add_derived(base.sid, KIND_ROLLING_MEAN, {"window": 5})
    assert len(s.values(d.sid)) == len(s.x)
    s.save()
    raw = json.loads(s.json_path.read_text())
    assert raw["app"] == "tsbox"
    assert all("values" not in x for x in raw["series"])   # nunca guardamos arrays
    assert raw["series"][-1]["params"]["window"] == 5

    s2 = Session()
    s2.open(csv, x_mode="column", x_column="ts")           # recarga el sidecar
    assert s2.project.by_id(d.sid) is not None
    assert np.allclose(s2.values(d.sid), s.values(d.sid), equal_nan=True)


def test_butterworth_es_receta_no_datos(csv, tmp_path):
    """Mismo contrato que test_derivada_es_receta_no_datos, para el filtro
    Butterworth: el JSON guarda la receta (tipo, orden, corte, fase), no el
    array filtrado -- y recargarla reproduce el mismo resultado."""
    s = Session()
    s.open(csv, x_mode="column", x_column="ts")
    base = s.project.by_name("sig")
    d = s.add_derived(base.sid, "butterworth",
                      {"btype": "low", "order": 3, "cutoff": 0.05,
                       "zero_phase": True})
    assert len(s.values(d.sid)) == len(s.x)
    s.save()
    raw = json.loads(s.json_path.read_text())
    assert all("values" not in x for x in raw["series"])
    saved = raw["series"][-1]["params"]
    assert saved == {"btype": "low", "order": 3, "cutoff": 0.05, "zero_phase": True}

    s2 = Session()
    s2.open(csv, x_mode="column", x_column="ts")
    assert np.allclose(s2.values(d.sid), s.values(d.sid), equal_nan=True)


def test_borrar_padre_borra_hijos(csv):
    s = Session()
    s.open(csv, x_mode="index", x_column=None)
    base = s.project.by_name("sig")
    d = s.add_derived(base.sid, KIND_DERIVATIVE, {})
    s.project.remove_series(base.sid)
    assert s.project.by_id(d.sid) is None


def test_no_pisa_json_ajeno(tmp_path):
    p = tmp_path / "otro.json"
    p.write_text('{"algo": 1}')
    with pytest.raises(FileExistsError):
        store.save(Project(), p)


def test_guardado_atomico_deja_bak(tmp_path):
    p = tmp_path / "x.json"
    pr = Project()
    store.save(pr, p)
    store.save(pr, p)
    assert (tmp_path / "x.json.bak").exists()


def test_reordenar(csv):
    s = Session()
    s.open(csv, x_mode="index", x_column=None)
    ids = [x.sid for x in s.project.ordered()]
    s.project.reorder(list(reversed(ids)))
    assert [x.sid for x in s.project.ordered()] == list(reversed(ids))


def test_eje_x_desordenado_se_ordena(tmp_path):
    df = pd.DataFrame({"t": [3.0, 1.0, 2.0], "v": [30.0, 10.0, 20.0]})
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    s = Session()
    s.open(p, x_mode="column", x_column="t")
    assert list(s.x) == [1.0, 2.0, 3.0]
    assert list(s.values(s.project.by_name("v").sid)) == [10.0, 20.0, 30.0]
