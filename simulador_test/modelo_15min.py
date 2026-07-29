"""
Modelo de prediccion a 15 minutos vista del estado de la termobatidora.
======================================================================

Planteamiento
-------------
Un modelo por objetivo, de la forma

    y(t+H) = f( historico de senales hasta t ,  acciones aplicadas en (t, t+H] )

La inclusion explicita de las ACCIONES FUTURAS es lo que convierte al modelo en
utilizable para PRESCRIPCION: el prescriptor evalua "que pasaria si subo el agua
a 52 C" cambiando solo ese bloque de columnas. Es la misma idea que el modelo
interno de un MPC, pero aprendido de datos en lugar de derivado de balances.

Dos familias de objetivos, que se tratan de forma distinta:

  (a) MEDIBLES ONLINE  -> T_pasta_s
      Se modela el INCREMENTO  y(t+H) - y(t).  Bate a la persistencia porque
      el modelo solo tiene que explicar la deriva, no el nivel.

  (b) NO MEDIBLES ONLINE -> grasa y humedad del alpeorujo, polifenoles
      Es el problema del sensor inferencial de Bordons & Zafra (2003).
      Se modela el NIVEL, con linea base "ultimo resultado de laboratorio".

Retardos usados (compatibles con el articulo: retardos 4.5-8 min, tiempos
caracteristicos 4-7 min): 0, 3, 6, 10, 15 y 30 minutos.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

HORIZONTE = 15                      # minutos

SENALES = [
    "T_pasta_e", "T_pasta_1", "T_pasta_2", "T_pasta_s",
    "T_agua_ida", "T_agua_ret", "caudal_pasta", "caudal_agua",
    "T_ambiente", "amperios_batidora",
    "ratio_agua_proceso", "t_residencia_est", "salto_termico", "dT_agua", "Q_agua_kW",
]

# Analitica de recepcion de la partida en curso (NIR / Abencor en el patio).
# El articulo de 2003 no la usaba porque no se media, y anticipa que incluirla
# "could only make our results improve". El experimento lo confirma.
RECEPCION = ["rec_humedad_aceituna", "rec_grasa_aceituna", "rec_indice_madurez"]

# Variables manipulables por el operario / el sistema de control
ACCIONES = ["T_agua_ida", "caudal_agua", "caudal_pasta", "ratio_agua_proceso"]

RETARDOS = [0, 3, 6, 10, 15, 30]    # min
VENTANAS = [15, 30, 60]             # min

# objetivo -> (columna de verdad de campo, se_mide_online)
OBJETIVOS = {
    "T_pasta_s":         ("real_T_pasta_s", True),
    "grasa_alpeorujo":   ("real_grasa_alpeorujo", False),
    "humedad_alpeorujo": ("real_humedad_alpeorujo", False),
    "polifenoles":       ("real_polifenoles", False),
}


# ---------------------------------------------------------------------------
# Caracteristicas
# ---------------------------------------------------------------------------

def construir_caracteristicas(df: pd.DataFrame,
                              horizonte: int = HORIZONTE,
                              usar_recepcion: bool = True) -> pd.DataFrame:
    """Columnas de entrada al modelo. Todo es calculable en planta."""
    cols = {}

    # --- 1. Historico: retardos, medias moviles y tendencia ----------------
    for s in SENALES:
        for d in RETARDOS:
            cols[f"{s}__lag{d}"] = df[s].shift(d)
        for w in VENTANAS:
            cols[f"{s}__med{w}"] = df[s].rolling(w, min_periods=w // 2).mean()
        cols[f"{s}__pend15"] = df[s] - df[s].shift(15)

    # --- 2. Analitica de recepcion de la partida ---------------------------
    if usar_recepcion:
        for c in RECEPCION:
            cols[c] = df[c]
        cols["min_en_partida"] = df.groupby("partida_id").cumcount().clip(upper=180)

    # --- 3. Contexto operativo --------------------------------------------
    cols["modo_auto"] = df["modo_auto"]
    marcha = df["en_marcha"]
    cols["min_desde_arranque"] = (
        marcha.groupby((marcha != marcha.shift()).cumsum()).cumcount()
        .where(marcha == 1, 0).clip(upper=180)
    )
    cols["hora_dia"] = df["ts"].dt.hour + df["ts"].dt.minute / 60.0

    # --- 4. Anclaje de laboratorio -----------------------------------------
    for c in ["lab_grasa_alpeorujo", "lab_humedad_alpeorujo"]:
        cols[c] = df[c].ffill()
        idx = df.index.to_series()
        cols[f"{c}__antig"] = (idx - idx.where(df[c].notna()).ffill()).clip(upper=300)

    # --- 5. ACCIONES FUTURAS en la ventana (t, t+H] ------------------------
    for a in ACCIONES:
        fut = df[a].shift(-horizonte).rolling(horizonte, min_periods=1).mean()
        cols[f"{a}__fut_media"] = fut
        cols[f"{a}__fut_delta"] = fut - df[a]

    return pd.DataFrame(cols, index=df.index)


def mascara_valida(df: pd.DataFrame, horizonte: int = HORIZONTE) -> pd.Series:
    """
    Filtra paradas, arranques y transitorios de puesta en marcha.
    El articulo insiste en este punto: los datos de parada o limpieza deben
    eliminarse antes de entrenar, o el modelo aprende dinamicas espurias.
    """
    marcha = df["en_marcha"] == 1
    pasado_ok = marcha.rolling(60, min_periods=1).min().astype(bool)
    futuro_ok = marcha[::-1].rolling(horizonte + 1, min_periods=1).min()[::-1].astype(bool)
    tras_arranque = marcha.groupby((marcha != marcha.shift()).cumsum()).cumcount() >= 45
    return marcha & pasado_ok & futuro_ok & tras_arranque


def _estimador(tipo: str):
    if tipo == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=6,
        min_samples_leaf=25, l2_regularization=1.0, random_state=0)


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def entrenar(csv: str = "datos/termobatidora.csv",
             horizonte: int = HORIZONTE,
             frac_train: float = 2 / 3,
             tipo: str = "gbt",
             usar_recepcion: bool = True,
             verbose: bool = True):

    df = pd.read_csv(csv, parse_dates=["ts"])
    X = construir_caracteristicas(df, horizonte, usar_recepcion)
    ok = mascara_valida(df, horizonte) & X.notna().all(axis=1)

    t_corte = df["t_min"].max() * frac_train
    itr = ok & (df["t_min"] < t_corte)
    ite = ok & (df["t_min"] >= t_corte)

    if verbose:
        print(f"Muestras validas: {ok.sum()} de {len(df)} "
              f"({len(df)-ok.sum()} descartadas por parada/transitorio/NaN)")
        print(f"  entrenamiento: {itr.sum()} min   test: {ite.sum()} min\n")
        print(f"{'objetivo':<19}{'modo':<8}{'MAE':>8}{'RMSE':>8}{'R2':>7}"
              f"{'baseline':>11}{'MAE':>7}{'mejora':>8}")
        print("-" * 76)

    modelos, metricas = {}, {}
    for nombre, (col, online) in OBJETIVOS.items():
        y_fut = df[col].shift(-horizonte)
        val = itr & y_fut.notna()
        vte = ite & y_fut.notna()

        if online:
            # ---- se modela el incremento respecto al valor medido ahora ----
            est = _estimador(tipo).fit(X[val], y_fut[val] - df.loc[val, col])
            pred = est.predict(X[vte]) + df.loc[vte, col].to_numpy()
            baseline = df.loc[vte, col]              # persistencia (es medible)
            etq_base, modo = "persist.", "delta"
        else:
            est = _estimador(tipo).fit(X[val], y_fut[val])
            pred = est.predict(X[vte])
            lab = {"grasa_alpeorujo": "lab_grasa_alpeorujo",
                   "humedad_alpeorujo": "lab_humedad_alpeorujo"}.get(nombre)
            if lab:
                baseline = df[lab].ffill()[vte]
                etq_base = "ult. lab"
            else:
                baseline = pd.Series(y_fut[val].mean(), index=df.index[vte])
                etq_base = "media"
            modo = "nivel"

        real = y_fut[vte]
        m = dict(
            modo=modo,
            MAE=float(mean_absolute_error(real, pred)),
            RMSE=float(np.sqrt(mean_squared_error(real, pred))),
            R2=float(r2_score(real, pred)),
            baseline=etq_base,
            MAE_baseline=float(mean_absolute_error(real, baseline)),
            rango=[round(float(real.min()), 2), round(float(real.max()), 2)],
        )
        metricas[nombre], modelos[nombre] = m, est

        if verbose:
            mej = 100 * (1 - m["MAE"] / m["MAE_baseline"])
            print(f"{nombre:<19}{modo:<8}{m['MAE']:>8.3f}{m['RMSE']:>8.3f}"
                  f"{m['R2']:>7.2f}{etq_base:>11}{m['MAE_baseline']:>7.2f}{mej:>7.0f}%")

    return dict(modelos=modelos, metricas=metricas, columnas=list(X.columns),
                horizonte=horizonte, usar_recepcion=usar_recepcion,
                df=df, X=X, test=ite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="datos/termobatidora.csv")
    ap.add_argument("--tipo", default="gbt", choices=["gbt", "ridge"])
    ap.add_argument("--horizonte", type=int, default=HORIZONTE)
    ap.add_argument("--sin-recepcion", action="store_true",
                    help="entrena sin la analitica de recepcion (escenario 2003)")
    args = ap.parse_args()

    res = entrenar(args.csv, args.horizonte, tipo=args.tipo,
                   usar_recepcion=not args.sin_recepcion)

    import joblib
    Path("modelos").mkdir(exist_ok=True)
    joblib.dump({"modelos": res["modelos"], "columnas": res["columnas"],
                 "horizonte": res["horizonte"],
                 "usar_recepcion": res["usar_recepcion"]},
                "modelos/predictor_15min.joblib")
    Path("modelos/metricas.json").write_text(json.dumps(res["metricas"], indent=2))
    print("\nGuardado: modelos/predictor_15min.joblib")


if __name__ == "__main__":
    main()
