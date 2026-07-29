"""
Prescripcion sobre el modelo de 15 minutos.
===========================================

Idea
----
El modelo de modelo_15min.py predice y(t+15) en funcion del historico Y de las
acciones aplicadas en (t, t+15]. Eso permite convertirlo en un optimizador
de una sola jugada (una especie de MPC de horizonte 1 con modelo aprendido):

    1. Se toma el estado actual (fila de caracteristicas).
    2. Se generan N combinaciones candidatas de acciones futuras, dentro de
       los limites fisicos Y de los limites de movimiento por paso.
    3. Se sustituyen las columnas *__fut_media / *__fut_delta por cada
       candidata y se predicen los cuatro objetivos.
    4. Se elige la candidata que minimiza un coste multiobjetivo.

El coste refleja el compromiso central del proceso, ya senalado en el paper:
subir temperatura y tiempo de batido mejora el agotamiento (menos grasa en el
alpeorujo) pero degrada la calidad del aceite. Aqui se resuelve como
minimizacion de perdida con la temperatura como restriccion blanda.

AVISO IMPORTANTE
----------------
El modelo solo es fiable dentro del dominio de datos con el que se entreno.
Las recomendaciones se acotan a movimientos pequenos por ese motivo, y la
funcion `comprobar_direccionalidad` verifica que el modelo reacciona a las
acciones en el sentido fisicamente correcto antes de fiarse de el.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import argparse

import numpy as np
import pandas as pd

from modelo_15min import (ACCIONES, HORIZONTE, construir_caracteristicas,
                          mascara_valida, entrenar)


# ---------------------------------------------------------------------------
# Objetivos de operacion
# ---------------------------------------------------------------------------

@dataclass
class Objetivos:
    """Zona optima de funcionamiento y pesos del coste."""
    T_max: float = 30.0            # degC, limite de calidad del aceite
    T_min: float = 25.0            # por debajo el agotamiento se dispara
    humedad_min: float = 46.0      # % exigido por la orujera
    humedad_max: float = 60.0
    polifenoles_min: float = 200.0 # mg/kg
    grasa_objetivo: float = 4.0    # % de grasa en alpeorujo que se considera bueno

    w_grasa: float = 1.0
    w_T: float = 2.0
    w_humedad: float = 0.30
    w_polifenoles: float = 0.015
    w_movimiento: float = 0.8


# limites absolutos y de movimiento por decision (cada 15 min)
LIMITES = {
    "T_agua_ida":         (30.0, 68.0, 8.0),
    "caudal_agua":        (2.5, 9.0, 2.0),
    "caudal_pasta":       (4500.0, 9500.0, 800.0),
    "ratio_agua_proceso": (0.0, 0.15, 0.04),
}


def coste(pred: dict, accion: dict, actual: dict, obj: Objetivos) -> float:
    """Coste multiobjetivo de una accion candidata."""
    J = obj.w_grasa * max(0.0, pred["grasa_alpeorujo"] - obj.grasa_objetivo) ** 2

    # calidad: penalizacion asimetrica y fuerte por pasarse de temperatura
    J += obj.w_T * max(0.0, pred["T_pasta_s"] - obj.T_max) ** 2
    J += 0.3 * obj.w_T * max(0.0, obj.T_min - pred["T_pasta_s"]) ** 2

    # ventana de humedad de la orujera
    h = pred["humedad_alpeorujo"]
    J += obj.w_humedad * (max(0.0, obj.humedad_min - h) ** 2
                          + max(0.0, h - obj.humedad_max) ** 2)

    # polifenoles: penaliza caer por debajo del minimo
    J += obj.w_polifenoles * max(0.0, obj.polifenoles_min - pred["polifenoles"])

    # esfuerzo de control: movimientos normalizados por su limite
    esf = sum(((accion[a] - actual[a]) / LIMITES[a][2]) ** 2 for a in ACCIONES)
    J += obj.w_movimiento * esf
    return float(J)


# ---------------------------------------------------------------------------
# Prescriptor
# ---------------------------------------------------------------------------

class Prescriptor:

    def __init__(self, modelos: dict, columnas: list, objetivos: Objetivos | None = None,
                 n_candidatas: int = 400, semilla: int = 0, horizonte: int = HORIZONTE):
        self.horizonte = horizonte
        self.modelos = modelos
        self.columnas = columnas
        self.obj = objetivos or Objetivos()
        self.n = n_candidatas
        self.rng = np.random.default_rng(semilla)

    # -- generacion de candidatas ------------------------------------------

    def _candidatas(self, actual: dict) -> list[dict]:
        cands = [dict(actual)]                      # "no tocar nada"
        for _ in range(self.n - 1):
            c = {}
            for a in ACCIONES:
                lo, hi, paso = LIMITES[a]
                v = actual[a] + self.rng.uniform(-paso, paso)
                c[a] = float(np.clip(v, lo, hi))
            cands.append(c)
        return cands

    # -- prediccion ---------------------------------------------------------

    def _predecir(self, fila: pd.Series, cands: list[dict], actual: dict,
                  base_T: float) -> pd.DataFrame:
        M = pd.DataFrame([fila.values] * len(cands), columns=fila.index)
        for a in ACCIONES:
            vals = np.array([c[a] for c in cands])
            M[f"{a}__fut_media"] = vals
            M[f"{a}__fut_delta"] = vals - actual[a]
        M = M[self.columnas].astype(float)

        out = pd.DataFrame({a: [c[a] for c in cands] for a in ACCIONES})
        for nombre, mod in self.modelos.items():
            p = mod.predict(M)
            if nombre == "T_pasta_s":       # el modelo devuelve el incremento
                p = p + base_T
            out[nombre] = p
        return out

    # -- API ----------------------------------------------------------------

    def recomendar(self, fila_X: pd.Series, fila_df: pd.Series) -> dict:
        actual = {a: float(fila_df[a]) for a in ACCIONES}
        cands = self._candidatas(actual)
        pred = self._predecir(fila_X, cands, actual, float(fila_df["T_pasta_s"]))

        pred["coste"] = [
            coste(pred.iloc[i][list(self.modelos)].to_dict(), cands[i], actual, self.obj)
            for i in range(len(cands))
        ]
        i_best = int(pred["coste"].idxmin())
        return dict(
            actual=actual,
            sin_accion=pred.iloc[0].to_dict(),
            recomendado=pred.iloc[i_best].to_dict(),
            mejora_coste=float(pred["coste"].iloc[0] - pred["coste"].iloc[i_best]),
            frontera=pred.nsmallest(5, "coste"),
        )


# ---------------------------------------------------------------------------
# Validacion: el modelo, ¿reacciona en la direccion fisica correcta?
# ---------------------------------------------------------------------------

def comprobar_direccionalidad(modelos, columnas, X, df, idx,
                              horizonte: int = HORIZONTE, base_T_col="T_pasta_s"):
    """
    Perturba una sola accion cada vez y comprueba el signo de la respuesta.
    Signos esperados por fisica del proceso:
        T_agua_ida  ↑ -> T_pasta ↑ , grasa_alpeorujo ↓ , polifenoles ↓
        caudal_pasta ↑ -> menos tiempo de batido -> grasa_alpeorujo ↑
    """
    filas = X.loc[idx]
    base = {n: m.predict(filas[columnas]) for n, m in modelos.items()}
    base["T_pasta_s"] = base["T_pasta_s"] + df.loc[idx, base_T_col].to_numpy()

    # respuesta causal REAL medida en el simulador en bucle abierto
    # (ver experimentos.py, seccion "sensibilidad")
    REAL = {15: {"T_agua_ida": {"T_pasta_s": 0.357, "grasa_alpeorujo": -0.023},
                 "caudal_pasta": {"T_pasta_s": -0.306, "grasa_alpeorujo": 0.383}},
            45: {"T_agua_ida": {"T_pasta_s": 1.070, "grasa_alpeorujo": -0.420},
                 "caudal_pasta": {"T_pasta_s": -0.800, "grasa_alpeorujo": 1.200}}}
    ref = REAL.get(horizonte, {})
    pruebas = [("T_agua_ida", +5.0), ("caudal_pasta", +1000.0), ("caudal_agua", +1.5)]
    print(f"\n{'perturbacion':<26}" + "".join(f"{k:>20}" for k in modelos))
    print("-" * (26 + 20 * len(modelos)))
    for a, d in pruebas:
        M = filas.copy()
        M[f"{a}__fut_media"] = M[f"{a}__fut_media"] + d
        M[f"{a}__fut_delta"] = M[f"{a}__fut_delta"] + d
        fila = f"{a} {d:+.0f}"
        vals = []
        for n, m in modelos.items():
            p = m.predict(M[columnas])
            if n == "T_pasta_s":
                p = p + df.loc[idx, base_T_col].to_numpy()
            vals.append(float(np.mean(p - base[n])))
        print(f"{fila:<26}" + "".join(f"{v:>20.3f}" for v in vals))
        if a in ref:
            r = "".join(f"{ref[a].get(k, float('nan')):>20.3f}" for k in modelos)
            print(f"{'  (real en simulador)':<26}{r}")
    print("\nEsperado: T_agua ↑ -> T_pasta ↑, grasa ↓, polifenoles ↓")
    print("          caudal_pasta ↑ -> menos tiempo de batido -> grasa ↑")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="datos/termobatidora.csv")
    ap.add_argument("--n", type=int, default=8, help="numero de instantes a prescribir")
    ap.add_argument("--horizonte", type=int, default=HORIZONTE,
                    help="minutos vista. 15 gobierna temperatura; ~45 (un tiempo de "
                         "residencia) es el horizonte real para grasa y humedad")
    args = ap.parse_args()

    res = entrenar(args.csv, horizonte=args.horizonte, verbose=True)
    modelos, columnas, X, df = res["modelos"], res["columnas"], res["X"], res["df"]
    test = res["test"]

    idx_test = df.index[test]
    comprobar_direccionalidad(modelos, columnas, X, df, idx_test[::10],
                              horizonte=args.horizonte)

    presc = Prescriptor(modelos, columnas, horizonte=args.horizonte)
    idx = idx_test[:: max(1, len(idx_test) // args.n)][: args.n]

    print("\n" + "=" * 100)
    print("RECOMENDACIONES  (comparacion: no hacer nada  vs  accion propuesta)")
    print("=" * 100)
    filas = []
    for i in idx:
        r = presc.recomendar(X.loc[i], df.loc[i])
        sa, rc, ac = r["sin_accion"], r["recomendado"], r["actual"]
        filas.append(dict(
            hora=df.loc[i, "ts"][11:16] if isinstance(df.loc[i, "ts"], str)
                 else str(df.loc[i, "ts"])[11:16],
            T_agua=f"{ac['T_agua_ida']:.0f}->{rc['T_agua_ida']:.0f}",
            Q_agua=f"{ac['caudal_agua']:.1f}->{rc['caudal_agua']:.1f}",
            Q_pasta=f"{ac['caudal_pasta']:.0f}->{rc['caudal_pasta']:.0f}",
            agua_pr=f"{ac['ratio_agua_proceso']:.3f}->{rc['ratio_agua_proceso']:.3f}",
            T_sin=round(sa["T_pasta_s"], 1), T_rec=round(rc["T_pasta_s"], 1),
            grasa_sin=round(sa["grasa_alpeorujo"], 2),
            grasa_rec=round(rc["grasa_alpeorujo"], 2),
            hum_rec=round(rc["humedad_alpeorujo"], 1),
            polif_rec=round(rc["polifenoles"], 0),
            dJ=round(r["mejora_coste"], 2),
        ))
    print(pd.DataFrame(filas).to_string(index=False))
    d = pd.DataFrame(filas)
    print(f"\nReduccion media de grasa prevista a 15 min: "
          f"{(d.grasa_sin - d.grasa_rec).mean():.2f} puntos porcentuales")


if __name__ == "__main__":
    main()
