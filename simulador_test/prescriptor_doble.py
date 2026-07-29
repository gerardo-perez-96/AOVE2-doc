"""
Prescriptor de doble horizonte.
===============================

Consecuencia directa del hallazgo (d) del README: el tiempo de residencia de la
batidora es de ~50 min, asi que una accion tomada ahora

    - mueve la TEMPERATURA en 10-15 min   (dinamica termica del encamisado)
    - mueve el AGOTAMIENTO en 45-60 min   (la pasta que sale ahora ya estaba dentro)

Usar un unico horizonte obliga a elegir mal:
    H=15 -> la grasa apenas responde, el optimizador no encuentra palanca
    H=45 -> la restriccion de temperatura llega tarde para protegerla

Aqui se entrenan DOS juegos de modelos y se evalua cada accion candidata contra
los dos a la vez:

    H_corto (15 min) -> T_pasta_s, polifenoles   -> RESTRICCIONES de calidad
    H_largo (45 min) -> grasa, humedad del alpeorujo -> OBJETIVO de rendimiento

La accion se mantiene constante durante toda la ventana larga, que es como
opera de verdad un operario: se toca la consigna y se deja actuar.

Uso:
    python prescriptor_doble.py --csv datos/termobatidora_exc.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from modelo_15min import ACCIONES, construir_caracteristicas, entrenar
from prescriptor import LIMITES, Objetivos


# ---------------------------------------------------------------------------
# Guardarrail: ganancias maximas creibles
# ---------------------------------------------------------------------------
#
# Un optimizador siempre explota el error de su modelo: si el modelo cree que
# bajar 600 kg/h el caudal quita 4 puntos de grasa, propondra eso aunque el
# ensayo de escalon diga que el efecto real no pasa de 0,8 puntos.
#
# Estas ganancias se miden en planta (o aqui, con experimentos.respuesta_real)
# y acotan cuanto se le permite creer al modelo. Unidades: variacion del
# objetivo por unidad de accion. Con margen x1.5 sobre lo medido.

GANANCIAS = {
    # horizonte: { accion: { objetivo: |ganancia maxima| por unidad de accion } }
    # Medidas con experimentos.respuesta_real (media de 3 semillas), x1.5 de margen.
    15: {
        "T_agua_ida":         {"T_pasta_s": 0.107,    "grasa_alpeorujo": 0.0069},
        "caudal_pasta":       {"T_pasta_s": 0.00048,  "grasa_alpeorujo": 0.00060},
        "caudal_agua":        {"T_pasta_s": 0.459,    "grasa_alpeorujo": 0.055},
        "ratio_agua_proceso": {"T_pasta_s": 0.0,      "grasa_alpeorujo": 6.56},
    },
    45: {
        "T_agua_ida":         {"T_pasta_s": 0.329,    "grasa_alpeorujo": 0.127},
        "caudal_pasta":       {"T_pasta_s": 0.00114,  "grasa_alpeorujo": 0.00174},
        "caudal_agua":        {"T_pasta_s": 1.044,    "grasa_alpeorujo": 0.453},
        "ratio_agua_proceso": {"T_pasta_s": 0.0,      "grasa_alpeorujo": 17.3},
    },
}


# Ganancias CON SIGNO, sin margen: las que usa el modo hibrido.
# Aviso: la del agua de proceso no es lineal (tiene un optimo en forma de U),
# asi que su signo depende del punto de operacion. Para usarla en serio hay que
# sustituirla por el termino fisico f(humedad efectiva) del decanter.
GANANCIAS_FIRMADAS = {
    15: {
        "T_agua_ida":         {"T_pasta_s": 0.0713,   "grasa_alpeorujo": -0.00457},
        "caudal_pasta":       {"T_pasta_s": -0.00032, "grasa_alpeorujo": 0.00040},
        "caudal_agua":        {"T_pasta_s": 0.3063,   "grasa_alpeorujo": -0.03677},
        "ratio_agua_proceso": {"T_pasta_s": 0.0,      "grasa_alpeorujo": 4.373},
    },
    45: {
        "T_agua_ida":         {"T_pasta_s": 0.2194,   "grasa_alpeorujo": -0.08451},
        "caudal_pasta":       {"T_pasta_s": -0.00076, "grasa_alpeorujo": 0.00116},
        "caudal_agua":        {"T_pasta_s": 0.6959,   "grasa_alpeorujo": -0.30198},
        "ratio_agua_proceso": {"T_pasta_s": 0.0,      "grasa_alpeorujo": 11.532},
    },
}


def aplicar_ganancias(pred: pd.DataFrame, sin_accion: pd.Series,
                      cands: list[dict], actual: dict, horizonte: int) -> pd.DataFrame:
    """
    Modo HIBRIDO. El modelo aprendido aporta solo la linea base ("que pasa si no
    toco nada"); el efecto de la accion lo pone la ganancia medida en ensayo de
    escalon. Es el modo defendible cuando hay pocos datos: separa lo que el
    modelo sabe hacer bien (predecir la deriva) de lo que no (atribuir causas).
    """
    g = GANANCIAS_FIRMADAS.get(horizonte)
    if g is None:
        return pred
    out = pred.copy()
    for obj in ("T_pasta_s", "grasa_alpeorujo"):
        if obj not in out.columns:
            continue
        efecto = np.array([
            sum((c[a] - actual[a]) * g[a].get(obj, 0.0) for a in ACCIONES)
            for c in cands])
        out[obj] = sin_accion[obj] + efecto
    return out


def acotar_por_ganancia(pred: pd.DataFrame, sin_accion: pd.Series,
                        cands: list[dict], actual: dict, horizonte: int) -> pd.DataFrame:
    """
    Recorta la variacion predicha respecto a "no hacer nada" al maximo que
    justifican las ganancias medidas en el ensayo de escalon.
    Deja intacto lo que el modelo prediga por evolucion natural del proceso.
    """
    g = GANANCIAS.get(horizonte)
    if g is None:
        return pred
    out = pred.copy()
    for obj in ("T_pasta_s", "grasa_alpeorujo"):
        if obj not in out.columns:
            continue
        tope = np.zeros(len(cands))
        for k, c in enumerate(cands):
            tope[k] = sum(abs(c[a] - actual[a]) * g[a].get(obj, 0.0) for a in ACCIONES)
        delta = out[obj].to_numpy() - sin_accion[obj]
        out[obj] = sin_accion[obj] + np.clip(delta, -tope, tope)
    return out


# ---------------------------------------------------------------------------
# Coste
# ---------------------------------------------------------------------------

def coste_doble(corto: dict, largo: dict, accion: dict, actual: dict,
                obj: Objetivos) -> float:
    """
    corto: predicciones a 15 min  (proteger calidad ya)
    largo: predicciones a 45 min  (perseguir rendimiento)
    """
    # --- objetivo: agotamiento, a horizonte largo -------------------------
    J = obj.w_grasa * max(0.0, largo["grasa_alpeorujo"] - obj.grasa_objetivo) ** 2

    # --- restriccion de temperatura: se vigila en AMBOS horizontes --------
    # el corto protege de un pico inminente, el largo del regimen permanente
    for pred, peso in ((corto, 1.0), (largo, 0.6)):
        J += peso * obj.w_T * max(0.0, pred["T_pasta_s"] - obj.T_max) ** 2
        J += peso * 0.3 * obj.w_T * max(0.0, obj.T_min - pred["T_pasta_s"]) ** 2

    # --- ventana de humedad exigida por la orujera (horizonte largo) ------
    h = largo["humedad_alpeorujo"]
    J += obj.w_humedad * (max(0.0, obj.humedad_min - h) ** 2
                          + max(0.0, h - obj.humedad_max) ** 2)

    # --- calidad del aceite: el dano termico se acumula, mira al largo ----
    J += obj.w_polifenoles * max(0.0, obj.polifenoles_min - largo["polifenoles"])

    # --- esfuerzo de control ---------------------------------------------
    J += obj.w_movimiento * sum(
        ((accion[a] - actual[a]) / LIMITES[a][2]) ** 2 for a in ACCIONES)
    return float(J)


# ---------------------------------------------------------------------------
# Prescriptor
# ---------------------------------------------------------------------------

class PrescriptorDoble:

    def __init__(self, res_corto: dict, res_largo: dict,
                 objetivos: Objetivos | None = None,
                 n_candidatas: int = 400, semilla: int = 0,
                 modo: str = "acotado"):
        # modo: "aprendido" (crudo) | "acotado" (techo) | "hibrido" (ganancias medidas)
        assert modo in ("aprendido", "acotado", "hibrido")
        self.modo = modo
        self.corto = res_corto          # salida de entrenar(horizonte=15)
        self.largo = res_largo          # salida de entrenar(horizonte=45)
        self.obj = objetivos or Objetivos()
        self.n = n_candidatas
        self.rng = np.random.default_rng(semilla)

    # -- candidatas ---------------------------------------------------------

    def _candidatas(self, actual: dict) -> list[dict]:
        cands = [dict(actual)]                      # "no tocar nada"
        for _ in range(self.n - 1):
            c = {}
            for a in ACCIONES:
                lo, hi, paso = LIMITES[a]
                c[a] = float(np.clip(actual[a] + self.rng.uniform(-paso, paso), lo, hi))
            cands.append(c)
        return cands

    # -- prediccion con un juego de modelos --------------------------------

    @staticmethod
    def _predecir(res: dict, fila: pd.Series, cands: list[dict],
                  actual: dict, base_T: float) -> pd.DataFrame:
        M = pd.DataFrame([fila.values] * len(cands), columns=fila.index)
        for a in ACCIONES:
            v = np.array([c[a] for c in cands])
            M[f"{a}__fut_media"] = v
            M[f"{a}__fut_delta"] = v - actual[a]
        M = M[res["columnas"]].astype(float)

        out = pd.DataFrame(index=range(len(cands)))
        for nombre, mod in res["modelos"].items():
            p = mod.predict(M)
            if nombre == "T_pasta_s":      # el modelo devuelve el incremento
                p = p + base_T
            out[nombre] = p
        return out

    # -- API ----------------------------------------------------------------

    def recomendar(self, i, X_corto: pd.DataFrame, X_largo: pd.DataFrame,
                   df: pd.DataFrame) -> dict:
        actual = {a: float(df.loc[i, a]) for a in ACCIONES}
        base_T = float(df.loc[i, "T_pasta_s"])
        cands = self._candidatas(actual)

        pc = self._predecir(self.corto, X_corto.loc[i], cands, actual, base_T)
        pl = self._predecir(self.largo, X_largo.loc[i], cands, actual, base_T)

        if self.modo == "acotado":
            pc = acotar_por_ganancia(pc, pc.iloc[0], cands, actual, self.corto["horizonte"])
            pl = acotar_por_ganancia(pl, pl.iloc[0], cands, actual, self.largo["horizonte"])
        elif self.modo == "hibrido":
            pc = aplicar_ganancias(pc, pc.iloc[0], cands, actual, self.corto["horizonte"])
            pl = aplicar_ganancias(pl, pl.iloc[0], cands, actual, self.largo["horizonte"])

        J = [coste_doble(pc.iloc[k].to_dict(), pl.iloc[k].to_dict(),
                         cands[k], actual, self.obj) for k in range(len(cands))]
        mejor = int(np.argmin(J))
        return dict(actual=actual, accion=cands[mejor],
                    corto_sin=pc.iloc[0].to_dict(), corto_rec=pc.iloc[mejor].to_dict(),
                    largo_sin=pl.iloc[0].to_dict(), largo_rec=pl.iloc[mejor].to_dict(),
                    dJ=float(J[0] - J[mejor]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="datos/termobatidora_exc.csv")
    ap.add_argument("--h-corto", type=int, default=15)
    ap.add_argument("--h-largo", type=int, default=45)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--modo", default="hibrido",
                    choices=["aprendido", "acotado", "hibrido"],
                    help="aprendido: modelo crudo | acotado: techo por ganancias | "
                         "hibrido: linea base del modelo + ganancias medidas")
    args = ap.parse_args()

    print(f"--- modelos de horizonte corto ({args.h_corto} min) ---")
    corto = entrenar(args.csv, horizonte=args.h_corto)
    print(f"\n--- modelos de horizonte largo ({args.h_largo} min) ---")
    largo = entrenar(args.csv, horizonte=args.h_largo)

    df = corto["df"]
    Xc = corto["X"]
    Xl = construir_caracteristicas(df, args.h_largo)
    # instantes validos para los dos horizontes a la vez
    test = corto["test"] & largo["test"] & Xl.notna().all(axis=1)
    idx = df.index[test]
    idx = idx[:: max(1, len(idx) // args.n)][: args.n]

    presc = PrescriptorDoble(corto, largo, modo=args.modo)

    filas = []
    for i in idx:
        r = presc.recomendar(i, Xc, Xl, df)
        a, n = r["actual"], r["accion"]
        filas.append(dict(
            hora=str(df.loc[i, "ts"])[11:16],
            T_agua=f"{a['T_agua_ida']:.0f}->{n['T_agua_ida']:.0f}",
            Q_agua=f"{a['caudal_agua']:.1f}->{n['caudal_agua']:.1f}",
            Q_pasta=f"{a['caudal_pasta']:.0f}->{n['caudal_pasta']:.0f}",
            agua_pr=f"{a['ratio_agua_proceso']:.3f}->{n['ratio_agua_proceso']:.3f}",
            T15_sin=round(r["corto_sin"]["T_pasta_s"], 1),
            T15_rec=round(r["corto_rec"]["T_pasta_s"], 1),
            grasa45_sin=round(r["largo_sin"]["grasa_alpeorujo"], 2),
            grasa45_rec=round(r["largo_rec"]["grasa_alpeorujo"], 2),
            hum45_rec=round(r["largo_rec"]["humedad_alpeorujo"], 1),
            polif45=round(r["largo_rec"]["polifenoles"]),
            dJ=round(r["dJ"], 2),
        ))

    d = pd.DataFrame(filas)
    print("\n" + "=" * 118)
    print("PRESCRIPCION DE DOBLE HORIZONTE")
    print(f"  restricciones de calidad a {args.h_corto} min  |  "
          f"agotamiento y humedad a {args.h_largo} min")
    print("=" * 118)
    print(d.to_string(index=False))
    print(f"\nReduccion media de grasa prevista a {args.h_largo} min: "
          f"{(d.grasa45_sin - d.grasa45_rec).mean():+.2f} puntos porcentuales")
    print(f"Correccion media de temperatura a {args.h_corto} min: "
          f"{(d.T15_rec - d.T15_sin).mean():+.2f} C")
    nota = {"aprendido": "modelo crudo: el optimizador puede explotar su error",
            "acotado":   "techo por ganancias medidas (el optimizador se pega al techo)",
            "hibrido":   "linea base del modelo + ganancias medidas en ensayo de escalon"}
    print(f"\nModo: {args.modo}  --  {nota[args.modo]}")


if __name__ == "__main__":
    main()
