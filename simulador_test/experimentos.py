"""
Experimentos de diagnostico.
============================

Tres preguntas que conviene responder ANTES de invertir en el proyecto real:

  1. ESCALADO      ¿cuantos dias de historico hacen falta?
  2. OBSERVABILIDAD ¿que aporta la analitica de recepcion de la partida?
  3. CAUSALIDAD    ¿la sensibilidad que aprende el modelo coincide con la real?
                   (sin esto la prescripcion es humo, aunque el MAE sea bueno)

Uso:  python experimentos.py [--dias 15]
"""

from __future__ import annotations

import argparse
import copy
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import modelo_15min as M
from generar_dataset import generar
from simulador_termobatidora import Termobatidora, ParamsOperacion

warnings.filterwarnings("ignore")


def _ajustar(df, X, tr, te, horizonte):
    """Entrena los cuatro objetivos y devuelve modelos y metricas."""
    mods, met = {}, {}
    for nom, (col, online) in M.OBJETIVOS.items():
        y = df[col].shift(-horizonte)
        est = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                            max_depth=6, min_samples_leaf=25,
                                            random_state=0)
        if online:
            est.fit(X[tr], y[tr] - df.loc[tr, col])
            p = est.predict(X[te]) + df.loc[te, col].to_numpy()
        else:
            est.fit(X[tr], y[tr])
            p = est.predict(X[te])
        mods[nom] = est
        met[nom] = (mean_absolute_error(y[te], p), r2_score(y[te], p))
    return mods, met


def _mascaras(df, X, horizonte, dias_tr, dias_te_desde):
    Y = pd.DataFrame({n: df[c].shift(-horizonte) for n, (c, _) in M.OBJETIVOS.items()})
    ok = M.mascara_valida(df, horizonte) & X.notna().all(1) & Y.notna().all(1)
    return (ok & (df.t_min < dias_tr * 1440),
            ok & (df.t_min >= dias_te_desde * 1440))


# ---------------------------------------------------------------------------
# 1. ¿Cuantos dias de historico?
# ---------------------------------------------------------------------------

def exp_escalado(dias=15, horizonte=15):
    print("\n" + "=" * 78)
    print("1. ESCALADO CON LA CANTIDAD DE HISTORICO  (test siempre en los ultimos 5 dias)")
    print("=" * 78)
    df = generar(dias=dias + 5, semilla=7, excitacion=1.0)
    X = M.construir_caracteristicas(df, horizonte)
    print(f"{'dias train':>11}{'filas':>8}" +
          "".join(f"{n:>26}" for n in ["T_pasta_s", "grasa_alpeorujo", "humedad_alpeorujo"]))
    print("-" * 91)
    for d in (2, 3, 5, 10, dias):
        tr, te = _mascaras(df, X, horizonte, d, dias)
        _, met = _ajustar(df, X, tr, te, horizonte)
        fila = f"{d:>11}{tr.sum():>8}"
        for n in ["T_pasta_s", "grasa_alpeorujo", "humedad_alpeorujo"]:
            mae, r2 = met[n]
            fila += f"{'MAE=%.2f R2=%.2f' % (mae, r2):>26}"
        print(fila)


# ---------------------------------------------------------------------------
# 2. ¿Aporta la analitica de recepcion?
# ---------------------------------------------------------------------------

def exp_recepcion(dias=15, horizonte=15):
    print("\n" + "=" * 78)
    print("2. APORTACION DE LA ANALITICA DE RECEPCION DE LA PARTIDA")
    print("   (el paper de 2003 no la usaba: 'not currently being measured')")
    print("=" * 78)
    df = generar(dias=dias + 5, semilla=7, excitacion=1.0)
    for usar in (False, True):
        X = M.construir_caracteristicas(df, horizonte, usar_recepcion=usar)
        tr, te = _mascaras(df, X, horizonte, dias, dias)
        _, met = _ajustar(df, X, tr, te, horizonte)
        etq = "CON recepcion " if usar else "SIN recepcion "
        print(etq + "  " + "   ".join(
            f"{n}: MAE={met[n][0]:6.2f} R2={met[n][1]:5.2f}"
            for n in ["grasa_alpeorujo", "humedad_alpeorujo"]))


# ---------------------------------------------------------------------------
# 3. Sensibilidad causal: modelo vs simulador
# ---------------------------------------------------------------------------

def respuesta_real(dT_agua=0.0, d_wpasta=0.0, d_wagua=0.0,
                   horizontes=(15, 30, 45, 60), semilla=3):
    """
    Escalon en bucle abierto sobre el simulador (Kp=0 congela el PI).
    Devuelve la respuesta REAL, que es la referencia contra la que hay que
    comparar lo que el modelo cree haber aprendido.
    """
    O = dict(p_parada=0, p_cambio_Tsp=0, p_cambio_wagua=0, p_cambio_wpasta=0,
             p_cambio_agua_proc=0, p_manual=0, Kp=0.0)
    a = Termobatidora(oper=ParamsOperacion(**O), semilla=semilla)
    a.simular(dias=200 / 1440)                       # estabilizar
    b = copy.deepcopy(a)
    b.integral += dT_agua
    b.w_pasta += d_wpasta
    b.w_agua += d_wagua
    ra = a.simular(dias=max(horizontes) / 1440)
    rb = b.simular(dias=max(horizontes) / 1440)
    return {h: (rb[h - 1]["real_T_pasta_s"] - ra[h - 1]["real_T_pasta_s"],
                rb[h - 1]["real_grasa_alpeorujo"] - ra[h - 1]["real_grasa_alpeorujo"])
            for h in horizontes}


def exp_causalidad(dias=15):
    print("\n" + "=" * 78)
    print("3. SENSIBILIDAD CAUSAL: lo que aprende el modelo vs la realidad")
    print("=" * 78)

    escalones = [("T_agua_ida", 5.0, dict(dT_agua=5.0)),
                 ("caudal_pasta", 1000.0, dict(d_wpasta=1000.0)),
                 ("caudal_agua", 2.0, dict(d_wagua=2.0))]

    print("\nRespuesta REAL del proceso (escalon en bucle abierto):")
    print(f"{'escalon':<22}" + "".join(f"{'t+%d min' % h:>22}" for h in (15, 30, 45, 60)))
    reales = {}
    for nom, mag, kw in escalones:
        r = respuesta_real(**kw)
        reales[nom] = r
        print(f"{nom + ' +' + str(mag):<22}" +
              "".join(f"{'dT=%+.2f dG=%+.2f' % r[h]:>22}" for h in (15, 30, 45, 60)))
    print("\n  Nota: la grasa apenas se mueve a 15 min. El tiempo de residencia es de")
    print("  ~50 min, asi que la pasta que sale ahora entro antes de la accion.")

    df = generar(dias=dias + 5, semilla=7, excitacion=1.0)
    for horizonte in (15, 45):
        X = M.construir_caracteristicas(df, horizonte)
        tr, te = _mascaras(df, X, horizonte, dias, dias)
        mods, met = _ajustar(df, X, tr, te, horizonte)
        F, bT = X[te], df.loc[te, "T_pasta_s"].to_numpy()
        base = {n: m.predict(F) + (bT if n == "T_pasta_s" else 0) for n, m in mods.items()}

        print(f"\nModelo con horizonte {horizonte} min "
              f"(grasa: MAE={met['grasa_alpeorujo'][0]:.2f}, "
              f"R2={met['grasa_alpeorujo'][1]:.2f})")
        print(f"{'escalon':<22}{'dT modelo':>12}{'dT real':>10}"
              f"{'dGrasa modelo':>16}{'dGrasa real':>14}")
        print("-" * 74)
        for nom, mag, _ in escalones:
            Fp = F.copy()
            for suf in ("__fut_media", "__fut_delta"):
                Fp[nom + suf] = Fp[nom + suf] + mag
            dT = float(np.mean(mods["T_pasta_s"].predict(Fp) + bT - base["T_pasta_s"]))
            dg = float(np.mean(mods["grasa_alpeorujo"].predict(Fp) - base["grasa_alpeorujo"]))
            rT, rg = reales[nom][horizonte]
            print(f"{nom + ' +' + str(mag):<22}{dT:>12.3f}{rT:>10.3f}{dg:>16.3f}{rg:>14.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", type=int, default=15,
                    help="dias de entrenamiento en los experimentos 2 y 3")
    ap.add_argument("--solo", choices=["escalado", "recepcion", "causalidad"])
    args = ap.parse_args()

    if args.solo in (None, "escalado"):
        exp_escalado(args.dias)
    if args.solo in (None, "recepcion"):
        exp_recepcion(args.dias)
    if args.solo in (None, "causalidad"):
        exp_causalidad(args.dias)


if __name__ == "__main__":
    main()
