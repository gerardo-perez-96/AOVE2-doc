"""
Genera el dataset de trabajo a partir del simulador.

Salida: datos/termobatidora.csv  (1 muestra por minuto)

Columnas:
  - Bloque ONLINE  : lo que existiria en el historico del SCADA.
  - Bloque LAB     : analitica de laboratorio (cada 2 h, con 90 min de retardo).
  - Bloque real_*  : verdad de campo del simulador. NO existe en planta;
                     se usa solo para entrenar/evaluar en este prototipo.
"""

from pathlib import Path
import argparse
import pandas as pd

from simulador_termobatidora import Termobatidora, ParamsOperacion

COLS_ONLINE = [
    "T_pasta_e", "T_pasta_1", "T_pasta_2", "T_pasta_s",
    "T_agua_ida", "T_agua_ret", "caudal_pasta", "caudal_agua",
    "agua_proceso", "T_ambiente", "amperios_batidora", "nivel_batidora",
    "sp_T_pasta", "sp_T_agua", "modo_auto", "en_marcha",
]


def generar(dias: float = 3.0, semilla: int = 7, dt_s: float = 10.0,
            excitacion: float = 0.0) -> pd.DataFrame:
    oper = ParamsOperacion(excitacion=excitacion)
    sim = Termobatidora(oper=oper, dt_s=dt_s, semilla=semilla)
    df = pd.DataFrame(sim.simular(dias=dias))

    # marca de tiempo: campana de molienda, arranque el 15-dic a las 06:00
    df.insert(0, "ts", pd.Timestamp("2024-12-15 06:00") + pd.to_timedelta(df["t_min"], unit="m"))

    # variables derivadas utiles y directamente calculables en planta
    df["ratio_agua_proceso"] = (df["agua_proceso"] / df["caudal_pasta"].clip(lower=1)).round(4)
    df["t_residencia_est"] = (6000.0 / (df["caudal_pasta"] / 60.0).clip(lower=1)).round(2)
    df["salto_termico"] = (df["T_pasta_s"] - df["T_pasta_e"]).round(3)
    df["dT_agua"] = (df["T_agua_ida"] - df["T_agua_ret"]).round(3)
    # potencia intercambiada estimada (kW) a partir del circuito de agua
    df["Q_agua_kW"] = (df["caudal_agua"] * 1000 / 3600 * 4.18 * df["dT_agua"]).round(2)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", type=float, default=3.0)
    ap.add_argument("--semilla", type=int, default=7)
    ap.add_argument("--excitacion", type=float, default=0.0,
                    help="0 = operacion normal, 1 = campana de identificacion")
    ap.add_argument("--salida", type=str, default="datos/termobatidora.csv")
    args = ap.parse_args()

    df = generar(args.dias, args.semilla, excitacion=args.excitacion)
    out = Path(args.salida)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    en_marcha = df["en_marcha"] == 1
    print(f"Filas: {len(df)}  ({args.dias} dias a 1 muestra/min)")
    print(f"En marcha: {en_marcha.mean():.1%}   Paradas: {(~en_marcha).sum()} min")
    print(f"Analiticas de laboratorio disponibles: "
          f"{df['lab_grasa_alpeorujo'].notna().sum()} filas con valor")
    print(f"\nGuardado en {out.resolve()}")
    print(df.loc[en_marcha, ["T_pasta_s", "caudal_pasta", "caudal_agua",
                             "real_grasa_alpeorujo", "real_humedad_alpeorujo"]]
          .describe().round(2).to_string())


if __name__ == "__main__":
    main()
