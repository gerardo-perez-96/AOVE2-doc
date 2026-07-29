"""
Simulador de la termobatidora (batidora térmica) de una almazara continua.
=========================================================================

Basado en la descripcion de proceso de Bordons & Zafra (2003), "Inferential
sensor for the olive oil industry", y en balances de energia/materia clasicos
de una batidora encamisada de N cuerpos en serie seguida de decanter.

Del articulo se toman:
  - Variables medibles en planta: caudal y temperatura de pasta, caudal y
    temperatura del agua.
  - Ordenes de magnitud de retardos (4.5-8 min) y tiempos caracteristicos
    (4-7 min)  ->  se reproducen de forma natural con la cadena de cuerpos
    (CSTR en serie) mas el retardo de transporte del decanter.
  - Rangos de las salidas de laboratorio:
        humedad del alpeorujo : 45.0 - 64.6 %
        grasa   del alpeorujo :  2.73 -  9.19 %
    El modelo esta calibrado para moverse dentro de esos rangos.

ESTRUCTURA DEL MODELO
---------------------
Estados (por cuerpo i = 1..N):
    Tp[i]  temperatura de la pasta            [degC]
    Tj[i]  temperatura del agua de la camisa  [degC]
    C[i]   fraccion de aceite coalescido      [0-1]   (aceite "libre")
    D[i]   dosis termica acumulada            [min]   (proxy de perdida de calidad)

Balances:
    M*cp * dTp/dt = w_p*cp*(Tp[i-1]-Tp[i]) + UA*(Tj[i]-Tp[i]) + P_bat
    Mj*cpw* dTj/dt = w_w*cpw*(Tw_in-Tj[i]) - UA*(Tj[i]-Tp[i]) - UAamb*(Tj[i]-Tamb)
    dC/dt = (C[i-1]-C[i])/tau_i + k_coal(Tp,H,IM)*(1-C[i])
    dD/dt = (D[i-1]-D[i])/tau_i + exp((Tp[i]-27)/6)

El aceite se recupera en el decanter con rendimiento
    eta = eta_max * C_salida * f_agua(humedad efectiva)
y de ahi salen, por balance de materia, la grasa y la humedad del alpeorujo.

Autor: borrador de trabajo para prototipado de modelos predictivos/prescriptivos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# 1. Parametros
# ---------------------------------------------------------------------------

@dataclass
class ParamsPlanta:
    """Parametros fisicos de la batidora y del decanter."""

    # --- Batidora -----------------------------------------------------------
    n_cuerpos: int = 3
    masa_cuerpo: float = 2000.0        # kg de pasta retenidos por cuerpo
    cp_pasta_a: float = 1.70           # cp = a + b*H  [kJ/(kg K)], H en %
    cp_pasta_b: float = 0.024
    UA0: float = 0.90                  # kW/K por cuerpo a caudal de agua nominal
    UA_exp: float = 0.6                # UA ~ (w_agua/w_nom)^UA_exp
    w_agua_nom: float = 5.0            # m3/h (caudal nominal por la camisa)
    masa_camisa: float = 250.0         # kg de agua retenidos por camisa
    cp_agua: float = 4.18              # kJ/(kg K)
    P_batido: float = 2.5              # kW disipados por el eje en cada cuerpo
    UA_amb: float = 0.25               # kW/K de perdidas de la pasta al ambiente
    UA_amb_camisa: float = 0.10        # kW/K de perdidas de la camisa al ambiente

    # --- Cinetica de coalescencia (formacion de aceite libre) ---------------
    k_coal0: float = 0.100             # 1/min a 25 degC y condiciones optimas
    k_coal_T: float = 8.0              # degC de "decada" termica (Arrhenius linealizado)
    H_opt: float = 50.0                # humedad optima de la pasta [%]
    H_sigma: float = 14.0              # anchura del optimo de humedad
    IM_opt: float = 4.0                # indice de madurez optimo (0-7)
    IM_pen: float = 0.22               # penalizacion por desviacion de madurez

    # --- Decanter -----------------------------------------------------------
    eta_max: float = 0.96              # rendimiento maximo de extraccion
    Heff_opt: float = 0.545            # humedad efectiva optima a la entrada del decanter
    Heff_sigma: float = 0.19
    frac_agua_orujo: float = 0.72      # fraccion del agua que queda en el orujo (resto: alpechin)
    retardo_decanter_min: float = 5.0  # retardo de transporte batidora -> muestra alpeorujo

    # --- Calidad ------------------------------------------------------------
    dosis_ref: float = 60.0            # min-equivalentes a 27 degC para perder ~37% de polifenoles
    polifenoles_0: float = 420.0       # mg/kg en el aceite "sin dano termico"


@dataclass
class ParamsOperacion:
    """Consignas, limites y comportamiento del operario / lazos de control."""

    # Limites de las variables manipuladas
    Tagua_min: float = 30.0
    Tagua_max: float = 68.0
    wagua_min: float = 2.5
    wagua_max: float = 9.0
    wpasta_min: float = 4500.0
    wpasta_max: float = 9500.0
    agua_proc_min: float = 0.00        # kg agua / kg pasta
    agua_proc_max: float = 0.15

    # Lazo PI de temperatura de pasta (manipula la consigna de agua caliente)
    Kp: float = 3.5
    Ti: float = 12.0                   # min
    tau_caldera: float = 3.0           # min, retardo de primer orden del circuito de agua

    # Consigna de temperatura de pasta
    Tsp_base: float = 28.0
    Tsp_min: float = 25.0
    Tsp_max: float = 32.0

    # Probabilidades por minuto de que el operario toque algo
    p_cambio_Tsp: float = 1 / 180
    p_cambio_wagua: float = 1 / 240
    p_cambio_wpasta: float = 1 / 150
    p_cambio_agua_proc: float = 1 / 200
    p_manual: float = 1 / 600          # pasa el lazo a manual un rato (dato "rico")
    dur_manual: int = 45               # min en manual

    # Paradas (limpieza, cambio de partida, averia)
    p_parada: float = 1 / 900
    dur_parada: tuple = (12, 45)       # min

    # --- Plan de excitacion (PRBS superpuesto a la operacion normal) -------
    # Sin excitacion, las acciones futuras son funcion determinista del estado
    # (el PI las decide), el modelo no puede separar causa de correlacion y las
    # recomendaciones salen planas. En planta esto se traduce en una campana de
    # identificacion: escalones deliberados durante unos turnos.
    excitacion: float = 0.0            # 0 = ninguna, 1 = campana de identificacion
    exc_dur: tuple = (10, 30)          # min que dura cada escalon
    exc_pausa: tuple = (15, 45)        # min entre escalones
    exc_amp_Tagua: float = 6.0         # degC
    exc_amp_wagua: float = 1.8         # m3/h
    exc_amp_wpasta: float = 900.0      # kg/h

    # Laboratorio
    periodo_lab_min: int = 120
    retardo_lab_min: int = 90
    sigma_lab_grasa: float = 0.25
    sigma_lab_humedad: float = 0.9


# ---------------------------------------------------------------------------
# 2. Materia prima (perturbacion no medida principal)
# ---------------------------------------------------------------------------

@dataclass
class Partida:
    """Una partida de aceituna que entra a la linea."""
    indice_madurez: float      # 0 (verde) - 7 (sobremadura)
    humedad_pasta: float       # % de agua en la pasta que sale del molino
    grasa_pasta: float         # % de grasa sobre pasta humeda
    dificultad: float          # 1.0 = normal; >1 pasta dificil (emulsion)
    temp_entrada: float        # degC de la pasta a la salida del molino
    minutos_restantes: int
    # --- datos de RECEPCION: medidos al descargar la partida (NIR / Abencor).
    #     Son estimaciones con error, disponibles antes de moler.
    rec_humedad: float = 0.0
    rec_grasa: float = 0.0
    rec_madurez: float = 0.0
    partida_id: int = 0


def nueva_partida(rng: np.random.Generator, Tamb: float) -> Partida:
    im = float(np.clip(rng.normal(3.8, 1.5), 0.5, 6.8))
    # aceituna mas madura -> menos agua y algo mas de grasa
    hum = float(np.clip(rng.normal(52.0 - 1.4 * (im - 3.8), 3.2), 41.0, 60.0))
    grasa = float(np.clip(rng.normal(20.0 + 0.9 * (im - 3.8), 1.6), 15.0, 25.0))
    # atrojada o muy verde -> pastas dificiles
    dif = float(np.clip(1.0 + 0.10 * abs(im - 3.8) + rng.normal(0, 0.07), 0.85, 1.5))
    tin = float(np.clip(Tamb + rng.normal(6.0, 1.5), 8.0, 30.0))
    dur = int(rng.integers(40, 150))
    p = Partida(im, hum, grasa, dif, tin, dur)
    # la analitica de recepcion mide la aceituna, no la pasta: hay sesgo y ruido
    p.rec_humedad = float(hum + rng.normal(0.0, 1.3))
    p.rec_grasa = float(grasa + rng.normal(0.0, 0.7))
    p.rec_madurez = float(np.clip(im + rng.normal(0.0, 0.5), 0.0, 7.0))
    return p


# ---------------------------------------------------------------------------
# 3. Simulador
# ---------------------------------------------------------------------------

class Termobatidora:
    """Integra el modelo y devuelve un registro tipo SCADA (1 muestra/minuto)."""

    def __init__(self,
                 params: ParamsPlanta | None = None,
                 oper: ParamsOperacion | None = None,
                 dt_s: float = 10.0,
                 semilla: int = 7):
        self.p = params or ParamsPlanta()
        self.o = oper or ParamsOperacion()
        self.dt = dt_s / 60.0                      # paso de integracion en minutos
        self.rng = np.random.default_rng(semilla)

        n = self.p.n_cuerpos
        self.Tp = np.full(n, 24.0)
        self.Tj = np.full(n, 45.0)
        self.C = np.full(n, 0.30)
        self.D = np.zeros(n)

        # Variables manipuladas / consignas
        self.Tsp = self.o.Tsp_base
        self.Tagua_sp = 50.0
        self.Tagua = 50.0                          # temperatura real (lag de caldera)
        self.w_agua = self.p.w_agua_nom
        self.w_pasta = 7000.0                      # kg/h
        self.agua_proc = 0.06                      # kg agua / kg pasta
        self.integral = 45.0
        self.modo_auto = True
        self.min_manual = 0
        self.min_parada = 0
        self.exc = dict(Tagua=0.0, wagua=0.0, wpasta=0.0)
        self.min_exc = 0
        self.min_pausa_exc = 0
        self._u_auto = 50.0

        self.Tamb = 12.0
        self._n_partidas = 1
        self.partida = nueva_partida(self.rng, self.Tamb)
        self.partida.partida_id = 1

        # Buffers de retardo de transporte hasta el punto de muestreo del alpeorujo
        self._buffer = []
        self._nbuf = max(1, int(round(self.p.retardo_decanter_min / self.dt)))

        # Derivas lentas de sensores
        self.drift_T = 0.0
        self.t = 0.0                               # minutos simulados

    # -- utilidades ---------------------------------------------------------

    def _cp_pasta(self, H: float) -> float:
        return self.p.cp_pasta_a + self.p.cp_pasta_b * H

    def _k_coal(self, T: float, H: float, im: float, dif: float) -> float:
        p = self.p
        f_T = np.exp((T - 25.0) / p.k_coal_T)
        f_H = np.exp(-((H - p.H_opt) / p.H_sigma) ** 2)
        f_IM = max(0.35, 1.0 - p.IM_pen * abs(im - p.IM_opt) / 4.0)
        return p.k_coal0 * f_T * f_H * f_IM / dif

    # -- decisiones del operario -------------------------------------------

    def _operario(self):
        o, rng = self.o, self.rng

        if self.min_parada > 0:
            self.min_parada -= 1
            self.w_pasta = 0.0
            return
        if rng.random() < o.p_parada:
            self.min_parada = int(rng.integers(*o.dur_parada))
            self.w_pasta = 0.0
            return

        if self.min_exc > 0 and self.o.excitacion > 0:
            # escalon de identificacion: lazo en MANUAL, el PI no compensa
            self.modo_auto = False
        elif self.min_manual > 0:
            self.min_manual -= 1
            self.modo_auto = False
        else:
            self.modo_auto = True
            if rng.random() < o.p_manual:
                self.min_manual = o.dur_manual
                # en manual el operario fija el agua a un valor arbitrario
                self.Tagua_sp = float(rng.uniform(38.0, 62.0))

        # ---- plan de excitacion ------------------------------------------
        if o.excitacion > 0:
            if self.min_exc > 0:
                self.min_exc -= 1
            elif self.min_pausa_exc > 0:
                self.min_pausa_exc -= 1
                self.exc = dict(Tagua=0.0, wagua=0.0, wpasta=0.0)
            else:
                k = o.excitacion
                self.exc = dict(
                    Tagua=float(rng.choice([-1.0, 0.0, 1.0])) * o.exc_amp_Tagua * k,
                    wagua=float(rng.choice([-1.0, 0.0, 1.0])) * o.exc_amp_wagua * k,
                    wpasta=float(rng.choice([-1.0, 0.0, 1.0])) * o.exc_amp_wpasta * k,
                )
                self.min_exc = int(rng.integers(*o.exc_dur))
                self.min_pausa_exc = int(rng.integers(*o.exc_pausa))

        if rng.random() < o.p_cambio_Tsp:
            self.Tsp = float(np.clip(self.Tsp + rng.normal(0, 1.6), o.Tsp_min, o.Tsp_max))
        if rng.random() < o.p_cambio_wagua:
            self.w_agua = float(np.clip(0.8 * self.w_agua + 0.2 * self.p.w_agua_nom
                                        + rng.normal(0, 1.2), o.wagua_min, o.wagua_max))
        if rng.random() < o.p_cambio_wpasta:
            nuevo = self.w_pasta if self.w_pasta > 0 else 7000.0
            self.w_pasta = float(np.clip(0.8 * nuevo + 0.2 * 7000.0 + rng.normal(0, 900.0),
                                         o.wpasta_min, o.wpasta_max))
        if rng.random() < o.p_cambio_agua_proc:
            self.agua_proc = float(np.clip(self.agua_proc + rng.normal(0, 0.03),
                                           o.agua_proc_min, o.agua_proc_max))

        if self.w_pasta == 0.0 and self.min_parada == 0:
            self.w_pasta = float(rng.uniform(5500, 8500))

    @property
    def w_agua_ef(self) -> float:
        if self.w_pasta == 0.0:
            return self.w_agua
        return float(np.clip(self.w_agua + self.exc["wagua"],
                             self.o.wagua_min, self.o.wagua_max))

    @property
    def w_pasta_ef(self) -> float:
        if self.w_pasta == 0.0:
            return 0.0
        return float(np.clip(self.w_pasta + self.exc["wpasta"],
                             self.o.wpasta_min, self.o.wpasta_max))

    def _pi_temperatura(self, dt_min: float):
        """PI que ajusta la consigna de agua caliente para llevar Tp_salida a Tsp."""
        if self.min_exc > 0 and self.o.excitacion > 0:
            # bucle abierto: se mantiene la ultima salida del PI mas el escalon
            self.Tagua_sp = float(np.clip(self._u_auto + self.exc["Tagua"],
                                          self.o.Tagua_min, self.o.Tagua_max))
            return
        if not self.modo_auto or self.w_pasta == 0.0:
            return
        e = self.Tsp - self.Tp[-1]
        u = self.o.Kp * e + self.integral
        Tagua_sp = float(np.clip(u, self.o.Tagua_min, self.o.Tagua_max))
        # accion integral con anti-windup por retro-calculo
        if self.o.Tagua_min < u < self.o.Tagua_max:
            self.integral += (self.o.Kp / self.o.Ti) * e * dt_min
        self.integral = float(np.clip(self.integral, self.o.Tagua_min, self.o.Tagua_max))
        self._u_auto = Tagua_sp
        self.Tagua_sp = Tagua_sp

    # -- dinamica -----------------------------------------------------------

    def _paso(self, dt: float):
        p = self.p
        pa = self.partida
        H = pa.humedad_pasta
        cp = self._cp_pasta(H)
        n = p.n_cuerpos

        # circuito de agua caliente (retardo de caldera)
        self.Tagua += dt / p_safe(self.o.tau_caldera) * (self.Tagua_sp - self.Tagua)

        w_p = self.w_pasta_ef / 60.0                    # kg/min
        w_w = self.w_agua_ef * 1000.0 / 60.0            # kg/min (1 m3 ~ 1000 kg)
        UA = p.UA0 * (max(self.w_agua_ef, 0.1) / p.w_agua_nom) ** p.UA_exp   # kW/K

        # kW -> kJ/min
        UA_m = UA * 60.0
        P_bat_m = p.P_batido * 60.0
        UAamb_m = p.UA_amb * 60.0            # perdidas de la pasta
        UAamb_j = p.UA_amb_camisa * 60.0     # perdidas de la camisa

        Tp_new, Tj_new, C_new, D_new = self.Tp.copy(), self.Tj.copy(), self.C.copy(), self.D.copy()

        for i in range(n):
            Tp_ant = pa.temp_entrada if i == 0 else self.Tp[i - 1]
            C_ant = 0.05 if i == 0 else self.C[i - 1]
            D_ant = 0.0 if i == 0 else self.D[i - 1]

            # --- energia pasta ---
            conv = w_p * cp * (Tp_ant - self.Tp[i])
            trans = UA_m * (self.Tj[i] - self.Tp[i])
            perd = UAamb_m * (self.Tp[i] - self.Tamb)
            dTp = (conv + trans + P_bat_m - perd) / (p.masa_cuerpo * cp)
            Tp_new[i] = self.Tp[i] + dt * dTp

            # --- energia camisa ---
            dTj = (w_w * p.cp_agua * (self.Tagua - self.Tj[i])
                   - trans
                   - UAamb_j * (self.Tj[i] - self.Tamb)) / (p.masa_camisa * p.cp_agua)
            Tj_new[i] = self.Tj[i] + dt * dTj

            # --- coalescencia y dosis termica ---
            tau_i = p.masa_cuerpo / max(w_p, 1e-6)      # min de residencia en el cuerpo
            k = self._k_coal(self.Tp[i], H, pa.indice_madurez, pa.dificultad)
            dC = (C_ant - self.C[i]) / tau_i + k * (1.0 - self.C[i])
            C_new[i] = float(np.clip(self.C[i] + dt * dC, 0.0, 1.0))

            dD = (D_ant - self.D[i]) / tau_i + np.exp((self.Tp[i] - 27.0) / 6.0)
            D_new[i] = max(0.0, self.D[i] + dt * dD)

        self.Tp, self.Tj, self.C, self.D = Tp_new, Tj_new, C_new, D_new

    # -- salidas de proceso -------------------------------------------------

    def _salidas_alpeorujo(self, C_out: float, D_out: float):
        """Balance de materia del decanter -> grasa y humedad del alpeorujo."""
        p, pa = self.p, self.partida
        x_o = pa.grasa_pasta / 100.0
        x_w = pa.humedad_pasta / 100.0
        q = self.agua_proc

        H_eff = (x_w + q) / (1.0 + q)
        f_agua = np.exp(-((H_eff - p.Heff_opt) / p.Heff_sigma) ** 2)
        # caudal alto -> menos tiempo de residencia en el decanter
        f_caudal = 1.0 - 0.05 * max(0.0, (self.w_pasta_ef - 8000.0) / 1500.0)

        eta = float(np.clip(p.eta_max * C_out * f_agua * f_caudal, 0.05, 0.98))
        o_rec = x_o * eta                       # kg aceite recuperado / kg pasta

        # decanter de tres fases: parte del agua sale como alpechin
        f_ret = p.frac_agua_orujo * (1.0 - 0.35 * q / 0.15)
        f_ret = float(np.clip(f_ret, 0.40, 0.80))
        agua_orujo = (x_w + q) * f_ret
        x_s = max(0.0, 1.0 - x_o - x_w)         # solidos (hueso, pulpa seca)
        grasa_orujo = x_o * (1.0 - eta)
        m_alp = x_s + agua_orujo + grasa_orujo  # kg alpeorujo / kg pasta

        grasa_alp = 100.0 * grasa_orujo / m_alp
        humedad_alp = 100.0 * agua_orujo / m_alp

        polifenoles = p.polifenoles_0 * np.exp(-D_out / p.dosis_ref)
        rendimiento = 100.0 * o_rec             # kg aceite / 100 kg pasta

        return grasa_alp, humedad_alp, polifenoles, rendimiento, eta

    # -- bucle principal ----------------------------------------------------

    def simular(self, dias: float = 3.0) -> "list[dict]":
        minutos = int(dias * 24 * 60)
        n_sub = max(1, int(round(1.0 / self.dt)))
        registros = []

        lab_pend = []        # (minuto_disponible, grasa, humedad)
        lab_grasa = np.nan
        lab_humedad = np.nan

        for m in range(minutos):
            # temperatura ambiente: ciclo diario + ruido
            self.Tamb = 11.0 + 6.0 * np.sin(2 * np.pi * (m / 1440.0 - 0.25)) + self.rng.normal(0, 0.3)

            self.partida.minutos_restantes -= 1
            if self.partida.minutos_restantes <= 0:
                self._n_partidas += 1
                self.partida = nueva_partida(self.rng, self.Tamb)
                self.partida.partida_id = self._n_partidas

            self._operario()
            self._pi_temperatura(1.0)

            for _ in range(n_sub):
                self._paso(self.dt)
                self._buffer.append((float(self.C[-1]), float(self.D[-1])))
                if len(self._buffer) > self._nbuf:
                    self._buffer.pop(0)

            C_ret, D_ret = self._buffer[0]
            grasa, humedad, polif, rend, eta = self._salidas_alpeorujo(C_ret, D_ret)

            # --- laboratorio: muestra cada 2 h, resultado 90 min despues ----
            if m % self.o.periodo_lab_min == 0 and self.w_pasta > 0:
                lab_pend.append((m + self.o.retardo_lab_min,
                                 grasa + self.rng.normal(0, self.o.sigma_lab_grasa),
                                 humedad + self.rng.normal(0, self.o.sigma_lab_humedad)))
            while lab_pend and lab_pend[0][0] <= m:
                _, lab_grasa, lab_humedad = lab_pend.pop(0)

            # --- instrumentacion (con ruido y deriva) -----------------------
            self.drift_T += self.rng.normal(0, 0.002)
            self.drift_T = float(np.clip(self.drift_T, -0.4, 0.4))
            rn = self.rng.normal

            parado = self.w_pasta == 0.0
            # amperaje del motor de batido: proxy de viscosidad de la pasta
            if parado:
                amp = 4.0 + rn(0, 0.2)
            else:
                amp = (14.0
                       + 7.0 * np.exp(-(self.Tp[-1] - 24.0) / 11.0)
                       + 0.18 * (50.0 - self.partida.humedad_pasta)
                       + 3.0 * (self.w_pasta_ef / 7000.0 - 1.0)
                       + 4.0 * (self.partida.dificultad - 1.0)
                       + rn(0, 0.35))

            registros.append(dict(
                t_min=m,
                # ---------- variables medidas online (disponibles al modelo) ----------
                T_pasta_e=round(self.partida.temp_entrada + self.drift_T + rn(0, 0.20), 3),
                T_pasta_1=round(self.Tp[0] + self.drift_T + rn(0, 0.15), 3),
                T_pasta_2=round(self.Tp[1] + self.drift_T + rn(0, 0.15), 3),
                T_pasta_s=round(self.Tp[-1] + self.drift_T + rn(0, 0.15), 3),
                T_agua_ida=round(self.Tagua + rn(0, 0.25), 3),
                T_agua_ret=round(float(self.Tj.mean()) + rn(0, 0.25), 3),
                caudal_pasta=round(max(0.0, self.w_pasta_ef * (1 + rn(0, 0.012))), 1),
                caudal_agua=round(self.w_agua_ef * (1 + rn(0, 0.015)), 3),
                agua_proceso=round(self.agua_proc * max(self.w_pasta_ef, 0.0), 1),   # kg/h
                T_ambiente=round(self.Tamb, 2),
                amperios_batidora=round(amp, 2),
                nivel_batidora=round(0.0 if parado else 92.0 + rn(0, 1.2), 1),
                # ---------- datos de recepcion de la partida en curso ----------
                partida_id=self.partida.partida_id,
                rec_humedad_aceituna=round(self.partida.rec_humedad, 2),
                rec_grasa_aceituna=round(self.partida.rec_grasa, 2),
                rec_indice_madurez=round(self.partida.rec_madurez, 2),
                # ---------- consignas / variables manipuladas ----------
                sp_T_pasta=round(self.Tsp, 2),
                sp_T_agua=round(self.Tagua_sp, 2),
                modo_auto=int(self.modo_auto),
                en_marcha=int(not parado),
                # ---------- laboratorio (baja frecuencia, retardado) ----------
                lab_grasa_alpeorujo=None if np.isnan(lab_grasa) else round(lab_grasa, 3),
                lab_humedad_alpeorujo=None if np.isnan(lab_humedad) else round(lab_humedad, 3),
                # ---------- VERDAD DE CAMPO (no existe en planta real) ----------
                real_grasa_alpeorujo=round(grasa, 4),
                real_humedad_alpeorujo=round(humedad, 4),
                real_polifenoles=round(polif, 2),
                real_rendimiento=round(rend, 4),
                real_eta_extraccion=round(eta, 4),
                real_coalescencia=round(C_ret, 4),
                real_T_pasta_s=round(float(self.Tp[-1]), 4),
                real_humedad_pasta=round(self.partida.humedad_pasta, 3),
                real_grasa_pasta=round(self.partida.grasa_pasta, 3),
                real_indice_madurez=round(self.partida.indice_madurez, 3),
                real_dificultad=round(self.partida.dificultad, 3),
                real_t_residencia=round(min(999.0,
                    self.p.masa_cuerpo * self.p.n_cuerpos / max(self.w_pasta_ef / 60.0, 1e-3)), 2),
            ))
            self.t += 1.0

        return registros


def p_safe(x: float) -> float:
    return max(float(x), 1e-6)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    sim = Termobatidora(semilla=7)
    df = pd.DataFrame(sim.simular(dias=3))
    print(df[["T_pasta_s", "caudal_pasta", "caudal_agua", "sp_T_agua",
              "real_grasa_alpeorujo", "real_humedad_alpeorujo",
              "real_polifenoles", "real_t_residencia"]].describe().round(2))
