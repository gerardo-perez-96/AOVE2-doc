# Simulador de termobatidora + predicción a 15 min + prescripción

Prototipo para explorar algoritmos de predicción y prescripción en una almazara,
mientras no hay datos reales. Todo está calibrado sobre la descripción de proceso
de **Bordons & Zafra (2003)**, *Inferential sensor for the olive oil industry*.

```
generar_dataset.py  →  datos/*.csv  →  modelo_15min.py  →  prescriptor.py
       ↑                                      ↑
simulador_termobatidora.py              experimentos.py  (diagnóstico)
```

---

## 1. Qué se toma del artículo y qué se ha añadido

| Del artículo | Valor |
|---|---|
| Variables de entrada disponibles | caudal y temperatura de pasta, caudal y temperatura de agua |
| Retardos de transporte | 4,5 – 8 min |
| Tiempos característicos | 4 – 7 min |
| Rango de humedad del alpeorujo | 45,0 – 64,6 % |
| Rango de grasa del alpeorujo | 2,73 – 9,19 % |
| Temperatura de batido | ~35 °C, con límite superior por calidad |
| Necesidad de filtrar paradas y limpiezas antes de entrenar | sí, explícito |

Añadido porque el artículo no lo da: balances de energía y materia, cinética de
coalescencia, decánter de tres fases y un índice de calidad (polifenoles). Los
parámetros están ajustados para que las salidas caigan **dentro de los rangos
publicados**, que es la única calibración objetiva disponible.

## 2. El simulador

Batidora de **3 cuerpos encamisados en serie** (2000 kg de pasta cada uno) +
decánter. Estados por cuerpo:

| Estado | Ecuación |
|---|---|
| `Tp[i]` temperatura de pasta | `M·cp·dTp/dt = w_p·cp·(Tp[i-1]−Tp[i]) + UA·(Tj[i]−Tp[i]) + P_batido − pérdidas` |
| `Tj[i]` temperatura de camisa | `Mj·cpw·dTj/dt = w_w·cpw·(Tagua−Tj[i]) − UA·(Tj[i]−Tp[i]) − pérdidas` |
| `C[i]` fracción de aceite coalescido | `dC/dt = (C[i-1]−C[i])/τ + k(T,H,IM)·(1−C[i])` |
| `D[i]` dosis térmica | `dD/dt = (D[i-1]−D[i])/τ + exp((Tp−27)/6)` |

`k` crece con la temperatura (tipo Arrhenius), tiene un óptimo de humedad de pasta
en 50 % y penaliza aceitunas muy verdes o atrojadas. El decánter cierra el balance:

```
η = η_max · C_salida · f(humedad efectiva) · f(caudal)
grasa_alpeorujo   = grasa_pasta·(1−η) / masa_alpeorujo
humedad_alpeorujo = agua_retenida    / masa_alpeorujo
```

Los rangos del artículo salen **del balance**, no de un ajuste ad hoc.

El simulador incluye además: partidas de aceituna que cambian cada 40–150 min
(perturbación no medida principal), ciclo diario de temperatura ambiente, lazo PI
de temperatura con anti-windup, cambios manuales del operario, paradas de limpieza,
ruido y deriva de sensores, y analítica de laboratorio cada 2 h con 90 min de retardo.

### Dataset generado

```bash
python generar_dataset.py --dias 3                       # operación normal
python generar_dataset.py --dias 3 --excitacion 1.0 \
       --salida datos/termobatidora_exc.csv              # campaña de identificación
```

Una muestra por minuto. Tres bloques de columnas:

- **online** — lo que existiría en el histórico del SCADA (temperaturas, caudales,
  amperaje de batido, nivel, consignas).
- **lab_\*** — analítica de laboratorio, con su periodicidad y su retardo.
- **real_\*** — verdad de campo del simulador. **No existe en planta**; sirve solo
  para entrenar y evaluar en este prototipo.

## 3. El modelo de 15 minutos

```
y(t+15) = f( histórico hasta t ,  acciones aplicadas en (t, t+15] )
```

Incluir las **acciones futuras** como entrada es lo que convierte el predictor en
prescriptor: el optimizador evalúa "¿y si subo el agua a 52 °C?" cambiando solo
ese bloque de columnas. Es el modelo interno de un MPC, aprendido de datos.

Dos tratamientos según el objetivo:

- **Medible online** (`T_pasta_s`): se modela el **incremento** `y(t+15) − y(t)`.
  Bate a la persistencia porque el modelo solo tiene que explicar la deriva.
- **No medible online** (grasa, humedad, polifenoles): es el sensor inferencial
  del artículo. Se modela el nivel, con línea base "último resultado de laboratorio".

Retardos: 0, 3, 6, 10, 15 y 30 min; medias móviles de 15, 30 y 60 min; pendiente
a 15 min. Modelo por defecto `HistGradientBoostingRegressor` (`--tipo ridge` para
la línea base lineal). Partición **temporal**, nunca aleatoria.

```bash
python modelo_15min.py --csv datos/termobatidora_exc.csv
python prescriptor.py  --csv datos/termobatidora_exc.csv --horizonte 15
```

## 4. Resultados y, sobre todo, los tres hallazgos

`python experimentos.py --dias 15` reproduce los tres.

### a) Con 3 días se predice la temperatura, no el rendimiento

| días de train | T_pasta_s | grasa_alpeorujo | humedad_alpeorujo |
|---|---|---|---|
| 2  | MAE 0,34 · R² 0,94 | MAE 1,26 · R² 0,26 | MAE 3,50 · R² 0,12 |
| 5  | MAE 0,25 · R² 0,97 | MAE 1,20 · R² 0,31 | MAE 3,23 · R² 0,25 |
| 10 | MAE 0,19 · R² 0,98 | MAE 1,08 · R² 0,39 | MAE 2,68 · R² 0,45 |
| 15 | MAE 0,17 · R² 0,98 | MAE 0,95 · R² 0,47 | MAE 2,65 · R² 0,45 |

La temperatura satura enseguida. Grasa y humedad siguen mejorando: el cuello de
botella no es el algoritmo, es el **número de partidas distintas vistas**. Dos días
son ~30 partidas.

### b) La analítica de recepción es el mayor salto de calidad

Sin ella, la humedad y la grasa de la aceituna entrante son perturbaciones
invisibles. Con 15 días de entrenamiento:

| | grasa_alpeorujo | humedad_alpeorujo |
|---|---|---|
| sin recepción | MAE 1,09 · R² 0,35 | MAE 3,53 · R² 0,13 |
| con recepción | MAE 0,95 · R² 0,47 | MAE 2,65 · R² 0,45 |

Es exactamente lo que anticipa el artículo cuando dice que no incluyeron esas
variables porque no se medían, pero que hacerlo *"could only make our results improve"*.
Hoy casi cualquier almazara mide grasa y humedad en el patio para el pago al agricultor.

### c) Sin excitación no hay prescripción, aunque el MAE sea bueno

Con el lazo PI cerrado, la temperatura del agua es función determinista del estado
pasado. El modelo no puede separar causa de correlación: predice bien y aun así
responde a las acciones con sensibilidad casi nula (+5 °C de agua → −0,006 pp de grasa).

La solución es la misma que en identificación clásica: **escalones deliberados con
el lazo en manual**. Con ellos, la sensibilidad aprendida se acerca a la real:

| escalón (H = 15 min) | ΔT modelo | ΔT real | Δgrasa modelo | Δgrasa real |
|---|---|---|---|---|
| T_agua +5 °C | +0,308 | +0,357 | +0,017 | −0,023 |
| caudal_pasta +1000 kg/h | −0,232 | −0,306 | +0,266 | +0,383 |
| caudal_agua +2 m³/h | +0,344 | +0,636 | −0,007 | −0,078 |

**Métrica de aceptación propuesta para el proyecto real:** que la sensibilidad del
modelo a cada acción tenga el signo correcto y al menos el 70 % de la magnitud
medida en un ensayo de escalón. Un R² alto sin esto no autoriza a prescribir nada.

### d) 15 minutos es el horizonte de la temperatura, no el del rendimiento

Respuesta real a un escalón de +1000 kg/h en el caudal de pasta:

| | t+15 | t+30 | t+45 | t+60 |
|---|---|---|---|---|
| Δ grasa_alpeorujo | +0,38 | +0,87 | +1,24 | +1,50 |

El tiempo de residencia es de ~50 min: la pasta que sale ahora entró antes de la
acción. A 15 min se gobierna **temperatura y calidad**; para gobernar el
agotamiento hace falta un horizonte de un tiempo de residencia (`--horizonte 45`).
Lo natural es un prescriptor de dos escalas.

## 5. Los prescriptores

Optimizador de una jugada sobre el modelo aprendido:

1. Estado actual → 400 combinaciones candidatas de acciones, dentro de los límites
   físicos y de los límites de movimiento por decisión.
2. Se sustituyen las columnas de acción futura y se predicen los cuatro objetivos.
3. Se elige la de menor coste multiobjetivo:

```
J = w_grasa·(grasa − objetivo)²                 pérdida de rendimiento
  + w_T·max(0, T − T_max)²                      calidad del aceite (asimétrico)
  + w_hum·(fuera de la ventana de la orujera)²
  + w_polif·max(0, polif_min − polifenoles)
  + w_mov·‖Δacción normalizada‖²                esfuerzo de control
```

Es el compromiso que ya señala el artículo: subir temperatura y tiempo de batido
mejora el agotamiento pero degrada el aceite. Los pesos están en `Objetivos`.

### 5.1 Doble horizonte (`prescriptor_doble.py`)

Consecuencia directa del hallazgo (d). Se entrenan dos juegos de modelos y cada
acción candidata se evalúa contra ambos:

- **15 min** → `T_pasta_s` y polifenoles → **restricciones** de calidad
- **45 min** → grasa y humedad del alpeorujo → **objetivo** de rendimiento

Así la restricción de temperatura llega a tiempo y el agotamiento se persigue en
el horizonte donde de verdad responde.

### 5.2 El optimizador explota el error del modelo

Con el modelo crudo, en un instante concreto propone bajar el caudal 600 kg/h y
predice que la grasa cae de 7,55 a 3,17 %. Imposible: el ensayo de escalón dice
que −600 kg/h no dan más de −0,75 pp. Es el fallo clásico de optimizar sobre un
modelo aprendido, y no lo detecta ninguna métrica de error.

`--modo` ofrece tres niveles de confianza:

| modo | qué hace | grasa 7,55 → |
|---|---|---|
| `aprendido` | modelo crudo | 3,17 (no creíble) |
| `acotado` | techo por ganancias medidas | 4,29 (se pega al techo) |
| `hibrido` | línea base del modelo + ganancias medidas | **5,90** (creíble) |

El modo **híbrido** es el defendible con pocos datos: separa lo que el modelo hace
bien (predecir la deriva si no tocas nada) de lo que hace mal (atribuir causas).
Las ganancias salen de un ensayo de escalón, que en planta son dos horas de trabajo:

| acción | ΔT/unidad (45 min) | Δgrasa/unidad (45 min) |
|---|---|---|
| `T_agua_ida` (°C) | +0,219 | −0,085 |
| `caudal_pasta` (kg/h) | −0,00076 | +0,00116 |
| `caudal_agua` (m³/h) | +0,696 | −0,302 |
| `ratio_agua_proceso` | 0 | +11,5 * |

\* no lineal: tiene un óptimo en U, el signo depende del punto de operación.
Para usarla en serio hay que sustituirla por el término físico del decánter.

## 6. Limitaciones honestas

- Los rangos coinciden con el artículo, pero la **dinámica interna es inventada**.
  Sirve para elegir arquitecturas y montar el pipeline, no para sacar conclusiones
  de proceso.
- Con 3 días, la humedad del alpeorujo no bate a "usar la última analítica".
- El prescriptor extrapola mal fuera del dominio de los datos; por eso las acciones
  candidatas están acotadas a movimientos pequeños.

## 7. Siguientes pasos sugeridos

1. **Modelos secuenciales**: con ≥15 días, comparar GBT contra GRU/LSTM o TCN sobre
   ventanas de 60 min. Aquí el GBT con retardos gana porque hay pocos datos.
2. **Modelo híbrido completo**: `--modo hibrido` es una primera versión con
   ganancias lineales. El paso siguiente es usar el balance de energía como capa
   física y dejar que la red aprenda solo el residuo.
3. **Cuantificar la incertidumbre** (regresión cuantílica o *ensembles*) y prescribir
   sobre el cuantil pesimista.
4. **Reentreno con laboratorio**, como en el artículo: cada nueva analítica reajusta
   el modelo, ponderando la información nueva frente a la acumulada.
5. **Plan de excitación real**: 2–3 turnos con escalones en manual sobre temperatura
   de agua y caudal de pasta. Es la inversión con mayor retorno de todo el proyecto.

## Ficheros

| Fichero | Contenido |
|---|---|
| `simulador_termobatidora.py` | Modelo físico: balances, cinética, decánter, operario, sensores |
| `generar_dataset.py` | Genera el CSV (`--dias`, `--excitacion`) |
| `modelo_15min.py` | Características, filtrado, entrenamiento y evaluación |
| `prescriptor.py` | Optimización multiobjetivo + comprobación de direccionalidad |
| `prescriptor_doble.py` | Doble horizonte (15/45 min) y los tres modos de confianza |
| `experimentos.py` | Los tres diagnósticos: escalado, observabilidad, causalidad |

Dependencias: `numpy`, `pandas`, `scikit-learn`, `joblib`.
