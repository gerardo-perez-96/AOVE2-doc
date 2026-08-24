# El fichero que congelaba la ventana: diagnóstico completo

Fichero real: `chemical_process_timeseries.csv`, 200 MB, **777.600 filas × 21
columnas**. En un i7-10875H con 32 GB, Windows mostraba "No responde".

777.600 filas no son muchas. El tamaño nunca fue el problema.

## Lo que se midió

| Sospechoso | Coste | ¿Culpable? |
|---|---|---|
| `read_csv` con `engine="python"` | 16,3 s | parcialmente |
| `read_csv` con `engine="c"` | 2,4 s | — |
| Parseo de fechas | 0,14 s | no |
| `setData` de 777k puntos | 0,03 s | no |
| **44.161 `LinearRegionItem` por panel** | **~32 s/panel** | **sí** |
| **`median_step()` dentro de un bucle** | **~70 s/panel** | **sí, el mayor** |

Total: **152 segundos para pintar UN panel**. Con 18 series, cuarenta y cinco
minutos. Eso es lo que Windows llamaba "No responde".

---

## Causa 1: un bug O(n·m) escondido en tres líneas

`nan_intervals()` recorría los tramos de datos faltantes y, para los tramos
pegados a un extremo, llamaba a `median_step(x)` **dentro del bucle**.
`median_step` calcula `np.diff` y `np.median` sobre el **array completo**.

Con 28.667 tramos: 28.667 medianas de 777.600 elementos. El perfilador lo dejó
claro — 45 s en `median_step`, 16 s en `np.diff`, 10 s en `np.median`.

Analogía: pesar cada tomate volviendo a tarar la báscula entera cada vez. Con
tres tomates no lo notas; con veintiocho mil, cierras la tienda.

**Arreglo:** el paso se calcula una vez y se pasa como argumento; el bucle
entero está vectorizado. De 72 s a 0,2 s. Test de regresión:
`test_nan_intervals_no_es_cuadratico`, que falla si vuelve a superar 1 s.

## Causa 2: 44.161 objetos gráficos por panel

Cada tramo de NaN creaba un `LinearRegionItem`. Medido: 0,81 s por cada 1.000
items → 36 s para 44.161. Y luego la escena queda inmanejable para siempre: Qt
tiene que gestionar 44.000 bounding boxes en cada repintado.

Analogía: no es lo mismo imprimir una página con 44.000 puntos que pegar 44.000
pegatinas a mano.

**Arreglo:** `GapOverlay`, un único `QGraphicsObject` que pinta todos los tramos
en un `QPicture`. Por encima de 20.000 tramos se agrupan los contiguos, porque
44.000 franjas no se ven como 44.000 franjas: se ven como una mancha roja.

## Causa 3, la de fondo: el fichero no es una serie temporal

```
timestamp            reactor_id  reactor_temp
2024-01-01 00:00:00  A_R1        181.1
2024-01-01 00:00:00  A_R2        190.4    <- MISMO instante, otro reactor
2024-01-01 00:00:00  A_R3        188.7
```

**Seis reactores apilados** (`A_R1…B_R3`). 777.600 filas para 129.600 instantes
distintos: cada instante aparece 6 veces, 648.000 valores de X duplicados.

Esto no es solo un problema de velocidad, es un problema de **corrección**:

- El salto medio entre muestras consecutivas de `reactor_temp` es **5,72**,
  cuando el rango real de la señal es 24,56. Casi una cuarta parte del rango
  entero en cada paso. No estás viendo el proceso, estás viendo el entrelazado.
- Las estadísticas mezclan las seis máquinas:

  | | μ | σ |
  |---|---|---|
  | Apilado (lo que veías) | 188,0 | **6,8** |
  | Solo A_R1 | 182,6 | **0,62** |
  | Solo B_R3 | 195,8 | **0,73** |

  La σ global es **diez veces** la real. Casi toda esa "variabilidad" es la
  diferencia entre reactores, no la variabilidad de ningún reactor. Y μ=188 es
  un valor que ninguno de los seis tiene.
- La detección de huecos es basura: con X duplicado, `dt_mediano` sale 60 s
  cuando entre muestras del *mismo* reactor hay 6 minutos.
- Y la correlación entre dos señales apiladas mide sobre todo la diferencia
  entre reactores, no la relación física entre variables.

**Arreglo:** `longformat.detect()` lo reconoce en la vista previa (500 filas,
sin leer el fichero) y el diálogo de carga ofrece tres opciones:

- **Pivotar** (recomendado): una columna por (señal × entidad). 18 señales × 6
  reactores = **108 series** sobre un eje X limpio de 129.600 muestras. Es lo
  que permite comparar reactores entre sí.
- **Filtrar**: cargar un solo reactor. Menos RAM, una serie temporal de verdad.
- **En crudo**: se permite, pero con un aviso explícito de que las medias y los
  huecos van a mentir. No se bloquea nada — a veces quieres ver el fichero tal
  cual.

## Resultado

| | Carga | Pintar 6 paneles |
|---|---|---|
| Antes | no terminaba | **152 s por panel** |
| Crudo (arreglado) | 3,2 s | **0,74 s** |
| Solo A_R1 | 2,8 s | **0,66 s** |
| Pivotado (108 series) | 2,9 s | **0,87 s** |

## La lección que se repite

Las tres causas eran invisibles desde fuera y ninguna tenía que ver con "el
fichero es grande". Las tres aparecieron en cuanto se perfiló en vez de suponer.
El primer instinto —"optimiza los scripts", "el CSV pesa mucho"— apuntaba al
sitio equivocado en los tres casos.
