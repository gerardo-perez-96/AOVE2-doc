# Rendimiento: qué se midió y qué se arregló

Todo lo de aquí está medido, no supuesto. Fichero de prueba: **3.000.000 filas
× 13 columnas, CSV de 434 MB** (y su equivalente parquet de 183 MB).

## El diagnóstico

Lo primero fue medir cada etapa por separado. Resultado:

| Etapa | Coste | ¿Culpable? |
|---|---|---|
| `pd.read_csv` con `engine="python"` | **OOM / no termina** | **sí** |
| `pd.read_csv` con `engine="c"` | 7.8 s | referencia |
| parseo de fechas `format="mixed"` | minutos en 3M filas | **sí** |
| leer todo y *luego* truncar a 200k | lees 3M para tirar 2.8M | **sí** |
| `build_x` + `sanitize_axis` | 1.3 s | no |
| construir 12 paneles y pintarlos | 0.94 s | **no** |
| `setData` de 3M puntos en pyqtgraph | 0.12 s | **no** |
| `missing()` / `stats()` por serie | <0.02 s | no |

**El pintado nunca fue el problema.** `setClipToView(True)` +
`setDownsampling(mode="peak", auto=True)` ya estaban puestos y hacen su trabajo:
pyqtgraph solo rasteriza los puntos del rango visible. Si me hubiera puesto a
"optimizar los gráficos" habría perdido un día para ganar 0.1 s.

Por eso se mide antes de tocar nada. Optimizar por intuición es apostar.

## Los arreglos

### 1. Parser en C en vez de en Python (`sniff_sep`)

Pasar `sep=None` a pandas activa el parser escrito en Python para que adivine el
delimitador. En 300k filas: **3.79 s y 495 MB** frente a **0.58 s y 188 MB** del
parser en C. En 3M filas, eso es quedarse sin memoria.

Ahora el delimitador se detecta leyendo 64 KB de cabecera y se le pasa explícito
a `read_csv` con `engine="c"`. Hay un test (`test_no_usa_el_parser_de_python`)
que revienta si alguien reintroduce esto.

### 2. Empujar los filtros al lector (`plan_sampling`)

Antes: leer 3M filas → `apply_sampling` → quedarse con 200k. Leías el fichero
entero para tirar el 93%. Es ir a por la compra semanal para cocinar una cena.

Ahora `plan_sampling` traduce tu límite a argumentos del lector:
- **truncar** → `nrows=200_000`. pandas para de leer ahí. **0.34 s.**
- **decimar** → lectura por trozos de 1M filas, `iloc[::step]` en cada trozo,
  nunca más de un trozo en RAM. La rejilla es global (`(step - seen % step) % step`),
  no reinicia en cada chunk: sin eso se duplican muestras en las costuras. Hay
  test para eso.
- **columnas** → `usecols=` en CSV, `columns=` en parquet. Si pides 3 de 50, se
  leen 3.

### 3. Formato de fecha fijo (`_to_datetime_fast`)

`format="mixed"` reintenta el formato **fila por fila**. Se deduce el formato de
la primera fila válida, se aplica fijo (ruta vectorizada en C) y solo se cae a
`"mixed"` si falla en más del 1% de las filas.

El eje X se queda **siempre en float64**, pase lo que pase con `float32`: en
float32 dos instantes separados 100 ms colapsan en el mismo valor y el zoom deja
de funcionar. Hay test.

### 4. float32 opcional

Mitad de RAM en los datos. 7 dígitos significativos sobra para dibujar y para
estadísticas. Desactívalo si tus valores superan ~1e7 y los decimales importan.

### 5. Carga en un hilo con progreso

Esto no acelera nada, arregla otra cosa: **tú no dijiste "tarda", dijiste "está
atascado"**. Qt no procesa eventos mientras pandas trabaja, así que la ventana
se congelaba sin pintar ni un píxel y no había forma de distinguir 10 segundos
de trabajo de un cuelgue.

Ahora `LoadWorker` corre en un `QThread`, reporta progreso con filas leídas y se
puede cancelar. Verificado: la UI procesa ~16 eventos/s durante la carga.

### 6. Estimación de coste ANTES de leer

El diálogo de carga abre en **0.04 s** sobre un CSV de 434 MB (solo lee la
cabecera y 1 MB para estimar filas) y te dice cuántas filas hay y cuánta RAM te
va a costar la combinación que has elegido. Amarillo por encima de 600 MB, rojo
por encima de 2 GB.

## El resultado

| Configuración | Antes | Ahora |
|---|---|---|
| Todo (3M × 12) | OOM | **9.9 s**, 261 MB |
| Solo 3 columnas | OOM | **4.8 s** |
| Límite 200k, truncando | OOM | **0.34 s** |
| Límite 200k, decimando | OOM | **5.2 s** |
| 3 columnas + 200k decimando | OOM | **3.5 s** |

Nota sobre decimar: cuesta más que truncar porque hay que **leer** las 3M filas
para poder quedarte con 1 de cada 15. No hay atajo — el CSV no tiene índice. Si
vas a explorar el mismo fichero muchas veces, conviértelo a parquet una vez
(`df.to_parquet`): pesa menos de la mitad, guarda los tipos y permite leer
columnas sueltas de verdad.

## Si vuelve a ir lento

En este orden:

1. ¿Cuántas filas dice el diálogo? Si son decenas de millones, no es un problema
   de código: limita muestras o pásate a parquet.
2. `pytest tests/test_loading_perf.py` — si falla alguno, hay una regresión
   concreta y el test te dice cuál.
3. Perfila con `python -X importtime` y `cProfile` sobre `Session.open`. No
   optimices nada que no aparezca en el perfil.

**Límite conocido:** todo esto asume que el fichero cabe en RAM una vez
filtrado. Para ficheros que no caben ni así, la solución no es optimizar más
este código — es no cargarlo entero: índice por bloques y lectura bajo demanda
según el zoom. Es un rediseño, no un ajuste, y no está hecho.
