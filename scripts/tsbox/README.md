# tsbox

Visor y anotador de series temporales. PySide6 + pyqtgraph.

## Instalar y arrancar

    pip install -r requirements.txt
    python -m tsbox

## Qué hace

**Carga** CSV / TSV / Parquet. Al abrir eliges qué columnas cargar, qué va en el
eje X (una columna, numérica o de fechas, o el índice de muestra) y un límite de
muestras (truncar o decimar — decimar introduce aliasing, por eso no es el
default). Los filtros se empujan al lector: si pides 3 de 50 columnas y 200.000
de 3.000.000 de filas, se leen 3 columnas y 200.000 filas, no el fichero entero.
El diálogo te dice cuántas filas hay y cuánta RAM va a costar **antes** de leer.
La carga va en un hilo con barra de progreso y botón de cancelar.

Un CSV de 434 MB (3M × 13) pasa de quedarse sin memoria a cargar en 9,9 s
completo, o 0,34 s limitado a 200.000 muestras. El desglose de qué se midió y
qué se arregló está en `docs/rendimiento.md`.

**Formato largo.** Si el fichero apila varias entidades sobre el mismo eje de
tiempo (6 reactores, 40 sensores…), tsbox lo detecta en la vista previa y te
ofrece pivotar (una serie por señal × entidad), filtrar una sola entidad, o
cargar en crudo con aviso. Cargarlo apilado sin darte cuenta hace que la curva
salte entre entidades en cada muestra y que las medias describan a ninguna. Ver
`docs/formato_largo.md`.

**Rendimiento del pan.** El coste va con el número de paneles VISIBLES: 1 panel
≈50 FPS, 18 paneles ≈13 FPS. Si el arrastre va a tirones, pliega series en el
árbol o desactiva "Marcar huecos" (+37%). Medidas y el análisis de si compensa
pasar a C++ (spoiler: no) en `docs/rendimiento_pan.md`.

**Escala Y.** Por defecto cada panel ajusta su Y a la ventana visible con 6% de
margen: la señal llena el panel y al hacer zoom la escala se reajusta sola. En el
menú Ver puedes fijarla al rango completo de la serie (para ver si un tramo está
más alto que el resto, que con el automático es invisible). `Ctrl+Y` reajusta.

**Huecos.** Se marcan sobre la serie, agrupados a resolución de píxel: al hacer
zoom se separan en huecos individuales. Si cubren más del 25% del ancho visible
se dibujan como franja fina abajo en vez de tapar el gráfico. Ver
`docs/escala_y_huecos.md`.

**Paneles.** Una serie por panel, uno debajo de otro. Se reordenan arrastrando
**en la lista de la izquierda**, no sobre el gráfico: el canvas está reservado
para navegar y anotar. Rueda = zoom. `Ctrl+L` sincroniza el eje X de todos los
paneles; `Ctrl+Shift+L` propaga el zoom del panel activo al resto.

**Modos de edición** (`Ctrl+1/2/3`, el cursor cambia): navegar, marcar región
(clic y arrastre), poner marca (clic). Un solo gesto por modo, sin ambigüedad.
`Ctrl+Z` / `Ctrl+Y` deshacen todo.

**Datos faltantes.** Se distingue el NaN (la muestra existe, el valor no) del
salto en el eje X (la muestra no existe). Ambos se sombrean sobre la serie.
Menú *Análisis → Informe de datos faltantes* lista los timestamps exactos.

**Árbol de series.** La lista de la izquierda es un árbol: cada serie de origen
es una raíz, sus derivadas cuelgan debajo plegadas por defecto y tienen SU
PROPIO panel (no comparten gráfico con eje Y secundario salvo que actives
también "superponer"). Expandir/plegar el nodo muestra u oculta sus paneles sin
tocar los checkboxes.

**Series derivadas.** Media móvil, desviación móvil, derivada y desplazamiento.
Se guarda la **receta**, no los datos: el JSON pesa kilobytes y puedes cambiar
la ventana sin regenerar nada. Opción de superponerlas sobre la original con
eje Y secundario (necesario: la derivada no comparte rango con la señal).

**Estadísticas** (μ, σ, var, min, max) de la **ventana visible**, en la cabecera
del panel y opcionalmente como líneas horizontales.

**Análisis** (`Ctrl+A`): histogramas con detección de modos, matriz de
correlación con significancia, ACF/PACF y correlación cruzada con desfase.
Ver `docs/estadistica.md` — hay decisiones ahí que cambian las conclusiones.

**Guardado.** Sidecar `<fichero>.tsbox.json` junto al original. Autoguardado
cada 30 s, escritura atómica (`.tmp` + `os.replace`) y `.bak` rotativo. El JSON
guarda hash rápido, tamaño y mtime del origen: si no cuadran al abrir, avisa en
vez de mezclar anotaciones con datos que no les corresponden.

## Estructura

    tsbox/
      model.py       dataclasses del proyecto. Sin Qt.
      loader.py      lectura, construcción del eje X, saneado. Sin Qt.
      transforms.py  recetas y estadísticas de ventana. Sin Qt.
      gaps.py        NaN vs saltos de muestreo. Sin Qt.
      analysis.py    histogramas, correlación, ACF/PACF, CCF. Sin Qt.
      store.py       JSON atómico + verificación de origen. Sin Qt.
      session.py     estado en memoria y caché. Sin Qt.
      viewbox.py     máquina de estados de los gestos del ratón.
      panel.py       un panel = una serie.
      dialogs.py     diálogos de carga y de series derivadas.
      analysis_ui.py ventana de análisis.
      commands.py    QUndoCommand de cada mutación.
      mainwindow.py  ensamblado.

Todo lo que no depende de Qt está testeado sin abrir ventana:

    pytest tests/
