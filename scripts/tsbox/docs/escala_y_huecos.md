# "No veo las series": tres bugs y un malentendido

Capturas del usuario: señales que parecen líneas casi rectas, y en la segunda,
el panel entero teñido de rojo. Diagnóstico completo.

## Bug 1 — el marcador de huecos secuestraba el eje Y

`GapOverlay` pintaba rectángulos de **2·10¹² de alto** "para que siempre cubran
el panel". Dos consecuencias, ambas visibles:

- pyqtgraph incluye el `boundingRect` de los hijos en el autorange. Medido en el
  fichero real: el eje Y iba de **−1,13·10¹² a +1,13·10¹²** mientras los datos
  ocupaban de 68,6 a 83,0. **La señal ocupaba el 0% de la altura del panel.**
  Eso es la "línea continua" que veías.
- El rasterizador de Qt trabaja en punto fijo 26.6. Con rectángulos de 2·10¹²
  desborda y embadurna el panel entero. Eso es el rojo de la segunda captura.

**Arreglo:** el overlay implementa `dataBounds(ax)` devolviendo `[None, None]`
en Y, que es el gancho por el que pyqtgraph pregunta los límites — así el
autorange lo ignora. Un hueco es decoración, no un dato. Y el dibujo se hace en
altura unitaria (0..1) y se estira en cada repintado a la Y visible.

Resultado: la señal pasa de ocupar el **0%** a ocupar el **83%** del panel.

## Bug 2 — los huecos se agrupaban una sola vez, globalmente

Con 44.000 tramos y un tope de 20.000 bandas, la tolerancia de agrupado eran
390 s **fijos**. Al hacer zoom a un minuto seguías viendo las bandas gordas
calculadas para la vista completa, que ya no correspondían a ningún hueco real.

**Arreglo:** se guardan todos los tramos y en cada repintado se seleccionan solo
los del rango visible, agrupándolos a **resolución de píxel**. Como un mapa: no
dibuja cada calle cuando ves el país, pero al acercarte aparecen.

| Ventana visible | Huecos en pantalla | Bandas dibujadas |
|---|---|---|
| 90 días | 7.316 | franja fina |
| 1.000 min | 65 | 59 |
| 100 min | 8 | 8 |
| 20 min | 2 | 2 |

## Bug 3 — huecos densos que tapaban la señal

Aun agrupando bien, con el fichero completo a la vista los huecos cubren el
**83% del ancho**. Pintarlos a altura completa deja el gráfico rojo: técnicamente
correcto, visualmente inútil.

**Arreglo:** por encima del 25% de cobertura se pasa a una **franja fina en el
borde inferior**. Dice lo mismo sin comerse el gráfico. Es la diferencia entre
subrayar tres frases de una página y pintar la página entera de amarillo.

## Lo que pediste: escala Y automática

Ahora, por defecto, **cada panel ajusta su Y a la ventana visible** con un 6% de
margen (`setAutoVisible(y=True)`). Al hacer zoom en X, la Y se reajusta sola: la
señal siempre llena el panel.

Hay dos modos, en el menú Ver:

- **Y ajustada a lo visible** (defecto): la señal llena el panel siempre. Para
  ver la *forma* de la señal.
- **Y fija al rango completo** (`Ctrl+Y` para volver a ajustar): la escala no se
  mueve al hacer zoom. Para ver si un tramo está más alto o más bajo que el
  resto — con el modo automático eso es invisible, porque cada ventana se
  reescala a sí misma.

Los dos son necesarios y engañan de forma distinta. El automático te oculta el
nivel absoluto; el fijo te oculta la forma cuando la amplitud es pequeña.

## El malentendido: el diente de sierra no es el proceso

Las dos capturas muestran un diente de sierra muy regular. **No es el proceso
químico.** Volviste a cargar el fichero apilado. Las 12 primeras muestras que
dibuja el panel:

| i | reactor | x (min) | coolant |
|---|---|---|---|
| 0 | A_R1 | 0 | 79,15 |
| 1 | A_R2 | 0 | 78,45 |
| 2 | A_R3 | 0 | 79,85 |
| 3 | B_R1 | 0 | 72,09 |
| 4 | B_R2 | 0 | 72,84 |
| 5 | B_R3 | 0 | 70,58 |
| 6 | A_R1 | 1 | — |

El eje X no avanza durante seis muestras. Cada diente son los 6 reactores en el
mismo instante, recorridos en orden. Los reactores de la línea B están ~7
unidades por debajo de los de la A, y eso produce la caída de cada diente.

El salto medio entre muestras consecutivas es **3× mayor** apilado que separado.
Y las estadísticas de la cabecera lo confirman, con el mismo zoom:

| | μ | σ |
|---|---|---|
| Apilado | 76,02 | **4,105** |
| Solo A_R1 | 79,89 | **0,948** |

Ahora la cabecera avisa: **«eje X con valores repetidos»**, y no se reportan
"saltos" en ese modo porque `dt_mediano` no es el periodo de muestreo de nada.

Para verlo bien: en el diálogo de carga, **pivotar** por `reactor_id` (108
series limpias) o **filtrar** un reactor.
