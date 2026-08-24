# Por qué el pan va a tirones, y qué hacer (con números)

Todo medido sobre `chemical_process_timeseries.csv` filtrado a A_R1, 18 paneles,
ventana de 3 días, simulando un arrastre de 30-40 frames.

**Aviso sobre las cifras absolutas:** están tomadas con el rasterizador software
de Qt en modo offscreen. En tu Windows con GPU los FPS serán bastante mejores.
Lo que se mantiene es el **reparto relativo**, que es lo que decide qué merece
la pena arreglar.

## Tu hipótesis: "son procesos que no están paralelizados"

No. Un pan no es trabajo CPU paralelizable, es **rasterizado por frame**. No hay
un cálculo pesado esperando a repartirse entre núcleos: hay 18 paneles pidiendo
a Qt que redibuje 18 gráficos, 18 ejes y 18 capas de huecos, sesenta veces por
segundo. Meter hilos ahí no reparte nada — Qt no permite pintar desde hilos
secundarios en el hilo de GUI, y el trabajo ya está en C++.

La causa real es **cuántas cosas se repintan por frame**, no en cuántos núcleos.

## Lo que se arregló en esta sesión

| Cambio | Qué era | Efecto |
|---|---|---|
| `QPicture.play` → `drawRects` por lote | El overlay de huecos reejecutaba una secuencia grabada de comandos en cada frame | 0,057 → 0,014 ms por repintado (**4×**) |
| `ViewBox` del eje derecho perezoso | Se creaba en los 18 paneles y todos iban `setXLink`-ados, aunque no tuvieran nada dentro | cascada de señales 18.414 → 11.362 por pan |
| Eje X solo en el panel de abajo | Los 18 paneles redibujaban el **mismo** eje de fechas cada frame | `AxisItem.paint` era el **27%** del tiempo de frame |
| Sin padding 3× en el overlay | Se rasterizaban rectángulos tres veces más altos que el viewport | menos área a rellenar |

## El dato que decide todo: escalado con el número de paneles

| Paneles visibles | FPS | ms/frame |
|---|---|---|
| 1 | **50,7** | 19,7 |
| 2 | 29,8 | 33,6 |
| 4 | 20,2 | 49,4 |
| 8 | 14,7 | 68,2 |
| 18 | **13,1** | 76,5 |

Y el marcado de huecos, con 18 paneles: **12,9 FPS con, 17,7 sin** (+37%).

**Conclusión práctica, hoy, sin tocar código:** con 4 paneles va fluido; con 18
no. Trabaja con 3-6 series a la vista y usa el árbol para plegar el resto. Si
necesitas los 18, desactiva "Marcar huecos" mientras navegas.

## Pregunta 1: ¿pasarlo a C++?

Reparto real del tiempo de un pan de 30 frames:

| | tiempo | % |
|---|---|---|
| **Qt / C++ puro** (rasterizado, señales) | 1,714 s | **57,5%** |
| pyqtgraph (Python) | 0,826 s | 27,7% |
| Código de tsbox (Python) | 0,157 s | **5,3%** |
| numpy (wrapper Python) | 0,023 s | 0,8% |
| Resto | 0,260 s | 8,7% |

**El 57,5% del tiempo ya es C++ y no se movería ni un milisegundo.** Reescribir
*todo* el Python — pyqtgraph incluido — tiene un techo teórico del **42%**, o sea
pasar de 13 a ~22 FPS. Y "reescribir pyqtgraph en C++" no es un proyecto, es
adoptar QCustomPlot o Qt Charts y rehacer la aplicación entera: meses.

Reescribir **solo tu código** (el 5,3%) daría como mucho un 5% más. Nada.

Analogía: tienes un coche que va lento y el 57% del tiempo está parado en
semáforos. Cambiar el motor no toca los semáforos.

**Veredicto: no.** El coste es de meses y el techo es de 1,7×. Hay cosas que dan
más por muchísimo menos:

1. **Menos paneles a la vista.** 1 panel vs 18 es 50,7 vs 13,1 FPS: **3,9×**,
   gratis, hoy. Más que el techo entero de la reescritura.
2. **OpenGL.** `pg.setConfigOptions(useOpenGL=True)` mueve el rasterizado a la
   GPU. No se puede medir aquí (offscreen no tiene GPU), pero en tu portátil es
   la primera cosa que probar y son dos líneas.
3. **Downsampling más agresivo** durante el arrastre y detalle completo al
   soltar, que es lo que hacen los visores profesionales.

## Pregunta 2: ¿optimizar los scripts?

Ya está hecho lo que tenía retorno claro (la tabla de arriba). Lo que queda de
tu código Python es el 5,3% del frame: **optimizarlo más no se va a notar**.

Lo que sí queda, por orden de retorno:

1. **Probar `useOpenGL=True`.** Dos líneas, potencialmente el mayor salto.
2. **Repintado perezoso durante el arrastre:** dibujar los huecos y las
   estadísticas solo al soltar el ratón, no en cada frame. Vale ~35% con 18
   paneles.
3. **Un solo `PlotItem` con varios ejes Y** en lugar de N `PlotWidget`. Elimina
   de raíz la cascada de sincronización y los N ejes. Es el rediseño correcto
   si de verdad necesitas 18 señales simultáneas — un par de días, no meses, y
   da más que el C++.

## La comprobación que deberías hacer antes de nada

Abre con **3 paneles** en vez de 18 y arrastra. Si va fluido, no tienes un
problema de rendimiento: tienes un problema de cuántas señales miras a la vez, y
eso se arregla con el árbol de series, no con C++.
