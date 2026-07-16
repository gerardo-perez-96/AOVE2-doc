# Dudas

## Desarrollo del modelo
- Cuales son las funciones objetivo que hay que maximizar? calidad (cómo se mide?) consumo energético, cantidad de aceite 
- Variables optimizables (parámetros de control): cantidad de agua utilizada, velocidades de molienda, de decanter...
    - Tiempos controlables en las máquinas de la almazara. Pueden afectar al tiempo de entrenamiento
- Cómo hay que considerar la arquitectura al completo? (máquinas por separado, un modelo para toda la almazara) 
    - la salida de una máquina puede afectar al rendimiento de la máquina siguiente? 
    - si son varios modelos en bloque, hace falta tener para cada uno un input y un output que vayan al siguiente
- Variables a optimizar en la literatura
    - paste temperature in the thermomixer
    - flow of addition water
    - temperature of addition water
    - paste flow
    - paste moisture and mixing (residence) time

## Sensorización
- Frecuencia de inserción de datos por sensor (para saber qué tiene que sostener la base de datos)

## Instalación PC
- Hay sincronización de reloj entre los datos? (NTP/PTP)
- Si se cae la red hay algún buffer intermedio?

## Maquinaria
- Modelos de las máquinas (batidora, molino...). Se pueden sacar datos de los PLCs?
- Tiempo característico de los procesos involucrados (para seleccionar el tamaño de contexto temporal)
- Batidora de estancia contínua o por lotes?

## Visualización de datos
- Va a ir el modelo 3D integrado junto a la visualización de los datos?