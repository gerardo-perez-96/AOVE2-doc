# Stream forecasting: operates on data when arrives
## Requirements
- Continue ingestion: MQTT
- Minimal latency: pipeline completed in milliseconds
- Incremental updates: model update or retraining when new data enters. 

- desfases temporales entre máquinas (cuánto tarda en responder una señal ante el cambio de otra)
- ¿De cuántas máquinas se compone el sistema?
    - Limpiadora
    - Molino
    - Batidora
    - Decánter
    - Centrifugadora
    - Depósito
- ¿Sobre cuantas máquinas puede actuar el maestro de almazara? (variables actuables)
- Están los timestamps sincronizados entre los sensores?
- Cuales son las funciones objetivo que hay que maximizar? calidad (cómo se mide?) consumo energético, cantidad de aceite 
- Variables optimizables: cantidad de agua utilizada, velocidades de molienda, de decanter...
- Cómo hay que considerar la arquitectura al completo? 
    - la salida de una máquina puede afectar al rendimiento de la máquina siguiente? 
- Frecuencia de inserción de datos por sensor (para saber qué tiene que sostener la base de datos)

## PC requirements
- Entorno industrial: preparado para soportar temperaturas y carcasa resistente, tipo pc industrial
- Tarjeta de red con ancho de banda de 2.5 o 5 Gbps ¿Van a enviar video o son valores numéricos?
- RAM memory: 32-64 GB
- SAI por caidas eléctricas
- encendido por ethernet


## Real-time forecasting models which supports this:
- Exponential smoothing or Holt-Winters
- Stateful AR models with online updates
- Tree-based models like LightGBM or XGBoost, pre-trained and served
- Linear regression with lag features
- Naïve or ensemble hybrids that are easy to deploy
- LTSM???

## Training model with inputs and outputs


## Conceptos a revisar
- Engineered features: lags, rolling means, time-of-day encodings

## Equipamiento
- 32 GB RAM  (puede verse en un futuro si ampliable a 64, según tamaño de dataset)
- 2x 1TB NVMe (sistema y datasets, almacenamiento de modelos, DB...)
- Ubuntu (si no choca con requisitos de sensorización)
- GPU: cloud training? https://www.exoscale.com/gpu/3080ti/ (2€ por hora de uso. Si se puede desplegar el modelo en CPU puede ser una opción)
- Instalación de Tailscale o similar para acceso remoto. TeamViewer también por si acaso.


Si está en la almazara como tal, igual lo más apropiado es un pc industrial
- https://us.axiomtek.com/Default.aspx?MenuId=Solutions&FunctionId=SolutionView&ItemId=2929&Title=Guide+to+Edge+Computing+and+Choosing+the+Right+Hardware

https://huggingface.co/blog/daya-shankar/best-laptop-for-artificial-intelligence-and-ds
