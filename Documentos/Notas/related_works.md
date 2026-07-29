# A Cyber–Physical System Based on Digital Twin and 3D SCADA for Real-Time Monitoring of Olive Oil Mills (2024)

Implementación de sistema ciberfísico con integración de realidad virtual y 3D SCADA para monitorizar y optimizar una almazara. Uso de protocolo Open Platform Communication United Architecture (OPC-UA) para comunicación bidireccional y en tiempo real. La interconexión del gemelo digital con SCADA permite simular el comportamiento del modelo digital.

Tienen una almazara ya preparada que utiliza UNIFIK, un servidor OPC-UA y drivers distintos para cada protocolo

# A Digital Twin Architecture for Real-Time Supervision of an Olive Oil Production Mill

Implementación de un gemelo digital basado en la normativa ISO 23247 (leer) para toma de decisiones y simulación. Mejora el paper anterior añadiendo la posibilidad de simular, analizar los datos y dar capacidad de toma de decisiones. Añade un diseño modular que es facilmente intercambiable

Tres capas: 
- Recolección de datos y control de dispositivos 
    - Adquisición y preprocesado de datos
    - Recolección de datos:
        - Sensores para monitorizar características relevantes, conectados a microcontroladores.
        - Datos formateados y enviados con MQTT o HTTPS, tanto datos del densor como los datos de control.
    - Control de dispositivos:
        - Los microcontroladores y actuadores llevan a cabo las funciones requeridas
- Núcleo
    - Comunicación a través de MQTT y HTTPS.
    - Los modelos digitales capturan características relevantes de los dispositivos para tener una representación digital precisa.
    - Todos los datos se almacenan en una base de datos
- Usuario

Implementación:
    - Eclipse Ditto: framework open source para levantar gemelos digitales. Almacena cada dispositivo en una estructura denominada Thing que almacena diversos parámetros en jormato JSON:
        - Identificador
        - Política de acceso
        - Características específicas del sensor:
            - Unidad de medida
            - Valores umbral, categorizables como críticos o no
            
    - influxDB
    - React para la base de datos

